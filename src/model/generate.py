"""generate.py

Autoregressive MIDI generation from a trained MidiTransformer checkpoint.

Usage:
    # Generate from a custom MIDI prompt (auto-detects dominant instrument)
    python src/model/generate.py \
        --checkpoint checkpoints/best.pt \
        --midi_prompt mysong.mid \
        --prompt_len 512 \
        --output continuation.mid

    # Generate from dataset with instrument conditioning
    python src/model/generate.py \
        --checkpoint checkpoints/best.pt \
        --instrument piano \
        --prompt_len 256 \
        --output generated.mid

The script:
  1. Loads the checkpoint and rebuilds the model.
  2. Takes prompt tokens: either from a MIDI file (--midi_prompt, auto-detects instrument)
     or from the dataset (--prompt, optionally conditioned on --instrument).
    3. Autoregressively samples as many new tokens as safely fit in model context.
  4. Converts the combined sequence back into MIDI notes and writes a .mid file.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Callable

import torch
import mido

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "preprocessing"))
from model import MidiTransformer
from extract_notes import extract_tracks_from_midi
from tokenize_notes import tokenize_song


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Return a single token index sampled from the top-k logits."""
    if top_k < logits.size(-1):
        values, _ = torch.topk(logits, top_k)
        threshold = values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def infer_allowed_pitch_classes(prompt_tokens: list, vocab: dict, top_pcs: int = 7) -> set:
    """Infer a prompt key profile by selecting the top-N most frequent pitch classes."""
    note_on_off = int(vocab["note_on_offset"])
    note_off_off = int(vocab["note_off_offset"])
    counts = [0] * 12
    for tok in prompt_tokens:
        if note_on_off <= tok < note_off_off:
            pitch = tok - note_on_off
            counts[pitch % 12] += 1

    total = sum(counts)
    if total == 0:
        return set()

    order = sorted(range(12), key=lambda pc: counts[pc], reverse=True)
    k = max(1, min(int(top_pcs), 12))
    return set(order[:k])


def build_key_penalty_mask(
    vocab_size: int,
    note_on_offset: int,
    allowed_pcs: set,
    penalty: float,
    device: torch.device,
) -> torch.Tensor:
    """Create a mask where out-of-key NOTE_ON logits get subtracted by `penalty`."""
    mask = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    if not allowed_pcs or penalty <= 0:
        return mask

    for pitch in range(128):
        if (pitch % 12) not in allowed_pcs:
            mask[note_on_offset + pitch] = float(penalty)
    return mask


def compute_prompt_density(prompt_tokens: list, vocab: dict) -> float:
    """Compute notes per second from prompt tokens; biased toward prompt's natural note density."""
    ts_off = vocab["time_shift_offset"]
    ts_step = vocab["time_shift_step_ms"]
    note_on_off = vocab["note_on_offset"]
    note_off_off = vocab["note_off_offset"]

    current_time_ms = 0.0
    note_on_count = 0

    for tok in prompt_tokens:
        if ts_off <= tok < ts_off + vocab["time_shift_bins"]:
            k = tok - ts_off
            current_time_ms += (k + 1) * ts_step
        elif note_on_off <= tok < note_off_off:
            note_on_count += 1

    total_sec = current_time_ms / 1000.0
    if total_sec < 0.1 or note_on_count == 0:
        return 1.5  # default density (what model learned from LMD)
    return note_on_count / total_sec


def build_timeshift_density_penalty(
    vocab_size: int,
    prompt_density: float,
    ts_offset: int,
    ts_bins: int,
    ts_step_ms: float,
    penalty_base: float,
    device: torch.device,
) -> torch.Tensor:
    """Create penalty mask for TIME_SHIFT tokens scaled by prompt density.

    Fast prompt (high density) -> penalize long gaps more.
    Slow prompt (low density) -> penalize long gaps less.
    """
    mask = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    if penalty_base <= 0 or prompt_density <= 0:
        return mask

    # LMD average density is ~1.5 notes/sec (model learns this distribution)
    reference_density = 1.5
    density_ratio = prompt_density / reference_density
    density_ratio = min(max(density_ratio, 0.1), 10.0)  # clamp to avoid extremes

    # Use sqrt to moderate the scaling: less aggressive than linear
    # This prevents overshoot while still biasing toward matching prompt tempo
    density_scale = (density_ratio ** 0.5 - 1.0) * 0.5 + 1.0  # smooth scaling, clamped near 1.0

    # Convert prompt density into target inter-onset interval (IOI).
    target_ioi_ms = 1000.0 / max(prompt_density, 1e-6)

    # Higher density ratio = prompt is faster = apply stronger penalties to long gaps.
    # Use a hinge penalty (only above target gap) and cap it to prevent instability.
    max_penalty = 4.0
    for k in range(ts_bins):
        gap_ms = (k + 1) * ts_step_ms
        # No penalty for short/target-aligned gaps; smoothly increase for longer gaps.
        rel_excess = max(0.0, (gap_ms - target_ioi_ms) / max(target_ioi_ms, 1e-6))
        gap_penalty = penalty_base * density_scale * rel_excess
        gap_penalty = min(gap_penalty, max_penalty)
        mask[ts_offset + k] = gap_penalty

    return mask


@torch.no_grad()
def generate(
    model: MidiTransformer,
    prompt: list,
    gen_len: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    key_penalty_mask: torch.Tensor | None = None,
    timeshift_penalty_mask: torch.Tensor | None = None,
) -> list:
    model.eval()
    context_len = model.context_len
    tokens = list(prompt)

    for _ in range(gen_len):
        # Keep only the last context_len tokens as input
        idx = torch.tensor(
            tokens[-context_len:], dtype=torch.long, device=device
        ).unsqueeze(0)   # (1, T)

        logits = model(idx)          # (1, T, V)
        logits = logits[0, -1, :]   # last position  (V,)
        logits = logits / max(temperature, 1e-6)
        if key_penalty_mask is not None:
            logits = logits - key_penalty_mask
        if timeshift_penalty_mask is not None:
            logits = logits - timeshift_penalty_mask

        next_token = sample_top_k(logits, top_k).item()
        tokens.append(next_token)

    return tokens


# ---------------------------------------------------------------------------
# Token -> MIDI conversion
# ---------------------------------------------------------------------------

def tokens_to_midi(tokens: list, vocab: dict, output_path: Path, min_note_ms: float = 25.0):
    """Convert a flat token sequence back into a MIDI file."""
    NOTE_ON_OFF  = vocab["note_on_offset"]
    NOTE_OFF_OFF = vocab["note_off_offset"]
    TS_OFF       = vocab["time_shift_offset"]
    TS_STEP      = vocab["time_shift_step_ms"]   # ms per step
    VEL_OFF      = vocab["velocity_offset"]
    VEL_BINS     = vocab["velocity_bins"]

    TICKS_PER_BEAT = 480
    TEMPO          = 500_000   # 120 BPM in microseconds/beat
    MS_PER_TICK    = (TEMPO / 1_000) / TICKS_PER_BEAT  # ms per tick

    def ms_to_ticks(ms: float) -> int:
        return max(0, int(round(ms / MS_PER_TICK)))

    min_note_ticks = ms_to_ticks(max(0.0, float(min_note_ms)))

    mid  = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))

    current_tick = 0
    last_event_tick = 0
    pending_velocity = 64   # default if no VELOCITY token precedes a NOTE_ON
    active_notes = {}       # pitch -> start_tick

    def delta(tick: int) -> int:
        nonlocal last_event_tick
        tick = max(tick, last_event_tick)
        d = tick - last_event_tick
        last_event_tick = tick
        return d

    current_time_ms = 0.0

    for tok in tokens:
        if TS_OFF <= tok < TS_OFF + vocab["time_shift_bins"]:
            # TIME_SHIFT: bin k -> advance (k+1) * TS_STEP ms
            k = tok - TS_OFF
            current_time_ms += (k + 1) * TS_STEP
            current_tick = ms_to_ticks(current_time_ms)

        elif VEL_OFF <= tok < VEL_OFF + VEL_BINS:
            k = tok - VEL_OFF
            # Map bin index back to approximate MIDI velocity
            pending_velocity = max(1, min(127, int((k + 0.5) * 127 / VEL_BINS)))

        elif NOTE_ON_OFF <= tok < NOTE_ON_OFF + 128:
            pitch = tok - NOTE_ON_OFF
            track.append(mido.Message(
                "note_on", channel=0, note=pitch,
                velocity=pending_velocity,
                time=delta(current_tick),
            ))
            active_notes[pitch] = current_tick

        elif NOTE_OFF_OFF <= tok < NOTE_OFF_OFF + 128:
            pitch = tok - NOTE_OFF_OFF
            if pitch in active_notes:
                off_tick = current_tick
                if min_note_ticks > 0:
                    off_tick = max(off_tick, active_notes[pitch] + min_note_ticks)
                track.append(mido.Message(
                    "note_off", channel=0, note=pitch,
                    velocity=0,
                    time=delta(off_tick),
                ))
                del active_notes[pitch]

    # Close any notes still open at end of sequence
    for pitch in list(active_notes.keys()):
        off_tick = current_tick
        if min_note_ticks > 0:
            off_tick = max(off_tick, active_notes[pitch] + min_note_ticks)
        track.append(mido.Message(
            "note_off", channel=0, note=pitch,
            velocity=0, time=delta(off_tick),
        ))

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(output_path)
    print(f"MIDI saved -> {output_path}")


def _track_end_tick(track: mido.MidiTrack) -> int:
    """Return absolute end tick of a MIDI track."""
    tick = 0
    for msg in track:
        tick += msg.time
    return tick


def _tokens_to_track(
    tokens: list,
    vocab: dict,
    start_tick: int = 0,
    tempo: int = 500_000,
    channel: int = 0,
    min_note_ms: float = 25.0,
) -> mido.MidiTrack:
    """Convert tokens to a single MIDI track, offset by start_tick."""
    NOTE_ON_OFF  = vocab["note_on_offset"]
    NOTE_OFF_OFF = vocab["note_off_offset"]
    TS_OFF       = vocab["time_shift_offset"]
    TS_STEP      = vocab["time_shift_step_ms"]
    VEL_OFF      = vocab["velocity_offset"]
    VEL_BINS     = vocab["velocity_bins"]

    TICKS_PER_BEAT = 480
    MS_PER_TICK    = (tempo / 1_000) / TICKS_PER_BEAT

    def ms_to_ticks(ms: float) -> int:
        return max(0, int(round(ms / MS_PER_TICK)))

    min_note_ticks = ms_to_ticks(max(0.0, float(min_note_ms)))

    track = mido.MidiTrack()
    current_tick = start_tick
    last_event_tick = 0
    pending_velocity = 64
    active_notes = {}
    current_time_ms = 0.0

    def delta(abs_tick: int) -> int:
        nonlocal last_event_tick
        abs_tick = max(abs_tick, last_event_tick)
        d = abs_tick - last_event_tick
        last_event_tick = abs_tick
        return d

    for tok in tokens:
        if TS_OFF <= tok < TS_OFF + vocab["time_shift_bins"]:
            k = tok - TS_OFF
            current_time_ms += (k + 1) * TS_STEP
            current_tick = start_tick + ms_to_ticks(current_time_ms)

        elif VEL_OFF <= tok < VEL_OFF + VEL_BINS:
            k = tok - VEL_OFF
            pending_velocity = max(1, min(127, int((k + 0.5) * 127 / VEL_BINS)))

        elif NOTE_ON_OFF <= tok < NOTE_ON_OFF + 128:
            pitch = tok - NOTE_ON_OFF
            track.append(mido.Message(
                "note_on", channel=channel, note=pitch,
                velocity=pending_velocity,
                time=delta(current_tick),
            ))
            active_notes[pitch] = current_tick

        elif NOTE_OFF_OFF <= tok < NOTE_OFF_OFF + 128:
            pitch = tok - NOTE_OFF_OFF
            if pitch in active_notes:
                off_tick = current_tick
                if min_note_ticks > 0:
                    off_tick = max(off_tick, active_notes[pitch] + min_note_ticks)
                track.append(mido.Message(
                    "note_off", channel=channel, note=pitch,
                    velocity=0,
                    time=delta(off_tick),
                ))
                del active_notes[pitch]

    for pitch in list(active_notes.keys()):
        off_tick = current_tick
        if min_note_ticks > 0:
            off_tick = max(off_tick, active_notes[pitch] + min_note_ticks)
        track.append(mido.Message(
            "note_off", channel=channel, note=pitch,
            velocity=0, time=delta(off_tick),
        ))

    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def append_generated_to_original_midi(
    midi_prompt_path: Path,
    full_tokens: list,
    prompt_token_len: int,
    vocab: dict,
    output_path: Path,
    min_note_ms: float = 25.0,
):
    """Save output where the original MIDI stays untouched and generated part is appended."""
    mid = mido.MidiFile(midi_prompt_path)
    generated_only = full_tokens[prompt_token_len:]
    if not generated_only:
        mid.save(output_path)
        print(f"MIDI saved -> {output_path}")
        return

    # Use first tempo event if present; otherwise default 120 BPM.
    tempo = 500_000
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
        else:
            continue
        break

    start_tick = max((_track_end_tick(tr) for tr in mid.tracks), default=0)
    gen_track = _tokens_to_track(
        generated_only,
        vocab,
        start_tick=start_tick,
        tempo=tempo,
        channel=0,
        min_note_ms=min_note_ms,
    )
    mid.tracks.append(gen_track)
    mid.save(output_path)
    print(f"MIDI saved -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_from_midi_prompt(
    checkpoint: Path,
    vocab_path: Path,
    midi_prompt: Path,
    output: Path,
    *,
    min_note_ms: float = 25.0,
    key_filter: bool = False,
    key_top_pcs: int = 7,
    key_penalty: float = 1.5,
    density_aware: bool = False,
    density_penalty: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int | None = None,
    instrument: str | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    """Generate continuation MIDI from a user-supplied prompt file.

    This is a programmatic entry point used by the Qt app and packaging flow.
    """

    def emit(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)
        else:
            print(msg)

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(vocab_path) as f:
        vocab = json.load(f)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = MidiTransformer(
        vocab_size=int(cfg["vocab_size"]),
        context_len=int(cfg["context_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    emit(f"Loaded checkpoint (epoch {cfg.get('epoch','?')}, val_loss={ckpt.get('val_loss','?'):.4f})")

    tracks = extract_tracks_from_midi(midi_prompt)
    if not tracks:
        raise ValueError(f"No tracks found in {midi_prompt}")

    if instrument is not None:
        matching = [t for t in tracks if t["instrument"] == instrument]
        if not matching:
            emit(f"Warning: no tracks matching instrument '{instrument}'; using dominant track")
            selected_track = max(tracks, key=lambda t: len(t["notes"]))
        else:
            selected_track = max(matching, key=lambda t: len(t["notes"]))
    else:
        selected_track = max(tracks, key=lambda t: len(t["notes"]))

    notes = selected_track["notes"]
    inst_name = selected_track["instrument"]
    all_tokens = tokenize_song(notes, inst_name)
    prompt_tokens = all_tokens

    emit(f"Prompt: {midi_prompt.name}  channel={selected_track['channel']} instrument={inst_name}")
    emit(f"        {len(notes)} notes -> {len(all_tokens)} tokens, using first {len(prompt_tokens)}")

    if len(prompt_tokens) > model.context_len:
        emit(
            f"Note: prompt has {len(prompt_tokens)} tokens, but model context is {model.context_len}. "
            "Generation conditions on the most recent context window."
        )

    gen_len = max(0, model.context_len - len(prompt_tokens))
    emit(f"Auto gen_len: {gen_len} (context={model.context_len}, prompt_tokens={len(prompt_tokens)})")

    key_penalty_mask = None
    if key_filter:
        allowed_pcs = infer_allowed_pitch_classes(prompt_tokens, vocab, top_pcs=key_top_pcs)
        if not allowed_pcs:
            emit("Warning: key filter requested but no NOTE_ON events found in prompt; skipping key filter.")
        else:
            key_penalty_mask = build_key_penalty_mask(
                vocab_size=int(cfg["vocab_size"]),
                note_on_offset=int(vocab["note_on_offset"]),
                allowed_pcs=allowed_pcs,
                penalty=float(key_penalty),
                device=device,
            )
            note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            allowed_str = ", ".join(note_names[pc] for pc in sorted(allowed_pcs))
            emit(f"Key filter ON: allowed pitch classes = {allowed_str} | penalty={key_penalty}")

    timeshift_penalty_mask = None
    if density_aware:
        prompt_density = compute_prompt_density(prompt_tokens, vocab)
        timeshift_penalty_mask = build_timeshift_density_penalty(
            vocab_size=int(cfg["vocab_size"]),
            prompt_density=prompt_density,
            ts_offset=int(vocab["time_shift_offset"]),
            ts_bins=int(vocab["time_shift_bins"]),
            ts_step_ms=float(vocab["time_shift_step_ms"]),
            penalty_base=float(density_penalty),
            device=device,
        )
        emit(f"Density-aware ON: prompt density = {prompt_density:.2f} notes/sec | penalty_base={density_penalty}")

    full_tokens = generate(
        model,
        prompt_tokens,
        gen_len,
        temperature,
        top_k,
        device,
        key_penalty_mask=key_penalty_mask,
        timeshift_penalty_mask=timeshift_penalty_mask,
    )
    emit(f"Generated {len(full_tokens) - len(prompt_tokens)} new tokens  (total sequence: {len(full_tokens)})")

    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = output.with_stem(output.stem + "_prompt")
    shutil.copy2(midi_prompt, prompt_path)
    emit(f"MIDI saved -> {prompt_path}")

    append_generated_to_original_midi(
        midi_prompt,
        full_tokens,
        len(prompt_tokens),
        vocab,
        output,
        min_note_ms=min_note_ms,
    )

def main():
    parser = argparse.ArgumentParser(description="Generate MIDI continuations.")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt")
    parser.add_argument("--prompt",      default="data/processed/lakh_tokens.jsonl")
    parser.add_argument("--vocab",       default="data/processed/vocab.json")
    parser.add_argument("--midi_prompt", default=None,
                        help="Path to a .mid/.midi file to use as the prompt instead of the dataset")
    parser.add_argument("--prompt_song", type=int, default=None, help="Song index to use as prompt (random if omitted)")
    parser.add_argument("--prompt_len",  type=int, default=None,
                        help="Prompt token count: for --midi_prompt, defaults to full MIDI; for dataset prompts, defaults to 256")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k",       type=int, default=50)
    parser.add_argument("--key_filter",  action="store_true",
                        help="Apply soft key-aware filtering inferred from prompt pitch classes")
    parser.add_argument("--key_top_pcs", type=int, default=7,
                        help="Number of dominant pitch classes to keep for --key_filter (default 7)")
    parser.add_argument("--key_penalty", type=float, default=1.5,
                        help="Logit penalty for out-of-key NOTE_ON choices when --key_filter")
    parser.add_argument("--density_aware", action="store_true",
                        help="Adapt TIME_SHIFT penalties based on prompt note density (fast -> short gaps, slow -> long gaps)")
    parser.add_argument("--density_penalty", type=float, default=1.0,
                        help="Base logit penalty for TIME_SHIFT tokens when --density_aware (default 1.0)")
    parser.add_argument("--min_note_ms", type=float, default=25.0,
                        help="Minimum note duration in milliseconds when decoding generated MIDI (default 25)")
    parser.add_argument("--instrument",  default=None,
                        help="Instrument: if --midi_prompt given, filter that track by instrument (auto-detect if omitted); "
                             "if dataset prompt, prepend INST token (piano/guitar/bass/strings/brass/wind/synth/drums/etc.)")
    parser.add_argument("--output",      default="generated.mid")
    parser.add_argument("--seed",        type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load vocab ---
    with open(args.vocab) as f:
        vocab = json.load(f)

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg  = ckpt["config"]

    model = MidiTransformer(
        vocab_size  = int(cfg["vocab_size"]),
        context_len = int(cfg["context_len"]),
        d_model     = int(cfg["d_model"]),
        n_heads     = int(cfg["n_heads"]),
        n_layers    = int(cfg["n_layers"]),
        dropout     = 0.0,   # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint (epoch {cfg.get('epoch','?')}, val_loss={ckpt.get('val_loss','?'):.4f})")

    # --- Load prompt tokens ---
    if args.midi_prompt is not None:
        # User supplied their own MIDI file
        midi_path = Path(args.midi_prompt)
        tracks = extract_tracks_from_midi(midi_path)
        if not tracks:
            raise ValueError(f"No tracks found in {midi_path}")
        
        # If instrument is specified, use that track; otherwise pick dominant by note count
        if args.instrument is not None:
            matching = [t for t in tracks if t["instrument"] == args.instrument]
            if not matching:
                print(f"Warning: no tracks matching instrument '{args.instrument}'; using dominant track")
                selected_track = max(tracks, key=lambda t: len(t["notes"]))
            else:
                selected_track = max(matching, key=lambda t: len(t["notes"]))
        else:
            # Auto-detect: use the track with most notes
            selected_track = max(tracks, key=lambda t: len(t["notes"]))
        
        notes = selected_track["notes"]
        instrument = selected_track["instrument"]
        all_tokens = tokenize_song(notes, instrument)
        if args.prompt_len is None:
            prompt_tokens = all_tokens
        else:
            prompt_tokens = all_tokens[:args.prompt_len]
        print(f"Prompt: {midi_path.name}  channel={selected_track['channel']} instrument={instrument}")
        print(f"        {len(notes)} notes -> {len(all_tokens)} tokens, using first {len(prompt_tokens)}")
    else:
        songs = []
        song_names = []
        with open(args.prompt) as f:
            for line in f:
                record = json.loads(line)
                songs.append(record["tokens"])
                song_names.append(record.get("file", "unknown"))

        song_idx = args.prompt_song if args.prompt_song is not None else random.randrange(len(songs))
        dataset_prompt_len = args.prompt_len if args.prompt_len is not None else 256
        prompt_tokens = songs[song_idx][:dataset_prompt_len]
        source_name   = song_names[song_idx]
        print(f"Prompt: song {song_idx} ({source_name}), first {len(prompt_tokens)} tokens")
        if args.instrument is not None:
            inst_toks  = vocab.get("instrument_tokens", {})
            inst_off   = vocab.get("inst_offset", 388)
            if args.instrument in inst_toks:
                inst_tok = inst_toks[args.instrument]
                # Replace any existing leading INST token, otherwise prepend
                if prompt_tokens and prompt_tokens[0] >= inst_off:
                    prompt_tokens = [inst_tok] + list(prompt_tokens[1:])
                else:
                    prompt_tokens = [inst_tok] + list(prompt_tokens)
                print(f"Conditioning on instrument: {args.instrument} (token {inst_tok})")
            else:
                print(f"Warning: instrument '{args.instrument}' not in vocab, ignoring.")

    if len(prompt_tokens) > model.context_len:
        print(
            f"Note: prompt has {len(prompt_tokens)} tokens, but model context is {model.context_len}. "
            "Generation conditions on the most recent context window."
        )

    # Keep prompt+continuation within context window for stable generation.
    gen_len = max(0, model.context_len - len(prompt_tokens))
    print(
        f"Auto gen_len: {gen_len} "
        f"(context={model.context_len}, prompt_tokens={len(prompt_tokens)})"
    )

    key_penalty_mask = None
    if args.key_filter:
        allowed_pcs = infer_allowed_pitch_classes(prompt_tokens, vocab, top_pcs=args.key_top_pcs)
        if not allowed_pcs:
            print("Warning: key filter requested but no NOTE_ON events found in prompt; skipping key filter.")
        else:
            key_penalty_mask = build_key_penalty_mask(
                vocab_size=int(cfg["vocab_size"]),
                note_on_offset=int(vocab["note_on_offset"]),
                allowed_pcs=allowed_pcs,
                penalty=float(args.key_penalty),
                device=device,
            )
            note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            allowed_str = ", ".join(note_names[pc] for pc in sorted(allowed_pcs))
            print(f"Key filter ON: allowed pitch classes = {allowed_str} | penalty={args.key_penalty}")

    timeshift_penalty_mask = None
    if args.density_aware:
        prompt_density = compute_prompt_density(prompt_tokens, vocab)
        timeshift_penalty_mask = build_timeshift_density_penalty(
            vocab_size=int(cfg["vocab_size"]),
            prompt_density=prompt_density,
            ts_offset=int(vocab["time_shift_offset"]),
            ts_bins=int(vocab["time_shift_bins"]),
            ts_step_ms=float(vocab["time_shift_step_ms"]),
            penalty_base=float(args.density_penalty),
            device=device,
        )
        print(f"Density-aware ON: prompt density = {prompt_density:.2f} notes/sec | penalty_base={args.density_penalty}")

    # --- Generate ---
    full_tokens = generate(
        model, prompt_tokens, gen_len,
        args.temperature, args.top_k, device,
        key_penalty_mask=key_penalty_mask,
        timeshift_penalty_mask=timeshift_penalty_mask,
    )
    print(f"Generated {len(full_tokens) - len(prompt_tokens)} new tokens  "
          f"(total sequence: {len(full_tokens)})")

    # --- Save prompt MIDI (so you can compare prompt vs continuation) ---
    output_path = Path(args.output)
    prompt_path = output_path.with_stem(output_path.stem + "_prompt")
    if args.midi_prompt is not None:
        # Keep the exact original MIDI bytes for prompt output.
        shutil.copy2(args.midi_prompt, prompt_path)
        print(f"MIDI saved -> {prompt_path}")
    else:
        tokens_to_midi(prompt_tokens, vocab, prompt_path)

    # --- Save full (prompt + continuation) MIDI ---
    if args.midi_prompt is not None:
        append_generated_to_original_midi(
            Path(args.midi_prompt),
            full_tokens,
            len(prompt_tokens),
            vocab,
            output_path,
            min_note_ms=args.min_note_ms,
        )
    else:
        tokens_to_midi(full_tokens, vocab, output_path, min_note_ms=args.min_note_ms)


if __name__ == "__main__":
    main()
