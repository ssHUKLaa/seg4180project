"""tokenize_notes.py

Converts extracted note sequences (maestro_notes.jsonl) into integer token
sequences suitable for training an autoregressive transformer model.

Vocabulary (388 tokens total):
    NOTE_ON_0   ... NOTE_ON_127     IDs   0-127   pitch 0-127
    NOTE_OFF_0  ... NOTE_OFF_127    IDs 128-255   pitch 0-127
    TIME_SHIFT_1... TIME_SHIFT_100  IDs 256-355   bin k (0-based) = (k+1) x TIME_SHIFT_STEP ms
    VELOCITY_0  ... VELOCITY_31     IDs 356-387   32 linear bins over velocity 1-127

TIME_SHIFT encoding:
    Bin k (0-indexed) represents a time advance of (k+1) * TIME_SHIFT_STEP model units.
    With TIME_SHIFT_STEP=10, bin 0=10ms, bin 99=1000ms.
    Gaps larger than 1000ms are encoded with multiple TIME_SHIFT tokens.
    All 100 TIME_SHIFT IDs (256-355) are utilised.

Sequence structure:
    Events (note_on, note_off) are sorted by time.
    Each NOTE_ON is immediately preceded by a VELOCITY token.
    At equal timestamps, NOTE_OFF events are emitted before NOTE_ON events.
"""

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTES_FILE  = PROJECT_ROOT / "data" / "processed" / "lakh_notes.jsonl"
TOKENS_FILE = PROJECT_ROOT / "data" / "processed" / "lakh_tokens.jsonl"
VOCAB_FILE  = PROJECT_ROOT / "data" / "processed" / "vocab.json"

# ---------------------------------------------------------------------------
# Vocabulary layout
# ---------------------------------------------------------------------------
NOTE_ON_OFFSET    = 0    # IDs   0-127
NOTE_OFF_OFFSET   = 128  # IDs 128-255
TIME_SHIFT_OFFSET = 256  # IDs 256-355
VELOCITY_OFFSET   = 356  # IDs 356-387

TIME_SHIFT_BINS = 100   # IDs 256-355  (bin k = (k+1) steps)
TIME_SHIFT_STEP = 10    # 10 model units == 10 ms per step
VELOCITY_BINS   = 32

# Instrument-category prefix tokens (one prepended per sequence)
INST_OFFSET = VELOCITY_OFFSET + VELOCITY_BINS   # 388
INST_CATS   = [
    "piano",    "chrom_perc", "organ",    "guitar",
    "bass",     "strings",    "ensemble", "brass",
    "wind",     "synth",      "drums",    "other",
]
INST_TOKEN  = {cat: INST_OFFSET + i for i, cat in enumerate(INST_CATS)}
VOCAB_SIZE  = INST_OFFSET + len(INST_CATS)      # 400

# Sort priority at identical timestamps: close before open
_NOTE_OFF = 0
_NOTE_ON  = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def velocity_to_bin(vel):
    """Map MIDI velocity 1-127 -> bin index 0-31."""
    return min(int((vel - 1) * VELOCITY_BINS / 127), VELOCITY_BINS - 1)


def quantize_time_shift(dt_float):
    """Convert a time delta (model units) into TIME_SHIFT token IDs.

    Uses integer step counting to avoid floating-point drift.
    Bin k (0-indexed) represents (k+1) * TIME_SHIFT_STEP model units.
    The maximum single-token shift is TIME_SHIFT_BINS * TIME_SHIFT_STEP (1000 ms).
    Larger gaps are encoded with multiple tokens.

    Returns an empty list for gaps smaller than TIME_SHIFT_STEP.
    """
    n_steps = int(round(dt_float / TIME_SHIFT_STEP))
    tokens = []
    while n_steps > 0:
        chunk = min(n_steps, TIME_SHIFT_BINS)   # 1..100
        tokens.append(TIME_SHIFT_OFFSET + chunk - 1)  # bin 0..99
        n_steps -= chunk
    return tokens


def tokenize_song(notes, instrument=None):
    """Convert a list of note dicts into a list of integer token IDs.

    If instrument is provided, the matching INST_* token is prepended.
    """
    tokens = (
        [INST_TOKEN.get(instrument, INST_OFFSET + len(INST_CATS) - 1)]
        if instrument is not None else []
    )
    events = []
    for n in notes:
        start = float(n["start"])
        end   = start + float(n["duration"])
        events.append((start, _NOTE_ON,  n["pitch"], n["velocity"]))
        events.append((end,   _NOTE_OFF, n["pitch"], 0))

    # At equal timestamps: _NOTE_OFF (0) sorts before _NOTE_ON (1)
    events.sort(key=lambda e: (e[0], e[1]))

    current_time = 0.0

    for time, etype, pitch, vel in events:
        dt = time - current_time
        if dt >= TIME_SHIFT_STEP:
            tokens.extend(quantize_time_shift(dt))
            current_time = time

        if etype == _NOTE_ON:
            tokens.append(VELOCITY_OFFSET + velocity_to_bin(vel))
            tokens.append(NOTE_ON_OFFSET  + pitch)
        else:
            tokens.append(NOTE_OFF_OFFSET + pitch)

    return tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)

    vocab = {
        "vocab_size":                VOCAB_SIZE,
        "note_on_offset":            NOTE_ON_OFFSET,
        "note_off_offset":           NOTE_OFF_OFFSET,
        "time_shift_offset":         TIME_SHIFT_OFFSET,
        "velocity_offset":           VELOCITY_OFFSET,
        "inst_offset":               INST_OFFSET,
        "time_shift_bins":           TIME_SHIFT_BINS,
        "time_shift_step_ms":        TIME_SHIFT_STEP,
        "velocity_bins":             VELOCITY_BINS,
        "instrument_tokens":         INST_TOKEN,
        "description": (
            "NOTE_ON 0-127 | NOTE_OFF 128-255 | "
            "TIME_SHIFT 256-355 (bin k => (k+1)*10 ms) | "
            "VELOCITY 356-387 (32 linear bins) | "
            "INST 388-399 (12 instrument categories)"
        ),
    }
    with open(VOCAB_FILE, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary saved -> {VOCAB_FILE}")

    seq_lengths = []
    skipped = 0

    with open(NOTES_FILE) as fin, open(TOKENS_FILE, "w") as fout:
        for line in fin:
            song   = json.loads(line)
            tokens = tokenize_song(song["notes"], song.get("instrument"))

            if len(tokens) < 10:
                skipped += 1
                continue

            seq_lengths.append(len(tokens))
            fout.write(json.dumps({"file": song["file"], "tokens": tokens}) + "\n")

    n = len(seq_lengths)
    print(f"Tokenization complete.")
    print(f"  Songs tokenized : {n}")
    print(f"  Songs skipped   : {skipped}")
    if seq_lengths:
        mean_len = sum(seq_lengths) / n
        print(f"  Sequence length : min={min(seq_lengths):,}  max={max(seq_lengths):,}  mean={mean_len:,.0f}")
    print(f"  Vocabulary size : {VOCAB_SIZE}")
    print(f"  Output          -> {TOKENS_FILE}")


if __name__ == "__main__":
    main()
