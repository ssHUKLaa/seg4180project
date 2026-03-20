"""extract_notes.py

Extracts per-channel note sequences from MIDI files and writes them to JSONL.
Each output line is a JSON object:
    {"file": str, "channel": int, "instrument": str, "notes": [...]}

instrument is a General-MIDI family name derived from the channel's programme
number.  Channel 9 is always labelled "drums".

Supports recursive directory scanning, so it works for both flat collections
(MAESTRO) and nested collections (Lakh MIDI Dataset).

Usage:
    python src/preprocessing/extract_notes.py [--raw_dir DIR] [--out_path FILE]
"""

from pathlib import Path
import json
import argparse
import mido

PROJECT_ROOT     = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR  = PROJECT_ROOT / "data" / "raw" / "lakh"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "lakh_notes.jsonl"
QUARANTINE_DIR   = "filtered_out"

# 1 model unit == 1 ms
SCALE = 1000
MAX_DURATION_SEC = 30.0


# ---------------------------------------------------------------------------
# General MIDI programme → instrument category
# ---------------------------------------------------------------------------

def program_to_category(program: int, is_drum_channel: bool = False) -> str:
    """Map a GM programme number (0-127) to a coarse instrument category."""
    if is_drum_channel:
        return "drums"
    if 0   <= program <= 7:   return "piano"
    if 8   <= program <= 15:  return "chrom_perc"
    if 16  <= program <= 23:  return "organ"
    if 24  <= program <= 31:  return "guitar"
    if 32  <= program <= 39:  return "bass"
    if 40  <= program <= 47:  return "strings"
    if 48  <= program <= 55:  return "ensemble"
    if 56  <= program <= 63:  return "brass"
    if 64  <= program <= 79:  return "wind"
    if 80  <= program <= 103: return "synth"
    if 112 <= program <= 119: return "drums"
    return "other"


# ---------------------------------------------------------------------------
# Per-channel extraction
# ---------------------------------------------------------------------------

def extract_tracks_from_midi(midi_path) -> list:
    """
    Extract notes organised by MIDI channel.  Each channel becomes one entry.
    Returns a list of dicts:
        {"channel": int, "instrument": str, "notes": [note_dict, ...]}
    where each note_dict has keys: pitch, start (ms), duration (ms), velocity.
    """
    mid = mido.MidiFile(midi_path)

    # Iterating MidiFile directly yields merged messages with msg.time already in
    # seconds (tempo-aware), so this is O(number_of_messages).
    time_sec = 0.0
    channel_program: dict = {}   # channel -> programme number
    active:          dict = {}   # (note, channel) -> (start_sec, velocity)
    channel_notes:   dict = {}   # channel -> [note_dict, ...]

    for msg in mid:
        time_sec += msg.time
        if not hasattr(msg, "channel"):
            continue
        ch = msg.channel

        if msg.type == "program_change":
            channel_program[ch] = msg.program

        elif msg.type == "note_on" and msg.velocity > 0:
            active[(msg.note, ch)] = (time_sec, msg.velocity)

        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.note, ch)
            if key in active:
                start_sec, vel = active.pop(key)
                dur = time_sec - start_sec
                if dur <= 0 or dur > MAX_DURATION_SEC:
                    continue
                channel_notes.setdefault(ch, []).append({
                    "pitch":    msg.note,
                    "start":    round(start_sec * SCALE, 3),
                    "duration": round(dur * SCALE, 3),
                    "velocity": vel,
                })

    results = []
    for ch in sorted(channel_notes):
        notes = sorted(channel_notes[ch], key=lambda n: n["start"])
        if not notes:
            continue
        prog = channel_program.get(ch, 0)
        cat  = program_to_category(prog, is_drum_channel=(ch == 9))
        results.append({
            "channel":    ch,
            "instrument": cat,
            "notes":      notes,
        })
    return results


def extract_notes_from_midi(midi_path, instrument: str = None) -> list:
    """
    Convenience wrapper used by generate.py --midi_prompt.

    If instrument is None, all channels are merged and sorted by start time.
    If instrument is given, only channels matching that category are included
    (with fallback to all notes if no match is found).
    """
    tracks = extract_tracks_from_midi(midi_path)
    if not tracks:
        return []
    if instrument is not None:
        matching = [t for t in tracks if t["instrument"] == instrument]
        if matching:
            tracks = matching
    all_notes = []
    for t in tracks:
        all_notes.extend(t["notes"])
    all_notes.sort(key=lambda n: n["start"])
    return all_notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-channel notes from MIDI files."
    )
    parser.add_argument("--raw_dir",  default=str(DEFAULT_RAW_DIR),
                        help="Directory to scan recursively for MIDI files")
    parser.add_argument("--out_path", default=str(DEFAULT_OUT_PATH),
                        help="Output JSONL file path")
    args = parser.parse_args()

    raw_dir  = Path(args.raw_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Recursive scan — works for flat (MAESTRO) and nested (LMD) layouts
    midi_files = sorted(
        p for p in raw_dir.rglob("*")
        if p.suffix.lower() in {".mid", ".midi"}
        and QUARANTINE_DIR not in p.parts
    )
    print(f"Found {len(midi_files)} MIDI files in {raw_dir}")

    written = 0
    failed  = 0
    with open(out_path, "w") as fout:
        total = len(midi_files)
        for i, midi_path in enumerate(midi_files, 1):
            try:
                tracks = extract_tracks_from_midi(midi_path)
                for t in tracks:
                    fout.write(json.dumps({
                        "file":       midi_path.name,
                        "channel":    t["channel"],
                        "instrument": t["instrument"],
                        "notes":      t["notes"],
                    }) + "\n")
                    written += 1
            except Exception as e:
                failed += 1
                if failed <= 20:
                    print(f"  FAILED: {midi_path.name} -- {e}")
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total} files processed | written={written} | failed={failed}")

    print(f"Done. {written} track entries written to {out_path}  ({failed} files failed)")


if __name__ == "__main__":
    main()
