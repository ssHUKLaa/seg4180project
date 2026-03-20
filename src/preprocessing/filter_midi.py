"""filter_midi.py

Quick validation pass over a flat directory of MIDI files.
Bad files are moved to a quarantine folder rather than deleted,
so the operation is fully reversible.

Rejection criteria
------------------
1. Unreadable / corrupt   -- mido raises any exception while opening
2. No melodic content     -- zero note_on events outside channel 9 (drums)
3. Too few notes          -- fewer than MIN_NOTES melodic note_on events
4. Too short              -- total MIDI duration under MIN_DURATION_SEC
5. Bad ticks_per_beat     -- zero or negative (would cause divide-by-zero in
                             tempo conversion)

Files that pass all checks are left in place.

Usage
-----
    python src/preprocessing/filter_midi.py
    python src/preprocessing/filter_midi.py --raw_dir data/raw/lakh --min_notes 20
"""

import argparse
import shutil
from pathlib import Path

import mido

PROJECT_ROOT     = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR  = PROJECT_ROOT / "data" / "raw" / "lakh"
QUARANTINE_SUBDIR = "filtered_out"

MIN_NOTES        = 20     # melodic note_on events required
MIN_DURATION_SEC = 5.0    # minimum total file length in seconds


# ---------------------------------------------------------------------------
# Per-file checks
# ---------------------------------------------------------------------------

def check_midi(path: Path, min_notes: int, min_duration: float) -> str | None:
    """
    Return None if the file passes all checks.
    Return a short reason string if it should be rejected.
    """
    try:
        mid = mido.MidiFile(path)
    except Exception as exc:
        return f"unreadable ({exc})"

    if mid.ticks_per_beat is None or mid.ticks_per_beat <= 0:
        return "bad ticks_per_beat"

    melodic_notes = 0
    for track in mid.tracks:
        for msg in track:
            if (msg.type == "note_on"
                    and msg.velocity > 0
                    and getattr(msg, "channel", 0) != 9):
                melodic_notes += 1

    if melodic_notes == 0:
        return "no melodic notes (drums-only or empty)"
    if melodic_notes < min_notes:
        return f"too few notes ({melodic_notes} < {min_notes})"

    # mid.length uses the merged iterator which correctly accumulates time
    try:
        duration = mid.length
    except Exception:
        return "could not compute duration"

    if duration < min_duration:
        return f"too short ({duration:.1f}s < {min_duration}s)"

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter out bad MIDI files.")
    parser.add_argument("--raw_dir",     default=str(DEFAULT_RAW_DIR),
                        help="Flat directory of MIDI files to validate")
    parser.add_argument("--min_notes",   type=int,   default=MIN_NOTES,
                        help=f"Minimum melodic note_on events (default {MIN_NOTES})")
    parser.add_argument("--min_duration", type=float, default=MIN_DURATION_SEC,
                        help=f"Minimum duration in seconds (default {MIN_DURATION_SEC})")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Report what would be rejected without moving any files")
    args = parser.parse_args()

    raw_dir    = Path(args.raw_dir)
    quarantine = raw_dir / QUARANTINE_SUBDIR

    midi_files = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mid", ".midi"}
    )
    total = len(midi_files)
    print(f"Scanning {total:,} MIDI files in {raw_dir} ...")
    if args.dry_run:
        print("(dry-run mode — no files will be moved)")

    if not args.dry_run:
        quarantine.mkdir(exist_ok=True)

    reasons: dict = {}   # reason -> count
    rejected = 0

    for i, path in enumerate(midi_files, 1):
        if i % 500 == 0 or i == total:
            print(f"  {i:>6}/{total}  rejected so far: {rejected}", end="\r")

        reason = check_midi(path, args.min_notes, args.min_duration)
        if reason is None:
            continue

        rejected += 1
        key = reason.split("(")[0].strip()   # group by reason type
        reasons[key] = reasons.get(key, 0) + 1

        if not args.dry_run:
            shutil.move(str(path), str(quarantine / path.name))

    kept = total - rejected
    print(f"\nDone.")
    print(f"  Total scanned : {total:,}")
    print(f"  Kept          : {kept:,}")
    print(f"  Rejected      : {rejected:,}  →  {quarantine if not args.dry_run else '(dry-run)'}")
    if reasons:
        print("  Rejection breakdown:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:>5}  {reason}")


if __name__ == "__main__":
    main()
