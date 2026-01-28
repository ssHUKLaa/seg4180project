from pathlib import Path
import json
import mido

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "maestro"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "maestro_notes.jsonl"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_TEMPO = 500000  # microseconds per beat


def extract_notes_from_midi(midi_path: Path):
    mid = mido.MidiFile(midi_path)

    tempo = DEFAULT_TEMPO
    time_sec = 0.0
    active = {}  # pitch -> (start_time, velocity)
    notes = []

    for msg in mid:
        time_sec += mido.tick2second(
            msg.time,
            mid.ticks_per_beat,
            tempo
        )

        if msg.type == "set_tempo":
            tempo = msg.tempo

        elif msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (time_sec, msg.velocity)

        elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
            if msg.note in active:
                start, vel = active.pop(msg.note)
                duration = time_sec - start

                
                if duration <= 0 or duration > 30:
                    continue

                SCALE = 1000  # converts seconds → “model units”

                notes.append({
                    "pitch": msg.note,
                    "start": start * SCALE,
                    "duration": duration * SCALE,
                    "velocity": vel
                })

    
    notes.sort(key=lambda n: n["start"])
    return notes


def main():
    midi_files = [
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() in {".mid", ".midi"}
    ]

    print(f"Processing {len(midi_files)} MIDI files...")

    with open(OUT_PATH, "w") as f:
        for midi_path in midi_files:
            try:
                notes = extract_notes_from_midi(midi_path)
                if not notes:
                    continue

                record = {
                    "file": midi_path.name,
                    "notes": notes
                }
                f.write(json.dumps(record) + "\n")

            except Exception as e:
                print(f"Failed: {midi_path.name} ({e})")

    print("Extraction complete.")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
