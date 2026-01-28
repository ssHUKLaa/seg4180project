from pathlib import Path
from collections import Counter
import mido

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIDI_DIR = PROJECT_ROOT / "data" / "raw" / "maestro"

print("Scanning:", MIDI_DIR)
print("Exists:", MIDI_DIR.exists())

stats = {
    "files": 0,
    "failed": 0,
    "note_on": 0,
    "note_off": 0,
    "tempos": Counter(),
    "ticks_per_beat": Counter(),
    "tracks": Counter(),
}

bad_files = []

midi_files = list(MIDI_DIR.glob("*.mid")) + list(MIDI_DIR.glob("*.midi"))

for midi_path in midi_files:
    stats["files"] += 1
    try:
        mid = mido.MidiFile(midi_path)
        stats["ticks_per_beat"][mid.ticks_per_beat] += 1
        stats["tracks"][len(mid.tracks)] += 1

        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    stats["tempos"][msg.tempo] += 1
                elif msg.type == "note_on" and msg.velocity > 0:
                    stats["note_on"] += 1
                elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
                    stats["note_off"] += 1

    except Exception as e:
        stats["failed"] += 1
        bad_files.append((midi_path.name, str(e)))

print("=== MIDI SMOKETEST ===")
for k, v in stats.items():
    if isinstance(v, Counter):
        print(f"{k}: {len(v)} unique")
    else:
        print(f"{k}: {v}")
