from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTES_FILE = PROJECT_ROOT / "data" / "processed" / "maestro_notes.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


all_notes_counts = []
all_durations = []
all_velocities = []
max_polyphony_list = []

print(f"Validating notes in: {NOTES_FILE}")


with open(NOTES_FILE) as f:
    for line in f:
        song = json.loads(line)
        notes = song["notes"]

        if not notes:
            continue

        all_notes_counts.append(len(notes))
        all_durations.extend([n["duration"] for n in notes])
        all_velocities.extend([n["velocity"] for n in notes])

        # Compute max polyphony for this song
        active_notes = []
        max_poly = 0
        for n in sorted(notes, key=lambda x: x["start"]):
            # Remove notes that have ended
            active_notes = [end for end in active_notes if end > n["start"]]
            # Add current note
            active_notes.append(n["start"] + n["duration"])
            max_poly = max(max_poly, len(active_notes))
        max_polyphony_list.append(max_poly)

# === Compute summary stats ===
num_songs = len(all_notes_counts)
min_notes, max_notes = min(all_notes_counts), max(all_notes_counts)
mean_notes = sum(all_notes_counts) / num_songs

min_duration, max_duration = min(all_durations), max(all_durations)
mean_duration = sum(all_durations) / len(all_durations)

min_velocity, max_velocity = min(all_velocities), max(all_velocities)
mean_velocity = sum(all_velocities) / len(all_velocities)

min_poly, max_poly = min(max_polyphony_list), max(max_polyphony_list)
mean_poly = sum(max_polyphony_list) / len(max_polyphony_list)

# === Print stats ===
print("=== Dataset Validation Summary ===")
print(f"Songs: {num_songs}")
print(f"Notes per song: min={min_notes}, max={max_notes}, mean={mean_notes:.1f}")
print(f"Note duration (s): min={min_duration:.3f}, max={max_duration:.3f}, mean={mean_duration:.3f}")
print(f"Note velocity: min={min_velocity}, max={max_velocity}, mean={mean_velocity:.1f}")
print(f"Max polyphony per song: min={min_poly}, max={max_poly}, mean={mean_poly:.1f}")

# === Histograms ===
plt.figure(figsize=(10, 5))
plt.hist(all_durations, bins=100, color='skyblue')
plt.xlabel("Duration (s)")
plt.ylabel("Count")
plt.title("Note Duration Distribution")
plt.savefig(OUTPUT_DIR / "note_duration_hist.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.hist(all_velocities, bins=127, color='orange')
plt.xlabel("Velocity")
plt.ylabel("Count")
plt.title("Note Velocity Distribution")
plt.savefig(OUTPUT_DIR / "note_velocity_hist.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.hist(max_polyphony_list, bins=range(1, max(max_polyphony_list)+2), color='green', align='left')
plt.xlabel("Max Polyphony per Song")
plt.ylabel("Count")
plt.title("Max Polyphony Distribution")
plt.savefig(OUTPUT_DIR / "max_polyphony_hist.png")
plt.close()

print(f"Histograms saved to: {OUTPUT_DIR}")
print("Validation complete!")
