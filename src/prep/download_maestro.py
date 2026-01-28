import hashlib
import zipfile
import urllib.request
from pathlib import Path

MAESTRO_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0-midi.zip"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "maestro"
ARCHIVE_PATH = DATA_DIR / "maestro-v3.0.0-midi.zip"


def download():
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        print("Archive already exists, skipping download.")
        return

    print("Downloading MAESTRO v3.0.0 MIDI zip...")
    urllib.request.urlretrieve(MAESTRO_URL, ARCHIVE_PATH)
    print("Download complete.")


def extract_flattened():
    print("Extracting and flattening MIDIs...")
    count = 0

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as z:
        for name in z.namelist():
            if not (name.endswith(".mid") or name.endswith(".midi")):
                continue

            filename = Path(name).name
            target_path = RAW_DIR / filename

            i = 1
            while target_path.exists():
                target_path = RAW_DIR / f"{Path(filename).stem}_{i}.mid"
                i += 1

            with z.open(name) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

            count += 1

    print(f"Extracted {count} MIDI files.")


def main():
    download()
    extract_flattened()
    print("MAESTRO dataset ready.")


if __name__ == "__main__":
    main()
