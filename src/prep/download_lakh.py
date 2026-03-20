"""download_lakh.py

Downloads and extracts a subset of the Lakh MIDI Dataset.

Available subsets (pass --subset NAME):

    clean   -- "Clean MIDI subset": artist/title filenames, well-curated.
               File: clean_midi.tar.gz  (~17 k files)   [DEFAULT]

    full    -- LMD-full: all 176,581 deduplicated MIDI files.
               File: lmd_full.tar.gz   (much larger, ~2 GB)

    matched -- LMD-matched: 45,129 files matched to the Million Song Dataset.
               File: lmd_matched.tar.gz

Output: data/raw/lakh/

Usage:
    python src/prep/download_lakh.py
    python src/prep/download_lakh.py --subset full
    python src/prep/download_lakh.py --archive <path/to/already_downloaded.tar.gz>

If the automatic download fails, download the archive manually from:
    https://colinraffel.com/projects/lmd/
then pass it via --archive.
"""

import argparse
import re
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEST_DIR     = PROJECT_ROOT / "data" / "raw" / "lakh"

BASE_URL = "http://hog.ee.columbia.edu/craffel/lmd"

SUBSET_INFO = {
    "clean":   {"filename": "clean_midi.tar.gz",   "top_dir": "clean_midi"},
    "full":    {"filename": "lmd_full.tar.gz",      "top_dir": "lmd_full"},
    "matched": {"filename": "lmd_matched.tar.gz",   "top_dir": "lmd_matched"},
}

MANUAL_INSTRUCTIONS = """
Automatic download failed.  Please download the archive manually:

  1. Visit  https://colinraffel.com/projects/lmd/
  2. Download your chosen archive (e.g. clean_midi.tar.gz)
  3. Re-run this script pointing at the downloaded file:
       python src/prep/download_lakh.py --archive <path/to/archive.tar.gz>
"""


# Characters Windows forbids in filenames (not including path separators)
_WIN_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

def sanitize_filename(name: str) -> str:
    """Replace Windows-invalid characters in a filename with '_'."""
    return _WIN_INVALID.sub("_", name)



    def _hook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb  = downloaded / 1_048_576
            print(f"\r  {pct:5.1f}%  {mb:7.1f} MB downloaded", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()  # newline after progress bar


def try_download(url: str, dest: Path) -> bool:
    """Attempt a single URL; return True on success."""
    print(f"Downloading: {url}")
    try:
        download_with_progress(url, dest)
        print(f"Saved to {dest}")
        return True
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"  Failed: {exc}")
        if dest.exists():
            dest.unlink()   # remove partial download
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and extract a Lakh MIDI subset.")
    parser.add_argument(
        "--subset", default="clean", choices=list(SUBSET_INFO),
        help="Which LMD subset to download: clean (default), full, matched",
    )
    parser.add_argument(
        "--archive", default=None,
        help="Path to a locally downloaded archive (skips download)",
    )
    args = parser.parse_args()

    info     = SUBSET_INFO[args.subset]
    filename = info["filename"]
    url      = f"{BASE_URL}/{filename}"
    archive  = Path(args.archive) if args.archive else (DEST_DIR.parent / filename)

    DEST_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        success = try_download(url, archive)
        if not success:
            print(MANUAL_INSTRUCTIONS)
            raise SystemExit(1)
    else:
        print(f"Archive already present: {archive}")

    if DEST_DIR.exists() and any(DEST_DIR.iterdir()):
        print(f"Already extracted at {DEST_DIR} — skipping extraction.")
        midi_count = sum(1 for _ in DEST_DIR.rglob("*.mid"))
        print(f"  {midi_count:,} MIDI files found.")
        return

    print(f"Extracting to {DEST_DIR} ...")
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    seen_names: dict = {}   # sanitized filename -> count, for dedup
    top_dir = info["top_dir"]
    with tarfile.open(archive, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            parts = Path(member.name).parts
            # Security: reject path-traversal entries
            if ".." in parts:
                continue
            # Only extract files (skip directory entries)
            if not member.isfile():
                continue
            # Flatten: keep only the filename, sanitized for Windows
            filename = sanitize_filename(parts[-1])
            if not filename.lower().endswith((".mid", ".midi")):
                continue
            # Deduplicate: append a counter if name already seen
            if filename in seen_names:
                seen_names[filename] += 1
                stem, ext = filename.rsplit(".", 1)
                filename = f"{stem}_{seen_names[filename]}.{ext}"
            else:
                seen_names[filename] = 0
            member.name = filename
            members.append(member)
        tar.extractall(DEST_DIR, members=members)

    midi_count = sum(1 for _ in DEST_DIR.rglob("*.mid"))
    print(f"Done. {midi_count:,} MIDI files extracted to {DEST_DIR}")
    print()
    print("Next steps:")
    print("  python src/preprocessing/extract_notes.py")
    print("  python src/preprocessing/tokenize_notes.py")
    print("  python src/model/train.py")


if __name__ == "__main__":
    main()
