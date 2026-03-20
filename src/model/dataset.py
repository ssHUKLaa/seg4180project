"""dataset.py

PyTorch Dataset that streams token sequences from a JSONL tokens file and
yields fixed-length (context_len) windows suitable for autoregressive training.

Uses byte-offset indexing so that only the requested window's line is read
from disk at __getitem__ time — suitable for large datasets (LMD-clean) that
don't fit comfortably in RAM.

Each sample is a pair:
    x  – token IDs [0, context_len)        (input)
    y  – token IDs [1, context_len + 1)    (target, shifted by one)

Long songs are split into as many non-overlapping windows as possible.
The final partial window of each song is discarded.

Split proportions (by song count, default):
    train : 90 %
    val   :  5 %   (used for early stopping / checkpoint selection)
    test  :  5 %   (held out until final evaluation — never seen during training)
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class MidiTokenDataset(Dataset):
    def __init__(
        self,
        tokens_file:   Path,
        context_len:   int   = 1024,
        split:         str   = "train",   # "train" | "val" | "test"
        val_fraction:  float = 0.05,
        test_fraction: float = 0.05,
        seed:          int   = 42,
    ):
        """
        Args:
            tokens_file:    Path to *_tokens.jsonl
            context_len:    Number of tokens per training window
            split:          Which subset to load: "train", "val", or "test"
            val_fraction:   Fraction of songs held out for validation
            test_fraction:  Fraction of songs held out for final testing
            seed:           RNG seed for the split (must be the same across all splits)
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split!r}"
        self.tokens_file = Path(tokens_file)
        self.context_len = context_len

        # First pass: scan JSONL once to record byte offsets and sequence
        # lengths without loading token arrays into RAM.
        song_meta = []   # list of (byte_offset, n_tokens)
        with open(self.tokens_file, "rb") as fh:
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                record = json.loads(line)
                song_meta.append((offset, len(record["tokens"])))

        # Deterministic three-way split by song index
        rng = random.Random(seed)
        indices = list(range(len(song_meta)))
        rng.shuffle(indices)

        n_test = max(1, int(len(indices) * test_fraction))
        n_val  = max(1, int(len(indices) * val_fraction))

        test_idx  = set(indices[:n_test])
        val_idx   = set(indices[n_test : n_test + n_val])
        train_idx = set(indices[n_test + n_val :])

        chosen = {"train": train_idx, "val": val_idx, "test": test_idx}[split]

        # Build window index: list of (song_meta_index, window_start_token)
        self._song_meta: list = song_meta
        self._windows:   list = []
        window_size = context_len + 1
        for idx in chosen:
            _, n_tokens = song_meta[idx]
            for start in range(0, n_tokens - window_size + 1, context_len):
                self._windows.append((idx, start))

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        song_idx, start = self._windows[idx]
        byte_offset, _  = self._song_meta[song_idx]
        # Seek directly to the song's line — O(1) disk access
        with open(self.tokens_file, "rb") as fh:
            fh.seek(byte_offset)
            tokens = json.loads(fh.readline())["tokens"]
        end    = start + self.context_len + 1
        window = tokens[start:end]
        t = torch.tensor(window, dtype=torch.long)
        return t[:-1], t[1:]   # x, y
