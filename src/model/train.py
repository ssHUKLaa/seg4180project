"""train.py

Training loop for MidiTransformer.

Usage:
    python src/model/train.py

Key features:
    - Streams windows from maestro_tokens.jsonl via MidiTokenDataset
    - Mixed-precision training (torch.amp) when a CUDA GPU is available
    - Gradient clipping to stabilise early training
    - Checkpoint saved to checkpoints/best.pt whenever validation loss improves
    - Loss curves saved to checkpoints/loss_curve.png after training
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "model"))

from dataset import MidiTokenDataset
from model   import MidiTransformer

# ---------------------------------------------------------------------------
# Hyperparameters — edit here, or override by importing and calling train()
# ---------------------------------------------------------------------------
CONFIG = {
    # Data
    "tokens_file":   PROJECT_ROOT / "data" / "processed" / "lakh_tokens.jsonl",
    "context_len":   1024,
    "val_fraction":  0.05,
    "test_fraction": 0.05,

    # Model
    "vocab_size":    400,
    "d_model":       512,
    "n_heads":       8,
    "n_layers":      6,
    "dropout":       0.1,

    # Training
    "batch_size":    8,
    "lr":            3e-4,
    "weight_decay":  0.01,
    "max_epochs":    20,
    "grad_clip":     1.0,
    "log_every":     50,  # print a progress line every N batches
    "num_workers":   0,   # set >0 on Linux; keep 0 on Windows to avoid spawn issues

    # Output
    "checkpoint_dir": PROJECT_ROOT / "checkpoints",
}


def train(cfg: dict = CONFIG):
    cfg["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device : {device}  |  AMP : {use_amp}")

    # --- Datasets & loaders ---
    train_ds = MidiTokenDataset(cfg["tokens_file"], cfg["context_len"], split="train", val_fraction=cfg["val_fraction"], test_fraction=cfg["test_fraction"])
    val_ds   = MidiTokenDataset(cfg["tokens_file"], cfg["context_len"], split="val",   val_fraction=cfg["val_fraction"], test_fraction=cfg["test_fraction"])
    test_ds  = MidiTokenDataset(cfg["tokens_file"], cfg["context_len"], split="test",  val_fraction=cfg["val_fraction"], test_fraction=cfg["test_fraction"])
    print(f"Train windows : {len(train_ds):,}")
    print(f"Val   windows : {len(val_ds):,}")
    print(f"Test  windows : {len(test_ds):,}  (held out — not used during training)")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=cfg["num_workers"], pin_memory=use_amp)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=use_amp)

    # --- Model ---
    model = MidiTransformer(
        vocab_size  = cfg["vocab_size"],
        context_len = cfg["context_len"],
        d_model     = cfg["d_model"],
        n_heads     = cfg["n_heads"],
        n_layers    = cfg["n_layers"],
        dropout     = cfg["dropout"],
    ).to(device)
    print(f"Parameters    : {model.num_params():,}")

    # --- Optimiser ---
    # Apply weight decay only to weight matrices, not biases / layer-norm params
    decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg["lr"])

    scaler = torch.amp.GradScaler(enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["max_epochs"], eta_min=cfg["lr"] / 10
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, cfg["max_epochs"] + 1):
        # -- Train --
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)                          # (B, T, V)
                loss   = nn.functional.cross_entropy(
                    logits.view(-1, cfg["vocab_size"]),
                    y.view(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1

            if (batch_idx + 1) % cfg["log_every"] == 0:
                avg = total_loss / n_batches
                print(f"  epoch {epoch}  batch {batch_idx+1}/{len(train_loader)}  loss={avg:.4f}")

        train_loss = total_loss / n_batches
        train_losses.append(train_loss)

        # -- Validate --
        model.eval()
        total_val, n_val = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                    loss   = nn.functional.cross_entropy(
                        logits.view(-1, cfg["vocab_size"]),
                        y.view(-1),
                    )
                total_val += loss.item()
                n_val     += 1

        val_loss = total_val / n_val
        val_losses.append(val_loss)

        scheduler.step()
        import math
        print(f"Epoch {epoch:3d}/{cfg['max_epochs']}  "
              f"train_loss={train_loss:.4f}  train_ppl={math.exp(train_loss):.1f}  "
              f"val_loss={val_loss:.4f}  val_ppl={math.exp(val_loss):.1f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # -- Checkpoint --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = cfg["checkpoint_dir"] / "best.pt"
            torch.save({
                "epoch":      epoch,
                "val_loss":   val_loss,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config":     {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint (val_loss={val_loss:.4f})")

    # --- Loss curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    plt.plot(range(1, len(val_losses)   + 1), val_losses,   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    curve_path = cfg["checkpoint_dir"] / "loss_curve.png"
    plt.savefig(curve_path)
    plt.close()
    print(f"Loss curve saved → {curve_path}")


if __name__ == "__main__":
    train()
