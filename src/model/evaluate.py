"""evaluate.py

Runs a comprehensive metrics evaluation of a trained MidiTransformer on the
held-out test split and saves a report + plots to checkpoints/.

Metrics computed
----------------
Loss / Perplexity
    Cross-entropy loss and perplexity (exp(loss)) on the test set.
    Perplexity is the standard language-model quality metric — lower is better,
    and it's interpretable as "the model is as confused as if choosing uniformly
    among PPL options at each step".

Per-token-type accuracy
    What fraction of NOTE_ON / NOTE_OFF / TIME_SHIFT / VELOCITY predictions are
    correct at each position (where the model's argmax matches the target).

Top-k accuracy  (k = 1, 5, 10)
    Whether the correct next token appears in the model's top-k predictions.
    Top-5 / Top-10 are more forgiving and better reflect musical plausibility.

Token-type distribution comparison
    Bar chart comparing the distribution of predicted token types vs ground truth,
    to check whether the model learns the rough event structure of MIDI.

Usage
    python src/model/evaluate.py --checkpoint checkpoints/best.pt
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "model"))
from dataset import MidiTokenDataset
from model   import MidiTransformer


# Token-type boundaries (must match vocab.json)
def get_token_type(tok: int, vocab: dict) -> str:
    if tok < vocab["note_off_offset"]:
        return "NOTE_ON"
    elif tok < vocab["time_shift_offset"]:
        return "NOTE_OFF"
    elif tok < vocab["velocity_offset"]:
        return "TIME_SHIFT"
    elif tok < vocab.get("inst_offset", vocab["vocab_size"]):
        return "VELOCITY"
    else:
        return "INST"


@torch.no_grad()
def evaluate(checkpoint: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load vocab ---
    vocab_path = PROJECT_ROOT / "data" / "processed" / "vocab.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"]

    # --- Load checkpoint ---
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg  = ckpt["config"]
    print(f"Checkpoint : epoch {cfg.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    model = MidiTransformer(
        vocab_size  = int(cfg["vocab_size"]),
        context_len = int(cfg["context_len"]),
        d_model     = int(cfg["d_model"]),
        n_heads     = int(cfg["n_heads"]),
        n_layers    = int(cfg["n_layers"]),
        dropout     = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    context_len  = int(cfg["context_len"])
    tokens_file  = PROJECT_ROOT / "data" / "processed" / "lakh_tokens.jsonl"
    test_ds = MidiTokenDataset(
        tokens_file, context_len, split="test",
        val_fraction=float(cfg.get("val_fraction", 0.05)),
        test_fraction=float(cfg.get("test_fraction", 0.05)),
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    print(f"Test windows: {len(test_ds):,}")

    # --- Accumulators ---
    total_loss   = 0.0
    total_tokens = 0
    n_batches    = 0

    # Top-k correct counts
    topk_correct = {1: 0, 5: 0, 10: 0}

    # Per-type correct / total
    type_correct = defaultdict(int)
    type_total   = defaultdict(int)

    # Token-type distribution (predicted argmax vs ground truth)
    pred_type_counts = defaultdict(int)
    true_type_counts = defaultdict(int)

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)           # (B, T)
        logits = model(x)                            # (B, T, V)

        B, T, V = logits.shape
        logits_flat = logits.view(-1, V)
        y_flat      = y.view(-1)                     # (B*T,)

        # Cross-entropy loss
        loss = F.cross_entropy(logits_flat, y_flat, reduction="sum")
        total_loss   += loss.item()
        total_tokens += y_flat.numel()
        n_batches    += 1

        # Top-k accuracy
        for k in (1, 5, 10):
            _, top_indices = torch.topk(logits_flat, k, dim=-1)  # (B*T, k)
            correct = (top_indices == y_flat.unsqueeze(1)).any(dim=1)
            topk_correct[k] += correct.sum().item()

        # Per-type accuracy and distribution — work on CPU for simplicity
        preds   = logits_flat.argmax(dim=-1).cpu().tolist()
        targets = y_flat.cpu().tolist()
        for pred, tgt in zip(preds, targets):
            ttype = get_token_type(tgt, vocab)
            type_total[ttype]   += 1
            type_correct[ttype] += int(pred == tgt)
            pred_type_counts[get_token_type(pred, vocab)] += 1
            true_type_counts[ttype] += 1

    # --- Compute summary metrics ---
    avg_loss    = total_loss / total_tokens
    perplexity  = math.exp(avg_loss)
    top1_acc    = topk_correct[1]  / total_tokens
    top5_acc    = topk_correct[5]  / total_tokens
    top10_acc   = topk_correct[10] / total_tokens

    # --- Print report ---
    print()
    print("=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    print(f"Tokens evaluated  : {total_tokens:,}")
    print(f"Cross-entropy loss: {avg_loss:.4f}")
    print(f"Perplexity        : {perplexity:.2f}")
    print()
    print(f"Top-1  accuracy   : {top1_acc*100:.2f}%")
    print(f"Top-5  accuracy   : {top5_acc*100:.2f}%")
    print(f"Top-10 accuracy   : {top10_acc*100:.2f}%")
    print()
    print("Per-type top-1 accuracy:")
    for ttype in ("NOTE_ON", "NOTE_OFF", "TIME_SHIFT", "VELOCITY"):
        n = type_total[ttype]
        c = type_correct[ttype]
        pct = 100 * c / n if n else 0.0
        print(f"  {ttype:<12s}: {pct:6.2f}%  ({c:,}/{n:,})")
    print("=" * 50)

    # --- Save metrics to JSON ---
    metrics = {
        "checkpoint":       checkpoint,
        "test_tokens":      total_tokens,
        "loss":             avg_loss,
        "perplexity":       perplexity,
        "top1_acc":         top1_acc,
        "top5_acc":         top5_acc,
        "top10_acc":        top10_acc,
        "per_type_acc":     {k: type_correct[k] / type_total[k] if type_total[k] else 0.0
                             for k in ("NOTE_ON", "NOTE_OFF", "TIME_SHIFT", "VELOCITY")},
    }
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # --- Plots ---
    _plot_topk(top1_acc, top5_acc, top10_acc, output_dir)
    _plot_per_type_acc(type_correct, type_total, output_dir)
    _plot_token_distribution(pred_type_counts, true_type_counts, total_tokens, output_dir)


def _plot_topk(top1, top5, top10, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ks    = ["Top-1", "Top-5", "Top-10"]
    vals  = [top1 * 100, top5 * 100, top10 * 100]
    bars  = ax.bar(ks, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-k Next-Token Accuracy (Test Set)")
    plt.tight_layout()
    plt.savefig(out_dir / "topk_accuracy.png")
    plt.close()


def _plot_per_type_acc(type_correct, type_total, out_dir: Path):
    types = ["NOTE_ON", "NOTE_OFF", "TIME_SHIFT", "VELOCITY"]
    accs  = [100 * type_correct[t] / type_total[t] if type_total[t] else 0.0 for t in types]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(types, accs, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Per-Token-Type Accuracy (Test Set)")
    plt.tight_layout()
    plt.savefig(out_dir / "per_type_accuracy.png")
    plt.close()


def _plot_token_distribution(pred_counts, true_counts, total: int, out_dir: Path):
    types  = ["NOTE_ON", "NOTE_OFF", "TIME_SHIFT", "VELOCITY"]
    true_f = [true_counts[t] / total * 100 for t in types]
    pred_f = [pred_counts[t] / total * 100 for t in types]

    x = range(len(types))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - w/2 for i in x], true_f, width=w, label="Ground truth", color="#4C72B0")
    ax.bar([i + w/2 for i in x], pred_f, width=w, label="Predicted",    color="#C44E52", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(types)
    ax.set_ylabel("% of tokens")
    ax.set_title("Token-Type Distribution: Predicted vs Ground Truth")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "token_distribution.png")
    plt.close()
    print(f"Plots saved → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt")
    parser.add_argument("--output_dir",  default="checkpoints")
    args = parser.parse_args()
    evaluate(args.checkpoint, Path(args.output_dir))


if __name__ == "__main__":
    main()
