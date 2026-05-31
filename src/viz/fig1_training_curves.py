"""
Figure 1: Training accuracy/loss curves — CNN14 vs scratch CNN.

Produces a 2x2 panel:
    top-left:  loss vs epoch (both models, train + val)
    top-right: accuracy vs epoch (both models, train + val)
    bottom-left: zoomed val-accuracy comparison
    bottom-right: text summary of final/best metrics

INPUT FILES EXPECTED:
    outputs/logs/training_metrics_cnn14.json
    outputs/logs/training_metrics_scratch.json

Each JSON must have the form:
{
    "run_name": "cnn14",
    "epochs": [1, 2, ...],
    "train_loss": [...],
    "train_acc":  [...],
    "val_loss":   [...],
    "val_acc":    [...],
    "best_val_acc": 0.9829,
    "best_epoch": 7
}

If you don't have these JSONs, use scripts/parse_train_log.py first
to generate them from your existing train_*.log files.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

from src.viz.style import apply_style, COLORS, save_fig


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def plot_training_curves(
    cnn14_path="outputs/logs/training_metrics_cnn14.json",
    scratch_path="outputs/logs/training_metrics_scratch.json",
    outname="fig1_training_curves",
):
    apply_style()
    cnn14 = load_metrics(cnn14_path)
    scratch = load_metrics(scratch_path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # --- top-left: loss ---
    ax = axes[0, 0]
    ax.plot(cnn14["epochs"], cnn14["train_loss"],
            color=COLORS["cnn14"], linestyle="-",  label="CNN14 train")
    ax.plot(cnn14["epochs"], cnn14["val_loss"],
            color=COLORS["cnn14"], linestyle="--", label="CNN14 val")
    ax.plot(scratch["epochs"], scratch["train_loss"],
            color=COLORS["scratch"], linestyle="-",  label="Scratch train")
    ax.plot(scratch["epochs"], scratch["val_loss"],
            color=COLORS["scratch"], linestyle="--", label="Scratch val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("(a) Loss over epochs")
    ax.legend(loc="upper right")

    # --- top-right: accuracy ---
    ax = axes[0, 1]
    ax.plot(cnn14["epochs"], cnn14["train_acc"],
            color=COLORS["cnn14"], linestyle="-",  label="CNN14 train")
    ax.plot(cnn14["epochs"], cnn14["val_acc"],
            color=COLORS["cnn14"], linestyle="--", label="CNN14 val")
    ax.plot(scratch["epochs"], scratch["train_acc"],
            color=COLORS["scratch"], linestyle="-",  label="Scratch train")
    ax.plot(scratch["epochs"], scratch["val_acc"],
            color=COLORS["scratch"], linestyle="--", label="Scratch val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("(b) Accuracy over epochs")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right")

    # --- bottom-left: val-accuracy zoom ---
    ax = axes[1, 0]
    ax.plot(cnn14["epochs"],  cnn14["val_acc"],
            color=COLORS["cnn14"],   marker="o", label="CNN14 val")
    ax.plot(scratch["epochs"], scratch["val_acc"],
            color=COLORS["scratch"], marker="s", label="Scratch val")
    # mark best points
    ax.axhline(cnn14["best_val_acc"], color=COLORS["cnn14"],
               linestyle=":", alpha=0.5)
    ax.axhline(scratch["best_val_acc"], color=COLORS["scratch"],
               linestyle=":", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("(c) Validation accuracy — zoomed")
    ax.legend(loc="lower right")

    # --- bottom-right: summary table ---
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"CNN14 (PANNs fine-tune)\n"
        f"   best val acc:  {cnn14['best_val_acc']:.4f} (epoch {cnn14['best_epoch']})\n"
        f"   final val acc: {cnn14['val_acc'][-1]:.4f}\n"
        f"   epochs:        {len(cnn14['epochs'])}\n"
        f"\n"
        f"Scratch CNN (ProxyAudioCNN)\n"
        f"   best val acc:  {scratch['best_val_acc']:.4f} (epoch {scratch['best_epoch']})\n"
        f"   final val acc: {scratch['val_acc'][-1]:.4f}\n"
        f"   epochs:        {len(scratch['epochs'])}\n"
        f"\n"
        f"Gap (best - final) indicates overfitting.\n"
        f"   CNN14:   {cnn14['best_val_acc']  - cnn14['val_acc'][-1]:+.4f}\n"
        f"   Scratch: {scratch['best_val_acc']- scratch['val_acc'][-1]:+.4f}"
    )
    ax.text(0.02, 0.98, summary, family="monospace", fontsize=10,
            verticalalignment="top", transform=ax.transAxes)
    ax.set_title("(d) Summary")

    fig.suptitle("Figure 1 — Training dynamics: CNN14 (PANNs) vs scratch CNN",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    paths = save_fig(fig, outname)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in plot_training_curves():
        print(f"saved {p}")
