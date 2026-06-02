"""Cross-dataset Figure 2 — CNN14 vs ProxyAudioCNN confusion matrices on SWARM."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.viz.style import apply_style, save_fig

LABELS = ["no_drone", "drone"]


def _draw_cm(ax, cm, title: str):
    cm = np.asarray(cm)
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(LABELS)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )
    return im


def _metrics_panel(ax, overall: dict, model_name: str):
    cm = np.asarray(overall["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel()
    text = (
        f"{model_name}\n"
        f"────────────────\n"
        f"Accuracy:  {overall['accuracy']:.4f}\n"
        f"Precision: {overall['precision']:.4f}\n"
        f"Recall:    {overall['recall']:.4f}\n"
        f"F1:        {overall['f1_drone']:.4f}\n"
        f"\n"
        f"TP {tp:4d}  FP {fp:3d}\n"
        f"FN {fn:4d}  TN {tn:3d}\n"
        f"\n"
        f"N = {overall['n']}"
    )
    ax.axis("off")
    ax.text(0.02, 0.98, text, family="monospace", fontsize=10,
            verticalalignment="top", transform=ax.transAxes)
    ax.set_title("Metrics")


def plot_cross_confusion_matrix(
    swarm_json: str = "outputs/results/cross_dataset_swarm.json",
    outname: str = "fig2_cross_confusion_matrix",
    outdir: str = "outputs/figures_cross",
):
    apply_style()
    with open(swarm_json) as f:
        data = json.load(f)

    models = data["models"]
    cnn = models["CNN14"]["overall"]
    proxy = models["ProxyAudioCNN"]["overall"]

    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 0.9, 1.1, 0.9], wspace=0.35)

    ax_cnn = fig.add_subplot(gs[0, 0])
    im = _draw_cm(ax_cnn, cnn["confusion_matrix"],
                  "(a) CNN14 — SWARM clean")
    fig.colorbar(im, ax=ax_cnn, fraction=0.046, pad=0.04)

    ax_cnn_m = fig.add_subplot(gs[0, 1])
    _metrics_panel(ax_cnn_m, cnn, "CNN14")

    ax_px = fig.add_subplot(gs[0, 2])
    im2 = _draw_cm(ax_px, proxy["confusion_matrix"],
                   "(b) ProxyAudioCNN — SWARM clean")
    fig.colorbar(im2, ax=ax_px, fraction=0.046, pad=0.04)

    ax_px_m = fig.add_subplot(gs[0, 3])
    _metrics_panel(ax_px_m, proxy, "Proxy (mel)")

    fig.suptitle(
        f"Figure 2 (cross-dataset) — Clean accuracy on SWARM "
        f"(N = {data['n_total']})",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    paths = save_fig(fig, outname, outdir=outdir)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in plot_cross_confusion_matrix():
        print(f"saved {p}")
