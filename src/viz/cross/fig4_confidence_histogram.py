"""
Cross-dataset Figure 4: confidence shift under PGD on SWARM.

INPUT:
    outputs/results/cross_pgd_samples_eps0.001.npz
    (from scripts/run_cross_pgd_with_samples.py)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.viz.style import apply_style, COLORS, save_fig


def plot_cross_confidence_histogram(
    npz_path: str = "outputs/results/cross_pgd_samples_eps0.001.npz",
    outname: str = "fig4_cross_confidence_histogram",
    outdir: str = "outputs/figures_cross",
):
    apply_style()
    data = np.load(npz_path, allow_pickle=True)
    clean_conf = data["clean_conf_correct"]
    adv_conf = data["adv_conf_correct"]
    eps = float(data["epsilon"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 1, 41)

    ax = axes[0]
    ax.hist(clean_conf, bins=bins, color=COLORS["clean"], alpha=0.6,
            label=f"Clean (mean = {clean_conf.mean():.3f})", edgecolor="white")
    ax.hist(adv_conf, bins=bins, color=COLORS["adversarial"], alpha=0.6,
            label=f"PGD (mean = {adv_conf.mean():.3f})", edgecolor="white")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.8,
               label="decision boundary (0.5)")
    ax.set_xlabel("P(correct class)")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"(a) SWARM — ε = {eps}")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center")

    ax = axes[1]
    drop = clean_conf - adv_conf
    ax.hist(drop, bins=40, color=COLORS["pgd"], alpha=0.75, edgecolor="white")
    ax.axvline(drop.mean(), color="black", linestyle="--",
               label=f"mean drop = {drop.mean():.3f}")
    ax.axvline(np.median(drop), color="gray", linestyle=":",
               label=f"median drop = {np.median(drop):.3f}")
    ax.set_xlabel("Confidence drop (clean − adversarial)")
    ax.set_ylabel("Number of samples")
    ax.set_title("(b) Per-sample confidence drop")
    ax.legend(loc="upper left")

    fig.suptitle(
        f"Figure 4 (cross-dataset) — PGD confidence shift "
        f"(N = {len(clean_conf)}, ε = {eps})",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    paths = save_fig(fig, outname, outdir=outdir)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in plot_cross_confidence_histogram():
        print(f"saved {p}")
