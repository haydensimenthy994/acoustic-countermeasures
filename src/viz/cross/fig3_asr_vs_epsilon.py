"""
Cross-dataset Figure 3: FGSM and PGD ASR vs epsilon on SWARM (overall).

INPUT:
    outputs/results/cross_dataset_attacks.csv  (scope=overall)
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.viz.style import apply_style, COLORS, save_fig


def plot_cross_asr_vs_epsilon(
    attacks_csv: str = "outputs/results/cross_dataset_attacks.csv",
    outname: str = "fig3_cross_asr_vs_epsilon",
    outdir: str = "outputs/figures_cross",
):
    apply_style()
    df = pd.read_csv(attacks_csv)
    overall = df[(df["scope"] == "overall") & (df["slice"] == "all")].copy()
    overall = overall.sort_values(["attack", "epsilon"])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for attack, color, marker in [
        ("FGSM", COLORS["fgsm"], "o"),
        ("PGD", COLORS["pgd"], "s"),
    ]:
        sub = overall[overall["attack"] == attack].sort_values("epsilon")
        asr_pct = sub["asr"] * 100
        ax.plot(sub["epsilon"], asr_pct,
                marker=marker, color=color, label=f"{attack}")

    pgd = overall[overall["attack"] == "PGD"].sort_values("epsilon").iloc[0]
    pgd_asr = pgd["asr"] * 100
    pgd_eps = pgd["epsilon"]
    ax.annotate(
        f"PGD @ ε={pgd_eps}\nASR = {pgd_asr:.1f}%",
        xy=(pgd_eps, pgd_asr),
        xytext=(pgd_eps * 8, 85),
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        fontsize=10, ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.9),
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Perturbation budget $\varepsilon$ ($L_\infty$ on raw waveform)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(
        "Figure 3 (cross-dataset) — ASR vs ε on SWARM "
        "(conditional, clean-correct denominator)"
    )
    ax.set_ylim(-2, 105)
    ax.legend(loc="center right")
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    paths = save_fig(fig, outname, outdir=outdir)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in plot_cross_asr_vs_epsilon():
        print(f"saved {p}")
