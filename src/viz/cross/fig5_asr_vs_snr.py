"""
Cross-dataset Figure 5: jamming + spoofing ASR vs SNR on SWARM, PGD reference point.

INPUT:
    outputs/results/cross_dataset_jamming.csv
    outputs/results/cross_dataset_spoofing.csv
    outputs/results/cross_dataset_attacks.csv
    outputs/results/cross_dataset_attacks_per_sample.csv  (PGD SNR)
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.viz.style import apply_style, COLORS, save_fig


def _pct(series):
    return series * 100 if series.max() <= 1.0 else series


def plot_cross_asr_vs_snr(
    jamming_csv: str = "outputs/results/cross_dataset_jamming.csv",
    spoofing_csv: str = "outputs/results/cross_dataset_spoofing.csv",
    attacks_csv: str = "outputs/results/cross_dataset_attacks.csv",
    per_sample_csv: str = "outputs/results/cross_dataset_attacks_per_sample.csv",
    pgd_epsilon: float = 0.001,
    outname: str = "fig5_cross_asr_vs_snr",
    outdir: str = "outputs/figures_cross",
):
    apply_style()
    jam = pd.read_csv(jamming_csv).sort_values("snr_db")
    spoof = pd.read_csv(spoofing_csv).sort_values("snr_db")

    if "conditional_asr" in jam.columns:
        jam_asr = _pct(jam["conditional_asr"])
        jam_label = "Jamming (conditional ASR)"
    else:
        jam_asr = _pct(jam["asr"])
        jam_label = "Jamming (population ASR)"

    spoof_asr = _pct(spoof["spoofing_asr"])

    attacks = pd.read_csv(attacks_csv)
    pgd_row = attacks[
        (attacks["attack"] == "PGD")
        & (attacks["epsilon"] == pgd_epsilon)
        & (attacks["scope"] == "overall")
    ].iloc[0]
    pgd_asr = pgd_row["asr"] * 100

    ps = pd.read_csv(per_sample_csv)
    pgd_ps = ps[(ps["attack"] == "PGD") & (ps["epsilon"] == pgd_epsilon)]
    pgd_snr = float(pgd_ps["snr_db"].mean())

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(jam["snr_db"], jam_asr,
            marker="o", color=COLORS["jamming"], label=jam_label,
            linewidth=2.2, markersize=7)
    ax.plot(spoof["snr_db"], spoof_asr,
            marker="^", color=COLORS["spoofing"],
            label="Acoustic spoofing (no_drone clips)",
            linewidth=2.2, markersize=7)

    ax.scatter([pgd_snr], [pgd_asr],
               color=COLORS["pgd"], marker="*", s=280, zorder=5,
               edgecolors="black", linewidths=0.8,
               label=f"PGD (ε={pgd_epsilon}) @ SNR={pgd_snr:.1f} dB → ASR={pgd_asr:.1f}%")

    xmax_snr = max(jam["snr_db"].max(), spoof["snr_db"].max(), pgd_snr) + 3
    xmin_snr = min(jam["snr_db"].min(), spoof["snr_db"].min(), 0) - 2
    ax.set_xlim(xmax_snr, xmin_snr)

    ax.axvspan(30, xmax_snr, alpha=0.10, color="green")
    green_mid = (30 + xmax_snr) / 2
    ax.text(green_mid, 12,
            "imperceptible\n(SNR > 30 dB)",
            fontsize=10, color="darkgreen", style="italic",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="darkgreen", alpha=0.9))

    ax.set_xlabel("Signal-to-Noise Ratio (dB)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(
        "Figure 5 (cross-dataset) — Baselines vs PGD on SWARM"
    )
    ax.set_ylim(-2, 105)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)

    fig.tight_layout()
    paths = save_fig(fig, outname, outdir=outdir)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in plot_cross_asr_vs_snr():
        print(f"saved {p}")
