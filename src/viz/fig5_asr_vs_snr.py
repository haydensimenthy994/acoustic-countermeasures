"""
Figure 5: ASR vs SNR — the headline comparison plot.

Jamming swept across SNR levels, with PGD and EOT-PGD shown as points
at their actual (SNR, ASR) operating coordinates. Demonstrates that
gradient-based attacks succeed inside the imperceptibility region
(SNR > 30 dB) while jamming does not.
"""
import pandas as pd
import matplotlib.pyplot as plt

from src.viz.style import apply_style, COLORS, save_fig


# Try these ASR column names in order — first hit wins
ASR_COLS = ["attack_success_rate", "asr", "ASR", "success_rate"]
SNR_COLS = ["avg_snr_db", "snr_db", "SNR_dB", "snr"]


def _get_col(df, candidates, source=""):
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise KeyError(
        f"None of {candidates} found in {source}. "
        f"Available columns: {df.columns.tolist()}"
    )


def _pct(series):
    """Accept values as either 0-1 or 0-100. Return 0-100."""
    if hasattr(series, "max"):
        return series * 100 if series.max() <= 1.0 else series
    return series * 100 if series <= 1.0 else series


def plot_asr_vs_snr(
    jamming_csv="outputs/results/jamming_results.csv",
    pgd_csv    ="outputs/results/pgd_results_cnn14.csv",
    eot_csv    ="outputs/results/eot_pgd_results_cnn14.csv",
    outname    ="fig5_asr_vs_snr",
):
    apply_style()
    jam = pd.read_csv(jamming_csv).sort_values("snr_db")

    # Jamming CSV: prefer `conditional_asr` (clean-correct denominator,
    # directly comparable with FGSM/PGD/EOT-PGD). Fall back to `asr`
    # (population, 1 - accuracy) for older CSVs that pre-date the fix.
    if "conditional_asr" in jam.columns:
        jam_asr = _pct(jam["conditional_asr"])
        jam_asr_label = "Jamming (broadband noise, conditional ASR)"
    elif "asr" in jam.columns:
        jam_asr = _pct(jam["asr"])
        jam_asr_label = "Jamming (broadband noise, population ASR)"
    elif "accuracy" in jam.columns:
        jam_asr = _pct(1 - jam["accuracy"])
        jam_asr_label = "Jamming (broadband noise, population ASR)"
    else:
        raise KeyError(
            f"jamming CSV needs 'conditional_asr', 'asr' or 'accuracy': "
            f"{jam.columns.tolist()}"
        )
    jam_snr = jam["snr_db"]

    # PGD row — smallest epsilon
    pgd = pd.read_csv(pgd_csv).sort_values("epsilon").reset_index(drop=True)
    pgd_row = pgd.iloc[0]
    pgd_asr = _pct(pd.Series([_get_col(pgd, ASR_COLS, pgd_csv).iloc[0]])).iloc[0]
    pgd_snr = _get_col(pgd, SNR_COLS, pgd_csv).iloc[0]
    pgd_eps = pgd_row["epsilon"]

    # EOT row — smallest epsilon
    eot = pd.read_csv(eot_csv).sort_values("epsilon").reset_index(drop=True)
    eot_row = eot.iloc[0]
    eot_d_asr = _pct(pd.Series([eot_row["digital_asr"]])).iloc[0]
    eot_o_asr = _pct(pd.Series([eot_row["ota_asr"    ]])).iloc[0]
    eot_snr   = _get_col(eot, SNR_COLS, eot_csv).iloc[0]
    eot_eps   = eot_row["epsilon"]

    # Wider figure so the external legend has room
    fig, ax = plt.subplots(figsize=(11, 6))

    # Jamming sweep curve
    ax.plot(jam_snr, jam_asr,
            marker="o", color=COLORS["jamming"],
            label=jam_asr_label,
            linewidth=2.2, markersize=7)

    # Reference points for optimised attacks — shown at their actual SNR
    ax.scatter([pgd_snr], [pgd_asr],
               color=COLORS["pgd"], marker="*", s=280, zorder=5,
               edgecolors="black", linewidths=0.8,
               label=f"PGD (ε={pgd_eps}) @ SNR={pgd_snr:.1f} dB → ASR={pgd_asr:.1f}%")
    ax.scatter([eot_snr], [eot_d_asr],
               color=COLORS["eot"], marker="D", s=130, zorder=5,
               edgecolors="black", linewidths=0.8,
               label=f"EOT-PGD digital (ε={eot_eps}) @ SNR={eot_snr:.1f} dB → ASR={eot_d_asr:.1f}%")
    ax.scatter([eot_snr], [eot_o_asr],
               color=COLORS["eot"], marker="D", s=130, zorder=5,
               edgecolors="black", linewidths=0.8, alpha=0.6,
               label=f"EOT-PGD OTA (ε={eot_eps}) @ SNR={eot_snr:.1f} dB → ASR={eot_o_asr:.1f}%")

    # --- axis limits and imperceptibility shading -------------------------
    # Explicitly set the x-axis so high-SNR (quieter attack) is on the left.
    # This replaces ax.invert_xaxis() which interacts badly with axvspan.
    xmax_snr = max(jam_snr.max(), pgd_snr, eot_snr) + 3
    xmin_snr = min(jam_snr.min(), 0) - 2
    ax.set_xlim(xmax_snr, xmin_snr)   # larger value on the left (reversed)

    # Green imperceptibility band covers SNR > 30 dB
    ax.axvspan(30, xmax_snr, alpha=0.10, color="green")

    # Label sits near the top of the green region, clear of the data markers
    green_mid = (30 + xmax_snr) / 2
    ax.text(green_mid, 15,
            "imperceptible\n(SNR > 30 dB)",
            fontsize=10, color="darkgreen", style="italic",
            ha="center", verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="darkgreen",
                      alpha=0.9))

    ax.set_xlabel("Signal-to-Noise Ratio (dB)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Figure 5 — Attack efficiency: gradient-based attacks vs jamming")
    ax.set_ylim(-2, 105)

    # Legend outside the plot, on the right — no collision with data
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=False)

    fig.tight_layout()
    paths = save_fig(fig, outname)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in plot_asr_vs_snr():
        print(f"saved {p}")