"""
Figure 3: Attack Success Rate vs epsilon — FGSM and PGD on the same plot.
"""
import pandas as pd
import matplotlib.pyplot as plt

from src.viz.style import apply_style, COLORS, save_fig


# Try these column names in order — first hit wins
ASR_COLS = ["attack_success_rate", "asr", "ASR", "success_rate"]


def _get_asr_column(df, source=""):
    for c in ASR_COLS:
        if c in df.columns:
            return df[c]
    raise KeyError(
        f"No ASR column found in {source}. "
        f"Looked for {ASR_COLS}, found columns: {df.columns.tolist()}"
    )


def _pct(series):
    """Accept ASR as either 0-1 or 0-100. Return 0-100."""
    return series * 100 if series.max() <= 1.0 else series


def plot_asr_vs_epsilon(
    fgsm_csv="outputs/results/fgsm_results_cnn14.csv",
    pgd_csv="outputs/results/pgd_results_cnn14.csv",
    outname="fig3_asr_vs_epsilon",
):
    apply_style()
    fgsm = pd.read_csv(fgsm_csv).sort_values("epsilon")
    pgd  = pd.read_csv(pgd_csv ).sort_values("epsilon")

    fgsm_asr = _pct(_get_asr_column(fgsm, fgsm_csv))
    pgd_asr  = _pct(_get_asr_column(pgd,  pgd_csv))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(fgsm["epsilon"], fgsm_asr,
            marker="o", color=COLORS["fgsm"], label="FGSM (single step)")
    ax.plot(pgd["epsilon"], pgd_asr,
            marker="s", color=COLORS["pgd"],  label="PGD (40 iter)")

    # annotate the headline result — smallest epsilon PGD row
    pgd_sorted = pgd.sort_values("epsilon").reset_index(drop=True)
    pgd_min_eps = pgd_sorted.iloc[0]["epsilon"]
    pgd_min_asr = _pct(_get_asr_column(pgd_sorted, pgd_csv)).iloc[0]
    ax.annotate(
        f"PGD @ ε={pgd_min_eps}\nASR = {pgd_min_asr:.1f}%",
        xy=(pgd_min_eps, pgd_min_asr),
        xytext=(pgd_min_eps * 8, 85),
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        fontsize=10, ha="left",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Perturbation budget $\varepsilon$ ($L_\infty$ on raw waveform)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Figure 3 — Attack Success Rate vs perturbation budget")
    ax.set_ylim(-2, 105)
    ax.legend(loc="center right")
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(fgsm["epsilon"].min(), 52, "chance-level (50%)",
            fontsize=8, color="gray")

    fig.tight_layout()
    paths = save_fig(fig, outname)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in plot_asr_vs_epsilon():
        print(f"saved {p}")