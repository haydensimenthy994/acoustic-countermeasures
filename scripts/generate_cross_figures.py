"""Regenerate the cross-dataset figures (fig2–fig6) for SWARM."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import traceback
from pathlib import Path


def run_one(name, fn):
    print(f"\n--- {name} ---")
    try:
        paths = fn()
        for p in paths:
            print(f"  saved {p}")
        return True
    except FileNotFoundError as e:
        print(f"  MISSING INPUT: {e}")
        return False
    except Exception:
        print("  FAILED:")
        traceback.print_exc()
        return False


def main():
    from src.viz.cross.fig2_confusion_matrix import plot_cross_confusion_matrix
    from src.viz.cross.fig3_asr_vs_epsilon import plot_cross_asr_vs_epsilon
    from src.viz.cross.fig4_confidence_histogram import plot_cross_confidence_histogram
    from src.viz.cross.fig5_asr_vs_snr import plot_cross_asr_vs_snr
    from src.viz.cross.fig6_spectrogram_comparison import plot_cross_spectrogram_comparison

    Path("outputs/figures_cross").mkdir(parents=True, exist_ok=True)

    npz = Path("outputs/results/cross_pgd_samples_eps0.001.npz")
    if not npz.exists():
        print(
            "NOTE: cross_pgd_samples_eps0.001.npz not found.\n"
            "      Run:  python scripts/run_cross_pgd_with_samples.py\n"
            "      (fig 4 and 6 will fail until then)\n"
        )

    results = {
        "Fig 2 cross — confusion (CNN14 + Proxy)": run_one(
            "fig2", plot_cross_confusion_matrix),
        "Fig 3 cross — ASR vs epsilon": run_one(
            "fig3", plot_cross_asr_vs_epsilon),
        "Fig 4 cross — confidence histogram": run_one(
            "fig4", plot_cross_confidence_histogram),
        "Fig 5 cross — ASR vs SNR": run_one(
            "fig5", plot_cross_asr_vs_snr),
        "Fig 6 cross — spectrograms": run_one(
            "fig6", plot_cross_spectrogram_comparison),
    }

    print("\n=== SUMMARY ===")
    n_ok = sum(results.values())
    for name, ok in results.items():
        print(f"  {'OK ' if ok else 'FAIL'}  {name}")
    print(f"\n{n_ok}/{len(results)} cross figures in outputs/figures_cross/")
    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
