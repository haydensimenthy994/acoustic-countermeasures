"""Regenerate all six in-distribution figures from the saved result CSVs/JSONs.

Outputs land in outputs/figures/ as both PNG (300 dpi) and PDF.
"""
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
        print(f"  FAILED:")
        traceback.print_exc()
        return False


def main():
    from src.viz.fig1_training_curves        import plot_training_curves
    from src.viz.fig2_confusion_matrix       import plot_confusion_matrix
    from src.viz.fig3_asr_vs_epsilon         import plot_asr_vs_epsilon
    from src.viz.fig4_confidence_histogram   import plot_confidence_histogram
    from src.viz.fig5_asr_vs_snr             import plot_asr_vs_snr
    from src.viz.fig6_spectrogram_comparison import plot_spectrogram_comparison

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    results = {
        "Figure 1 — Training curves":         run_one("fig1", plot_training_curves),
        "Figure 2 — Confusion matrix":        run_one("fig2", plot_confusion_matrix),
        "Figure 3 — ASR vs epsilon":          run_one("fig3", plot_asr_vs_epsilon),
        "Figure 4 — Confidence histogram":    run_one("fig4", plot_confidence_histogram),
        "Figure 5 — ASR vs SNR":              run_one("fig5", plot_asr_vs_snr),
        "Figure 6 — Spectrogram comparison":  run_one("fig6", plot_spectrogram_comparison),
    }

    print("\n=== SUMMARY ===")
    n_ok   = sum(results.values())
    n_fail = len(results) - n_ok
    for name, ok in results.items():
        print(f"  {'OK ' if ok else 'FAIL'}  {name}")
    print(f"\n{n_ok}/{len(results)} figures generated.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
