"""Check whether clip duration is leaking the class label.

DroneAudioDataset right-pads everything to 5 s. If drone and no_drone clips
differ systematically in native duration, the model could be learning "where
the signal ends" rather than anything acoustic. This runs Mann-Whitney U on
drone vs no_drone durations per split and writes per-clip durations to
outputs/results/duration_audit.csv.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import mannwhitneyu


SPLIT_CSV = Path("data/metadata/split_metadata.csv")
OUT_CSV = Path("outputs/results/duration_audit.csv")


def native_duration_seconds(path: str) -> float | None:
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return None


def main() -> None:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"missing {SPLIT_CSV} — run src.data.split first")

    df = pd.read_csv(SPLIT_CSV)
    print(f"Reading {len(df)} clip durations from {SPLIT_CSV} ...")

    df["duration_s"] = df["filepath"].apply(native_duration_seconds)

    missing = df["duration_s"].isna().sum()
    if missing:
        print(f"  WARNING: could not read {missing} files; dropping them")
        df = df.dropna(subset=["duration_s"]).copy()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["filepath", "label", "split", "duration_s"]].to_csv(OUT_CSV, index=False)
    print(f"Wrote per-clip durations to {OUT_CSV}")

    print("\n" + "=" * 70)
    print("Per-class duration summary (seconds)")
    print("=" * 70)
    summary = (
        df.groupby(["split", "label"])["duration_s"]
        .agg(["count", "min", "median", "mean", "std", "max"])
        .round(3)
    )
    print(summary)

    print("\n" + "=" * 70)
    print("Mann-Whitney U test: drone vs no_drone durations, per split")
    print("=" * 70)
    print(f"{'split':<8} {'U':>12}  {'p-value':>10}  {'median_drone':>13}  "
          f"{'median_no_drone':>15}  {'effect_size':>11}")
    print("-" * 78)
    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split]
        d = sub[sub["label"] == "drone"]["duration_s"].values
        n = sub[sub["label"] == "no_drone"]["duration_s"].values
        if len(d) == 0 or len(n) == 0:
            continue
        u_stat, p = mannwhitneyu(d, n, alternative="two-sided")
        # Rank-biserial correlation. |effect| near 1 = strong separation.
        n1, n2 = len(d), len(n)
        effect = 1 - (2 * u_stat) / (n1 * n2)
        print(
            f"{split:<8} {u_stat:>12.0f}  {p:>10.3e}  "
            f"{np.median(d):>13.3f}  {np.median(n):>15.3f}  {effect:>11.3f}"
        )

    print("\n" + "=" * 70)
    print("Interpretation guide")
    print("=" * 70)
    print(
        "  p < 0.001 with |effect| > 0.3  -> duration leakage is plausible\n"
        "                                    the model may be partly learning\n"
        "                                    'where the signal ends' instead\n"
        "                                    of acoustic features. Consider\n"
        "                                    retraining with a fixed random\n"
        "                                    5-second crop rather than\n"
        "                                    right-zero-padding.\n"
        "\n"
        "  p > 0.05  or  |effect| < 0.1  -> duration distributions overlap;\n"
        "                                    classification cannot be reduced\n"
        "                                    to a duration cue."
    )


if __name__ == "__main__":
    main()
