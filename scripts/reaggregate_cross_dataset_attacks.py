"""Re-aggregate cross-dataset attacks from the per-sample CSV.

Rebuilds outputs/results/cross_dataset_attacks.csv and .json with the
full per-(scope, slice) breakdown. Run after any cross-dataset attack
run that produced a per-sample CSV but a stripped-down aggregate
(e.g. if an older script version stored only the 'overall' rows).
"""
from __future__ import annotations

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_cross_dataset_attacks import _aggregate


def main() -> None:
    ps_path = Path("outputs/results/cross_dataset_attacks_per_sample.csv")
    mf_path = Path("data/metadata/swarm_test_manifest.csv")
    if not ps_path.exists():
        sys.exit(f"missing {ps_path}")
    if not mf_path.exists():
        sys.exit(f"missing {mf_path}")

    ps = pd.read_csv(ps_path)
    mf = pd.read_csv(mf_path)
    print(f"per_sample rows: {len(ps)}")
    print(f"by (attack, epsilon):")
    print(ps.groupby(["attack", "epsilon"]).size())

    nested, flat = _aggregate(ps, mf)

    out_csv = Path("outputs/results/cross_dataset_attacks.csv")
    out_json = Path("outputs/results/cross_dataset_attacks.json")
    flat.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(nested, f, indent=2)
    print(f"\nrebuilt {out_csv}  ({len(flat)} rows)")
    print(f"rebuilt {out_json}")

    # Print a sanity sample
    print("\nsanity: PGD eps=0.001 per source")
    print(flat[(flat["attack"] == "PGD") & (flat["epsilon"] == 0.001) &
               (flat["scope"] == "source_dataset")]
          [["slice", "n", "asr", "asr_ci_lo", "asr_ci_hi"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
