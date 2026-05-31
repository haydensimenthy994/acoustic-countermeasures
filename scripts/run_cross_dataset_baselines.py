"""
Cross-dataset acoustic baselines (jamming + spoofing) on SWARM.

Mirrors scripts/run_baselines.py but evaluates on the held-out
SWARM-AUDIO-DATASET corpus instead of the in-distribution split.
Produces directly comparable conditional-ASR numbers so the
in-distribution vs cross-dataset gap can be quantified for jamming
and spoofing alongside the gradient-attack results.

Inputs:
  data/metadata/swarm_test_manifest.csv  (built by build_swarm_manifest.py)
  outputs/checkpoints/best_model_cnn14.pt

Outputs:
  outputs/results/cross_dataset_jamming.csv
  outputs/results/cross_dataset_jamming.json
  outputs/results/cross_dataset_spoofing.csv
  outputs/results/cross_dataset_spoofing.json

Usage:
  python scripts/run_cross_dataset_baselines.py
"""
from __future__ import annotations

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.attacks.baselines import evaluate_jamming, evaluate_spoofing
from src.data.dataset import DroneAudioDataset
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.utils.seed import set_seed


MANIFEST = "data/metadata/swarm_test_manifest.csv"


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    if not Path(MANIFEST).exists():
        print(f"ERROR: {MANIFEST} not found. "
              f"Run scripts/build_swarm_manifest.py first.")
        sys.exit(1)

    print("Loading CNN14...")
    model = CNN14ProxyClassifier(
        num_classes=2,
        pretrained_path="outputs/checkpoints/Cnn14_16k_mAP=0.438.pth",
    ).to(device)
    model.load_state_dict(torch.load(
        "outputs/checkpoints/best_model_cnn14.pt",
        map_location=device, weights_only=False,
    ))
    model.eval()
    print("  CNN14 loaded\n")

    test_dataset = DroneAudioDataset(
        metadata_csv=MANIFEST,
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"SWARM test samples: {len(test_dataset)}\n")

    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Jamming -----------------------------------------------------------
    # Same SNR sweep as the in-distribution run for direct comparison.
    snr_jam = [40, 35, 30, 25, 20, 15, 10, 5, 0, -5]

    print("=" * 50)
    print("CROSS-DATASET JAMMING BASELINE")
    print("=" * 50)
    jam = evaluate_jamming(model, loader, snr_levels=snr_jam, device=device)
    pd.DataFrame(jam).to_csv(out_dir / "cross_dataset_jamming.csv", index=False)
    with open(out_dir / "cross_dataset_jamming.json", "w") as f:
        json.dump(jam, f, indent=2)

    # ---- Spoofing ----------------------------------------------------------
    snr_spf = [20, 15, 10, 5, 0, -5, -10, -15, -20]

    print("\n" + "=" * 50)
    print("CROSS-DATASET ACOUSTIC SPOOFING BASELINE")
    print("=" * 50)
    spf = evaluate_spoofing(
        model, loader, snr_levels_db=snr_spf, device=device, seed=42,
    )
    pd.DataFrame(spf).to_csv(out_dir / "cross_dataset_spoofing.csv", index=False)
    with open(out_dir / "cross_dataset_spoofing.json", "w") as f:
        json.dump(spf, f, indent=2)

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY  in-distribution  vs  SWARM cross-dataset")
    print("=" * 60)

    in_jam = out_dir / "jamming_results.csv"
    if in_jam.exists():
        in_df = pd.read_csv(in_jam)
        sw_df = pd.DataFrame(jam)
        print("\nJamming conditional ASR  (in-dist  -> cross-dataset)")
        print(f"  {'SNR':>5}   {'in-dist':>10}   {'SWARM':>10}")
        for snr in [40, 30, 20, 10, 0, -5]:
            ir = in_df[in_df["snr_db"] == snr]
            sr = sw_df[sw_df["snr_db"] == snr]
            if ir.empty or sr.empty:
                continue
            print(f"  {snr:>5.0f}   "
                  f"{ir.iloc[0]['conditional_asr']:>10.4f}   "
                  f"{sr.iloc[0]['conditional_asr']:>10.4f}")

    in_spf = out_dir / "spoofing_results.csv"
    if in_spf.exists():
        in_df = pd.read_csv(in_spf)
        sw_df = pd.DataFrame(spf)
        print("\nSpoofing ASR  (in-dist  -> cross-dataset)")
        print(f"  {'SNR':>5}   {'in-dist':>10}   {'SWARM':>10}")
        for snr in [20, 10, 0, -10, -20]:
            ir = in_df[in_df["snr_db"] == snr]
            sr = sw_df[sw_df["snr_db"] == snr]
            if ir.empty or sr.empty:
                continue
            print(f"  {snr:>5.0f}   "
                  f"{ir.iloc[0]['spoofing_asr']:>10.4f}   "
                  f"{sr.iloc[0]['spoofing_asr']:>10.4f}")


if __name__ == "__main__":
    main()
