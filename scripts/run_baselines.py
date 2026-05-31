from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.dataset import DroneAudioDataset
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.attacks.baselines import (
    evaluate_jamming,
    evaluate_drone_recall,
    evaluate_spoofing,
)
from src.utils.seed import set_seed


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNN14ProxyClassifier(
        num_classes=2,
        pretrained_path="outputs/checkpoints/Cnn14_16k_mAP=0.438.pth",
    ).to(device)
    model.load_state_dict(torch.load(
        "outputs/checkpoints/best_model_cnn14.pt",
        map_location=device,
        weights_only=False
    ))
    model.eval()
    print("Loaded CNN14 model")

    test_dataset = DroneAudioDataset(
        metadata_csv="data/metadata/split_metadata.csv",
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}\n")

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Jamming -------------------------------------------------------
    # SNR range: high SNR = weak jamming, low SNR = strong jamming.
    # `evaluate_jamming` now reports BOTH:
    #   - `asr` (population, 1 - accuracy)               — legacy
    #   - `conditional_asr` (clean-correct denominator)  — comparable with
    #     FGSM/PGD/EOT-PGD ASR on Figure 5.
    snr_levels = [40, 35, 30, 25, 20, 15, 10, 5, 0, -5]

    print("=" * 50)
    print("JAMMING BASELINE")
    print("=" * 50)
    jamming_results = evaluate_jamming(
        model, test_loader,
        snr_levels=snr_levels,
        device=device,
    )

    df_jamming = pd.DataFrame(jamming_results)
    df_jamming.to_csv(output_dir / "jamming_results.csv", index=False)
    with open(output_dir / "jamming_results.json", "w") as f:
        json.dump(jamming_results, f, indent=2)

    # ---- Drone-recall sanity check ------------------------------------
    # This is what the previous `spoofing_results.json` actually contained
    # (drone-class recall, mis-named as a spoofing ASR). Kept here under
    # the correct name so the historical number is still reproducible.
    print("\n" + "=" * 50)
    print("DRONE-RECALL SANITY CHECK  (legacy spoofing_results.json)")
    print("=" * 50)
    drone_recall = evaluate_drone_recall(model, test_loader, device=device)
    with open(output_dir / "drone_recall.json", "w") as f:
        json.dump(drone_recall, f, indent=2)

    # ---- Real acoustic spoofing baseline ------------------------------
    # Mix a drone clip into each clean-correct no_drone clip at a range
    # of no_drone:drone SNRs and measure how often the detector flips to
    # `drone`. High SNR = quiet (distant) drone; low SNR = loud drone.
    print("\n" + "=" * 50)
    print("ACOUSTIC SPOOFING BASELINE")
    print("=" * 50)
    spoof_snr_levels = [20, 15, 10, 5, 0, -5, -10, -15, -20]
    spoofing_results = evaluate_spoofing(
        model, test_loader,
        snr_levels_db=spoof_snr_levels,
        device=device,
        seed=42,
    )
    df_spoof = pd.DataFrame(spoofing_results)
    df_spoof.to_csv(output_dir / "spoofing_results.csv", index=False)
    with open(output_dir / "spoofing_results.json", "w") as f:
        json.dump(spoofing_results, f, indent=2)

    print(f"\nSaved jamming results to {output_dir / 'jamming_results.csv'}")
    print(f"Saved drone recall  to   {output_dir / 'drone_recall.json'}")
    print(f"Saved spoofing      to   {output_dir / 'spoofing_results.csv'}")

    # ---- Headline comparison table ------------------------------------
    print("\n" + "=" * 50)
    print("SUMMARY: Adversarial ASR vs Jamming (conditional) ASR")
    print("=" * 50)

    fgsm_csv = output_dir / "fgsm_results_cnn14.csv"
    pgd_csv = output_dir / "pgd_results_cnn14.csv"
    eot_csv = output_dir / "eot_pgd_results_cnn14.csv"

    def _row(df, eps):
        matches = df[df["epsilon"].round(4) == eps]
        return matches.iloc[0] if not matches.empty else None

    if fgsm_csv.exists():
        row = _row(pd.read_csv(fgsm_csv), 0.001)
        if row is not None:
            print(f"FGSM @ eps=0.001 ASR={row['attack_success_rate']*100:.2f}%  | SNR={row['avg_snr_db']:.2f}dB")
    if pgd_csv.exists():
        row = _row(pd.read_csv(pgd_csv), 0.001)
        if row is not None:
            print(f"PGD  @ eps=0.001 ASR={row['attack_success_rate']*100:.2f}%  | SNR={row['avg_snr_db']:.2f}dB")
    if eot_csv.exists():
        row = _row(pd.read_csv(eot_csv), 0.001)
        if row is not None:
            print(f"EOT  @ eps=0.001 OTA ASR={row['ota_asr']*100:.2f}%  | SNR={row['avg_snr_db']:.2f}dB")

    print("\nJamming conditional ASR at matched SNR levels:")
    for r in jamming_results:
        if r["snr_db"] in [40, 30, 20, 10]:
            print(f"  Jamming @ SNR={r['snr_db']:>4.0f}dB: cond ASR={r['conditional_asr']:.4f}  (pop ASR={r['asr']:.4f})")


if __name__ == "__main__":
    main()
