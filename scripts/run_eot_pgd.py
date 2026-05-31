from __future__ import annotations

import json
import torch
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from pathlib import Path

from src.data.dataset import DroneAudioDataset
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.attacks.eot_pgd import (
    build_rir_bank,
    build_rir_bank_tensors,
    evaluate_eot_pgd,
)
from src.utils.seed import set_seed


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- model ----------------------------------------------------------
    model = CNN14ProxyClassifier(
        num_classes=2,
        pretrained_path="outputs/checkpoints/Cnn14_16k_mAP=0.438.pth",
    ).to(device)
    model.load_state_dict(
        torch.load(
            "outputs/checkpoints/best_model_cnn14.pt",
            map_location=device,
            weights_only=False,
        )
    )
    model.eval()
    print("Loaded CNN14 model")

    # ---- RIR bank (seeded for bit-reproducible runs) --------------------
    print("Precomputing RIR bank (20 rooms, seed=42)...")
    rir_bank = build_rir_bank(n=20, sample_rate=16000, seed=42)
    rir_kernels = build_rir_bank_tensors(rir_bank, max_len=512, device=device)
    print(f"RIR kernels ready: {rir_kernels.shape}")

    # ---- dataset --------------------------------------------------------
    test_dataset = DroneAudioDataset(
        metadata_csv="data/metadata/split_metadata.csv",
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    # batch_size=8 fits comfortably in 8 GB VRAM and roughly halves the
    # number of CNN14 forward+backward passes vs the previous batch_size=4.
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=0,
    )
    print(f"Test samples: {len(test_dataset)}\n")

    # ---- attack ---------------------------------------------------------
    epsilons = [0.001, 0.005]
    all_results = []

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eot_pgd_results_cnn14.csv"
    json_path = output_dir / "eot_pgd_results_cnn14.json"

    print("EOT-PGD Results (CNN14, 20 steps, 5 EOT samples):")
    print(
        f"{'Epsilon':<10} {'Digital ASR':<14} {'OTA ASR':<12} "
        f"{'Conf Drop':<12} {'SNR(dB)':<10}"
    )
    print("-" * 62)

    for eps in epsilons:
        print(f"\nRunning epsilon={eps}...", flush=True)
        results = evaluate_eot_pgd(
            model,
            test_loader,
            epsilon=eps,
            num_steps=20,
            num_eot_samples=5,
            sample_rate=16000,
            device=device,
            rir_kernels=rir_kernels,
        )
        all_results.append(results)
        print(
            f"\n{results['epsilon']:<10.3f} "
            f"{results['digital_asr']:<14.4f} "
            f"{results['ota_asr']:<12.4f} "
            f"{results['avg_confidence_drop']:<12.4f} "
            f"{results['avg_snr_db']:<10.2f}"
        )

        # Save after every epsilon — kill-resilient. If you Ctrl+C
        # during the next epsilon, the results so far are still on disk.
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [saved partial results: {len(all_results)}/{len(epsilons)} epsilons]")

    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()