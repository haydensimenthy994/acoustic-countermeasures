from __future__ import annotations

import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import DroneAudioDataset
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.attacks.pgd import evaluate_pgd
from src.utils.seed import set_seed


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


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

    split_csv = "data/metadata/split_metadata.csv"
    test_dataset = DroneAudioDataset(
        metadata_csv=split_csv,
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    # Match the FGSM sweep so we get a direct apples-to-apples comparison.
    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05]
    all_results = []

    print("\nPGD Results (CNN14, 40 steps):")
    print(f"{'Epsilon':<10} {'ASR':<10} {'Adv Acc':<10} {'Conf Drop':<12} {'Std Drop':<12} {'L2':<10} {'L-inf':<10} {'SNR(dB)':<10}")
    print("-" * 85)

    for eps in epsilons:
        results = evaluate_pgd(
            model, test_loader,
            epsilon=eps,
            num_steps=40,
            device=device
        )
        all_results.append(results)
        print(
            f"{results['epsilon']:<10.3f} "
            f"{results['attack_success_rate']:<10.4f} "
            f"{results['adv_accuracy']:<10.4f} "
            f"{results['avg_confidence_drop']:<12.4f} "
            f"{results['std_confidence_drop']:<12.4f} "
            f"{results['avg_l2_norm']:<10.4f} "
            f"{results['avg_linf_norm']:<10.4f} "
            f"{results['avg_snr_db']:<10.2f}"
        )

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)
    csv_path = output_dir / "pgd_results_cnn14.csv"
    df.to_csv(csv_path, index=False)

    json_path = output_dir / "pgd_results_cnn14.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()