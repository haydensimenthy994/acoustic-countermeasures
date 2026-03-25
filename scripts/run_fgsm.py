from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from src.data.dataset import DroneAudioDataset
from src.models.pann_proxy import ProxyAudioCNN
from src.attacks.fgsm import evaluate_fgsm
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/model.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ProxyAudioCNN(
        input_channels=model_cfg["model"]["input_channels"],
        num_classes=model_cfg["model"]["num_classes"],
        dropout=model_cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pt", map_location=device))
    print("Loaded best model")

    split_csv = "data/metadata/split_metadata.csv"
    test_dataset = DroneAudioDataset(
        metadata_csv=split_csv,
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05]
    print("\nFGSM Results:")
    print(f"{'Epsilon':<10} {'ASR':<10} {'Adv Acc':<10} {'Conf Drop':<12}")
    print("-" * 45)

    for eps in epsilons:
        results = evaluate_fgsm(model, test_loader, epsilon=eps, device=device)
        print(
            f"{results['epsilon']:<10.3f} "
            f"{results['attack_success_rate']:<10.4f} "
            f"{results['adv_accuracy']:<10.4f} "
            f"{results['avg_confidence_drop']:<12.4f}"
        )


if __name__ == "__main__":
    main()