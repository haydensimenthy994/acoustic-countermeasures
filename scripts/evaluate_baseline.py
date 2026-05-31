from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.dataset import DroneAudioDataset
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.models.pann_proxy import ProxyAudioCNN
from src.utils.seed import set_seed


def load_model(model_name: str, device: torch.device):
    if model_name == "cnn14":
        model = CNN14ProxyClassifier(
            num_classes=2,
            pretrained_path="outputs/checkpoints/Cnn14_16k_mAP=0.438.pth",
        ).to(device)
        model.load_state_dict(torch.load(
            "outputs/checkpoints/best_model_cnn14.pt",
            map_location=device,
            weights_only=False,
        ))
        use_raw_waveform = True
        display_name = "CNN14ProxyClassifier"
        checkpoint = "outputs/checkpoints/best_model_cnn14.pt"
    elif model_name == "proxy":
        model = ProxyAudioCNN(
            input_channels=1, num_classes=2, dropout=0.3,
        ).to(device)
        model.load_state_dict(torch.load(
            "outputs/checkpoints/best_model.pt",
            map_location=device,
            weights_only=False,
        ))
        use_raw_waveform = False
        display_name = "ProxyAudioCNN"
        checkpoint = "outputs/checkpoints/best_model.pt"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model, use_raw_waveform, display_name, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Clean baseline evaluation on the held-out test split.",
    )
    parser.add_argument(
        "--model",
        choices=["cnn14", "proxy"],
        default="cnn14",
        help="cnn14: raw-waveform CNN14ProxyClassifier; "
             "proxy: log-mel ProxyAudioCNN (best_model.pt)",
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, use_raw_waveform, display_name, checkpoint = load_model(args.model, device)
    print(f"Loaded {display_name} from {checkpoint}")

    test_dataset = DroneAudioDataset(
        metadata_csv="data/metadata/split_metadata.csv",
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=use_raw_waveform,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=0,
    )
    print(f"Test samples: {len(test_dataset)}\n")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print("=" * 50)
    print(f"CLEAN BASELINE EVALUATION — {display_name}")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print()
    print("Confusion Matrix:")
    print("                 Pred: no_drone  Pred: drone")
    print(f"True: no_drone       {cm[0][0]:<12}  {cm[0][1]}")
    print(f"True: drone          {cm[1][0]:<12}  {cm[1][1]}")
    print()
    print("Per-class report:")
    print(classification_report(
        all_labels, all_preds, target_names=["no_drone", "drone"],
    ))

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        "baseline_evaluation.json"
        if args.model == "cnn14"
        else "proxy_baseline_evaluation.json"
    )
    confusion_stem = (
        "clean_baseline_confusion.json"
        if args.model == "cnn14"
        else "proxy_clean_baseline_confusion.json"
    )

    results = {
        "model": display_name,
        "checkpoint": checkpoint,
        "use_raw_waveform": use_raw_waveform,
        "test_samples": len(test_dataset),
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auc_roc": float(auc),
        "confusion_matrix": cm.tolist(),
    }

    with open(output_dir / stem, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    confusion = {
        "model": display_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "labels": ["no_drone", "drone"],
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
        "n_samples": len(test_dataset),
    }
    with open(output_dir / confusion_stem, "w", encoding="utf-8") as f:
        json.dump(confusion, f, indent=2)

    print(f"Saved to outputs/results/{stem}")
    print(f"Saved to outputs/results/{confusion_stem}")


if __name__ == "__main__":
    main()
