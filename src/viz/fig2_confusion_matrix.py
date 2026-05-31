"""
Figure 2: Confusion matrix on the clean test set using the best CNN14 checkpoint.

Loads best_model_cnn14.pt, evaluates on the held-out test split, and plots
the confusion matrix alongside per-class precision/recall.

Run:
    python -m src.viz.fig2_confusion_matrix

INPUT:
    outputs/checkpoints/best_model_cnn14.pt
    outputs/checkpoints/Cnn14_16k_mAP=0.438.pth
    data/metadata/split_metadata.csv
"""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, accuracy_score)
import matplotlib.pyplot as plt

from src.viz.style import apply_style, save_fig
from src.data.dataset import DroneAudioDataset
from src.models.cnn14_proxy import CNN14ProxyClassifier


LABELS = ["no_drone", "drone"]


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        features = batch["features"].to(device)
        labels   = batch["label"].to(device)
        logits = model(features)
        preds  = logits.argmax(dim=1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds .cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(
    ckpt_path="outputs/checkpoints/best_model_cnn14.pt",
    pann_path="outputs/checkpoints/Cnn14_16k_mAP=0.438.pth",
    split_csv="data/metadata/split_metadata.csv",
    outname="fig2_confusion_matrix",
    batch_size=16,
):
    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model (same pattern as run_pgd.py) ---
    model = CNN14ProxyClassifier(
        num_classes=2,
        pretrained_path=pann_path,
    ).to(device)
    model.load_state_dict(torch.load(
        ckpt_path, map_location=device, weights_only=False,
    ))
    model.eval()
    print("Loaded CNN14 model")

    # --- Load test dataset (same pattern as run_pgd.py) ---
    test_dataset = DroneAudioDataset(
        metadata_csv=split_csv,
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    # --- Run inference ---
    y_true, y_pred = collect_predictions(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # --- Metrics ---
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = cm.ravel()

    # --- Plot ---
    fig, (ax_cm, ax_txt) = plt.subplots(
        1, 2, figsize=(11, 4.8),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    im = ax_cm.imshow(cm, cmap="Blues", aspect="equal")
    ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(LABELS)
    ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(LABELS)
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_title("(a) Confusion matrix — clean test set")
    ax_cm.grid(False)

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=18, fontweight="bold")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # Side panel with metrics
    ax_txt.axis("off")
    text = (
        f"Clean test-set performance\n"
        f"──────────────────────────\n"
        f"Accuracy:   {acc:.4f}\n"
        f"Precision:  {prec:.4f}\n"
        f"Recall:     {rec:.4f}\n"
        f"F1 score:   {f1:.4f}\n"
        f"\n"
        f"Confusion counts\n"
        f"──────────────────────────\n"
        f"True positives:   {tp:4d}\n"
        f"True negatives:   {tn:4d}\n"
        f"False positives:  {fp:4d}\n"
        f"False negatives:  {fn:4d}\n"
        f"\n"
        f"Total test samples: {len(y_true)}"
    )
    ax_txt.text(0.02, 0.98, text, family="monospace", fontsize=11,
                verticalalignment="top", transform=ax_txt.transAxes)
    ax_txt.set_title("(b) Metrics")

    fig.suptitle(
        "Figure 2 — CNN14 classifier performance on clean test set",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    paths = save_fig(fig, outname)
    plt.close(fig)

    # Also dump numbers to JSON for future reuse
    from pathlib import Path
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    with open("outputs/results/clean_baseline_confusion.json", "w") as f:
        json.dump({
            "accuracy":  float(acc),
            "precision": float(prec),
            "recall":    float(rec),
            "f1":        float(f1),
            "confusion_matrix": cm.tolist(),
            "labels": LABELS,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "n_samples": int(len(y_true)),
        }, f, indent=2)
    print("saved outputs/results/clean_baseline_confusion.json")

    return paths


if __name__ == "__main__":
    for p in plot_confusion_matrix():
        print(f"saved {p}")