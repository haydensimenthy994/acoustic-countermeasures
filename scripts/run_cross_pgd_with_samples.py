"""
PGD at ε=0.001 on SWARM (clean-correct only) — saves confidences + one
waveform pair for cross-dataset fig 4 / fig 6.

Matches run_cross_dataset_attacks.py: 20 PGD steps, swarm manifest, seed 42.

Run:
    python scripts/run_cross_pgd_with_samples.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import DroneAudioDataset
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.attacks.pgd import pgd_attack
from src.utils.seed import set_seed

EPSILON = 0.001
ALPHA = EPSILON / 10
NUM_STEPS = 20
BATCH_SIZE = 16
MANIFEST = "data/metadata/swarm_test_manifest.csv"
OUT_PATH = "outputs/results/cross_pgd_samples_eps0.001.npz"


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not Path(MANIFEST).exists():
        print(f"ERROR: {MANIFEST} missing — run build_swarm_manifest.py first.")
        sys.exit(1)

    model = CNN14ProxyClassifier(
        num_classes=2,
        pretrained_path="outputs/checkpoints/Cnn14_16k_mAP=0.438.pth",
    ).to(device)
    model.load_state_dict(torch.load(
        "outputs/checkpoints/best_model_cnn14.pt",
        map_location=device,
        weights_only=False,
    ))
    model.eval()
    print("Loaded CNN14")

    dataset = DroneAudioDataset(
        metadata_csv=MANIFEST,
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"SWARM samples: {len(dataset)}")

    clean_conf_correct = []
    adv_conf_correct = []
    predicted_before = []
    predicted_after = []
    labels_all = []

    example_clean = None
    example_adv = None
    example_label = -1
    example_path = ""

    print(f"\nPGD eps={EPSILON}, {NUM_STEPS} steps (cross-dataset)...")
    for batch_idx, batch in enumerate(loader):
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        filepaths = batch["filepath"]

        with torch.no_grad():
            clean_logits = model(features)
            clean_probs = torch.softmax(clean_logits, dim=1)
            clean_preds = clean_logits.argmax(dim=1)

        correct_mask = clean_preds == labels
        if correct_mask.sum() == 0:
            continue

        features_correct = features[correct_mask]
        labels_correct = labels[correct_mask]
        clean_probs_corr = clean_probs[correct_mask]
        idx_correct = correct_mask.nonzero(as_tuple=False).squeeze(1).cpu().tolist()
        if isinstance(idx_correct, int):
            idx_correct = [idx_correct]
        paths_correct = [filepaths[i] for i in idx_correct]

        adv_features, _ = pgd_attack(
            model, features_correct, labels_correct,
            epsilon=EPSILON, alpha=ALPHA, num_steps=NUM_STEPS,
            device=device, random_start=False,
        )

        with torch.no_grad():
            adv_logits = model(adv_features)
            adv_probs = torch.softmax(adv_logits, dim=1)
            adv_preds = adv_logits.argmax(dim=1)

        labels_cpu = labels_correct.cpu().numpy()
        clean_probs_cpu = clean_probs_corr.cpu().numpy()
        adv_probs_cpu = adv_probs.cpu().numpy()
        clean_preds_cpu = clean_preds[correct_mask].cpu().numpy()
        adv_preds_cpu = adv_preds.cpu().numpy()

        for i, y in enumerate(labels_cpu):
            clean_conf_correct.append(float(clean_probs_cpu[i, y]))
            adv_conf_correct.append(float(adv_probs_cpu[i, y]))
            predicted_before.append(int(clean_preds_cpu[i]))
            predicted_after.append(int(adv_preds_cpu[i]))
            labels_all.append(int(y))

        if example_clean is None:
            for i, y in enumerate(labels_cpu):
                if y == 1 and adv_preds_cpu[i] != y:
                    example_clean = features_correct[i].detach().cpu().numpy()
                    example_adv = adv_features[i].detach().cpu().numpy()
                    example_label = int(y)
                    example_path = paths_correct[i]
                    print(f"  example: {example_path}")
                    break

        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx + 1}...", flush=True)

    n = len(labels_all)
    n_flipped = sum(1 for b, a in zip(predicted_before, predicted_after) if b != a)
    print(f"\nClean-correct samples: {n}")
    print(f"  PGD flipped: {n_flipped} ({100 * n_flipped / n:.1f}%)")
    print(f"  mean clean conf: {np.mean(clean_conf_correct):.3f}")
    print(f"  mean adv conf:   {np.mean(adv_conf_correct):.3f}")

    if example_clean is None:
        print("WARNING: no flipped drone example — fig6 may be empty.")
        example_clean = np.array([])
        example_adv = np.array([])

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        epsilon=EPSILON,
        clean_conf_correct=np.array(clean_conf_correct),
        adv_conf_correct=np.array(adv_conf_correct),
        predicted_before=np.array(predicted_before),
        predicted_after=np.array(predicted_after),
        labels=np.array(labels_all),
        example_clean=example_clean,
        example_adv=example_adv,
        example_label=example_label,
        example_filepath=example_path,
    )
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
