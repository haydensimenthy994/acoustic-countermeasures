"""PGD at a single epsilon that also dumps per-sample confidences (fig 4)
and one clean/adv waveform pair (fig 6) to a single .npz.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.dataset import DroneAudioDataset
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.attacks.pgd import pgd_attack


EPSILON   = 0.001
ALPHA     = EPSILON / 10
NUM_STEPS = 40
BATCH_SIZE = 16

CKPT_PATH = "outputs/checkpoints/best_model_cnn14.pt"
PANN_PATH = "outputs/checkpoints/Cnn14_16k_mAP=0.438.pth"
SPLIT_CSV = "data/metadata/split_metadata.csv"

OUT_PATH  = f"outputs/results/pgd_samples_eps{EPSILON}.npz"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNN14ProxyClassifier(
        num_classes=2,
        pretrained_path=PANN_PATH,
    ).to(device)
    model.load_state_dict(torch.load(
        CKPT_PATH,
        map_location=device,
        weights_only=False,
    ))
    model.eval()
    print("Loaded CNN14 model")

    test_dataset = DroneAudioDataset(
        metadata_csv=SPLIT_CSV,
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    clean_conf_correct = []
    adv_conf_correct   = []
    predicted_before   = []
    predicted_after    = []
    labels_all         = []

    example_clean = None
    example_adv   = None
    example_label = -1

    print(f"\nRunning PGD at ε={EPSILON}, α={ALPHA}, {NUM_STEPS} steps...")
    for batch_idx, batch in enumerate(test_loader):
        features = batch["features"].to(device)
        labels   = batch["label"].to(device)

        with torch.no_grad():
            clean_logits = model(features)
            clean_probs  = torch.softmax(clean_logits, dim=1)
            clean_preds  = clean_logits.argmax(dim=1)

        # Only attack samples the model gets right when clean (same as evaluate_pgd).
        correct_mask = clean_preds == labels
        if correct_mask.sum() == 0:
            continue

        features_correct = features[correct_mask]
        labels_correct   = labels[correct_mask]
        clean_probs_corr = clean_probs[correct_mask]

        adv_features, _ = pgd_attack(
            model, features_correct, labels_correct,
            epsilon=EPSILON, alpha=ALPHA, num_steps=NUM_STEPS, device=device,
        )

        with torch.no_grad():
            adv_logits = model(adv_features)
            adv_probs  = torch.softmax(adv_logits, dim=1)
            adv_preds  = adv_logits.argmax(dim=1)

        labels_cpu = labels_correct.cpu().numpy()
        clean_probs_cpu = clean_probs_corr.cpu().numpy()
        adv_probs_cpu   = adv_probs.cpu().numpy()
        clean_preds_cpu = clean_preds[correct_mask].cpu().numpy()
        adv_preds_cpu   = adv_preds.cpu().numpy()

        for i, y in enumerate(labels_cpu):
            clean_conf_correct.append(float(clean_probs_cpu[i, y]))
            adv_conf_correct  .append(float(adv_probs_cpu  [i, y]))
            predicted_before.append(int(clean_preds_cpu[i]))
            predicted_after .append(int(adv_preds_cpu  [i]))
            labels_all.append(int(y))

        # Grab the first drone clip the attack manages to flip — fig 6 needs it.
        if example_clean is None:
            for i, y in enumerate(labels_cpu):
                if y == 1 and adv_preds_cpu[i] != y:
                    example_clean = features_correct[i].detach().cpu().numpy()
                    example_adv   = adv_features    [i].detach().cpu().numpy()
                    example_label = int(y)
                    print(f"  captured example from batch {batch_idx} "
                          f"(clean_conf={clean_probs_cpu[i, y]:.3f} → "
                          f"adv_conf={adv_probs_cpu[i, y]:.3f})")
                    break

    n = len(labels_all)
    n_flipped = sum(1 for b, a in zip(predicted_before, predicted_after) if b != a)
    print(f"\nProcessed {n} correctly-classified test samples")
    print(f"  flipped by PGD:  {n_flipped} / {n}  ({100*n_flipped/n:.1f}%)")
    print(f"  mean clean conf: {np.mean(clean_conf_correct):.3f}")
    print(f"  mean adv   conf: {np.mean(adv_conf_correct):.3f}")

    if example_clean is None:
        print("WARNING: no drone sample was flipped — example_clean/adv will be empty.")
        example_clean = np.array([])
        example_adv   = np.array([])

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
    )
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()