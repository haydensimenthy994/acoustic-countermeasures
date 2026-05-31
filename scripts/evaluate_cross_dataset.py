"""
Cross-dataset clean-accuracy evaluation on SWARM-AUDIO-DATASET.

Loads the existing checkpoints (no re-training) and reports how well
they generalise to a held-out corpus they have never seen during
training. Addresses the source-level data leakage and limited drone
diversity flagged in the project audit.

Reads:
  outputs/checkpoints/best_model_cnn14.pt        — CNN14ProxyClassifier
  outputs/checkpoints/best_model.pt              — ProxyAudioCNN
  data/metadata/swarm_test_manifest.csv          — built by build_swarm_manifest.py

Writes:
  outputs/results/cross_dataset_swarm.json       — full numeric report
  outputs/results/cross_dataset_swarm.csv        — per-(model,source) summary

Usage:
  python scripts/evaluate_cross_dataset.py
"""
from __future__ import annotations

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from src.data.dataset import DroneAudioDataset
from src.features.spectrograms import LogMelSpectrogram
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.models.pann_proxy import ProxyAudioCNN
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_cnn14(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            x  = batch["features"].to(device)   # [B, samples] raw waveform
            y  = batch["label"].to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p[:, 1].cpu().numpy())
    return np.array(preds), np.array(labels), np.array(probs)


def _run_proxycnn(
    model: torch.nn.Module,
    loader: DataLoader,
    mel: LogMelSpectrogram,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            wav = batch["features"].to(device)   # [B, samples]
            y   = batch["label"].to(device)
            mel_spec = mel(wav)                  # [B, n_mels, time]
            logits = model(mel_spec)
            p = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p[:, 1].cpu().numpy())
    return np.array(preds), np.array(labels), np.array(probs)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _binom_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % CI for proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _summarise(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
) -> dict:
    n = len(labels)
    correct = int((preds == labels).sum())
    acc = correct / n if n > 0 else 0.0
    lo, hi = _binom_ci(correct, n)

    has_pos = (labels == 1).any()
    has_neg = (labels == 0).any()
    out: dict = {
        "n":              int(n),
        "n_drone":        int((labels == 1).sum()),
        "n_no_drone":     int((labels == 0).sum()),
        "accuracy":       float(acc),
        "accuracy_ci95":  [float(lo), float(hi)],
    }
    if has_pos and has_neg:
        out["f1_drone"]  = float(f1_score(labels, preds, pos_label=1, zero_division=0))
        out["precision"] = float(precision_score(labels, preds, pos_label=1, zero_division=0))
        out["recall"]    = float(recall_score(labels, preds, pos_label=1, zero_division=0))
        try:
            out["auc_roc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            out["auc_roc"] = float("nan")
        out["confusion_matrix"] = confusion_matrix(labels, preds).tolist()
    else:
        # single-class slice (e.g. trident-only is all drone)
        out["recall_drone"] = float(((preds == 1) & (labels == 1)).sum() /
                                    max((labels == 1).sum(), 1))
        out["recall_no_drone"] = float(((preds == 0) & (labels == 0)).sum() /
                                       max((labels == 0).sum(), 1))
    return out


def _balanced_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    pos = (labels == 1)
    neg = (labels == 0)
    if not pos.any() or not neg.any():
        return float("nan")
    tpr = (preds[pos] == 1).mean()
    tnr = (preds[neg] == 0).mean()
    return float(0.5 * (tpr + tnr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    manifest_csv = "data/metadata/swarm_test_manifest.csv"
    if not Path(manifest_csv).exists():
        print(f"ERROR: {manifest_csv} not found. "
              f"Run scripts/build_swarm_manifest.py first.")
        sys.exit(1)

    test_dataset = DroneAudioDataset(
        metadata_csv=manifest_csv,
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"SWARM test samples: {len(test_dataset)}\n")

    # Models -----------------------------------------------------------------
    print("Loading CNN14...")
    cnn14 = CNN14ProxyClassifier(
        num_classes=2,
        pretrained_path="outputs/checkpoints/Cnn14_16k_mAP=0.438.pth",
    ).to(device)
    cnn14.load_state_dict(torch.load(
        "outputs/checkpoints/best_model_cnn14.pt",
        map_location=device, weights_only=False,
    ))
    cnn14.eval()

    print("Loading ProxyAudioCNN...")
    proxy = ProxyAudioCNN(input_channels=1, num_classes=2, dropout=0.3).to(device)
    proxy.load_state_dict(torch.load(
        "outputs/checkpoints/best_model.pt",
        map_location=device, weights_only=False,
    ))
    proxy.eval()

    mel = LogMelSpectrogram().to(device)

    # Inference --------------------------------------------------------------
    print("\nRunning CNN14 inference...")
    cnn14_preds, labels, cnn14_probs = _run_cnn14(cnn14, loader, device)

    print("Running ProxyAudioCNN inference...")
    proxy_preds, _, proxy_probs = _run_proxycnn(proxy, loader, mel, device)

    # Per-row metadata for slicing
    meta = test_dataset.df.reset_index(drop=True)
    meta["label_int"]  = labels
    meta["cnn14_pred"] = cnn14_preds
    meta["proxy_pred"] = proxy_preds
    meta["cnn14_correct"] = (cnn14_preds == labels).astype(int)
    meta["proxy_correct"] = (proxy_preds == labels).astype(int)

    # Aggregate reports ------------------------------------------------------
    report: dict = {
        "manifest":     manifest_csv,
        "n_total":      int(len(labels)),
        "models": {
            "CNN14": {
                "checkpoint": "outputs/checkpoints/best_model_cnn14.pt",
                "overall":    _summarise(cnn14_preds, labels, cnn14_probs),
                "balanced_accuracy":
                    _balanced_accuracy(cnn14_preds, labels),
                "by_source":  {},
                "by_drone_type": {},
            },
            "ProxyAudioCNN": {
                "checkpoint": "outputs/checkpoints/best_model.pt",
                "overall":    _summarise(proxy_preds, labels, proxy_probs),
                "balanced_accuracy":
                    _balanced_accuracy(proxy_preds, labels),
                "by_source":  {},
                "by_drone_type": {},
            },
        },
    }

    # Per-source slices
    for src, sub in meta.groupby("source_dataset"):
        idx = sub.index.values
        report["models"]["CNN14"]["by_source"][src] = _summarise(
            cnn14_preds[idx], labels[idx], cnn14_probs[idx]
        )
        report["models"]["ProxyAudioCNN"]["by_source"][src] = _summarise(
            proxy_preds[idx], labels[idx], proxy_probs[idx]
        )

    # Per-drone-type slices (drones-only)
    drone_meta = meta[meta["label"] == "drone"]
    for dt, sub in drone_meta.groupby("drone_type"):
        idx = sub.index.values
        report["models"]["CNN14"]["by_drone_type"][str(dt)] = _summarise(
            cnn14_preds[idx], labels[idx], cnn14_probs[idx]
        )
        report["models"]["ProxyAudioCNN"]["by_drone_type"][str(dt)] = _summarise(
            proxy_preds[idx], labels[idx], proxy_probs[idx]
        )

    # Save -------------------------------------------------------------------
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "cross_dataset_swarm.json", "w") as f:
        json.dump(report, f, indent=2)

    flat_rows: list[dict] = []
    for model_name, m in report["models"].items():
        flat_rows.append({
            "model":             model_name,
            "scope":             "overall",
            "n":                 m["overall"]["n"],
            "accuracy":          m["overall"]["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "f1_drone":          m["overall"].get("f1_drone"),
            "auc_roc":           m["overall"].get("auc_roc"),
        })
        for src, s in m["by_source"].items():
            flat_rows.append({
                "model":             model_name,
                "scope":             f"source:{src}",
                "n":                 s["n"],
                "accuracy":          s["accuracy"],
                "balanced_accuracy": float("nan"),
                "f1_drone":          s.get("f1_drone"),
                "auc_roc":           s.get("auc_roc"),
            })
        for dt, s in m["by_drone_type"].items():
            flat_rows.append({
                "model":             model_name,
                "scope":             f"drone_type:{dt}",
                "n":                 s["n"],
                "accuracy":          s["accuracy"],
                "balanced_accuracy": float("nan"),
                "f1_drone":          s.get("f1_drone"),
                "auc_roc":           s.get("auc_roc"),
            })
    pd.DataFrame(flat_rows).to_csv(
        out_dir / "cross_dataset_swarm.csv", index=False
    )

    # Console report ---------------------------------------------------------
    def _hdr(s: str) -> None:
        print("\n" + "=" * 72)
        print(s)
        print("=" * 72)

    for model_name, m in report["models"].items():
        _hdr(f"{model_name}  |  cross-dataset SWARM evaluation")
        o = m["overall"]
        ci = o["accuracy_ci95"]
        print(f"  Overall   acc = {o['accuracy']:.4f}  "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]   (n = {o['n']})")
        print(f"  Balanced  acc = {m['balanced_accuracy']:.4f}")
        if "f1_drone" in o:
            print(f"  F1(drone) = {o['f1_drone']:.4f}   "
                  f"AUC = {o.get('auc_roc', float('nan')):.4f}")
            cm = o["confusion_matrix"]
            print(f"  Confusion (rows = true, cols = pred):")
            print(f"            no_drone  drone")
            print(f"  no_drone   {cm[0][0]:>6}  {cm[0][1]:>6}")
            print(f"  drone      {cm[1][0]:>6}  {cm[1][1]:>6}")

        print("\n  Per-source breakdown:")
        print(f"  {'source':<24} {'n':>5}  {'acc':>7}  {'CI95':>20}")
        for src, s in m["by_source"].items():
            ci = s["accuracy_ci95"]
            print(f"  {src:<24} {s['n']:>5}  {s['accuracy']:>7.4f}  "
                  f"[{ci[0]:.3f}, {ci[1]:.3f}]")

        print("\n  Per-drone-type recall:")
        for dt, s in m["by_drone_type"].items():
            r = s.get("recall_drone", s.get("recall", float("nan")))
            ci = s["accuracy_ci95"]
            print(f"  {dt:<16} {s['n']:>5}  recall = {r:>6.4f}  "
                  f"[{ci[0]:.3f}, {ci[1]:.3f}]")

    print(f"\nSaved JSON  -> {out_dir / 'cross_dataset_swarm.json'}")
    print(f"Saved CSV   -> {out_dir / 'cross_dataset_swarm.csv'}")


if __name__ == "__main__":
    main()
