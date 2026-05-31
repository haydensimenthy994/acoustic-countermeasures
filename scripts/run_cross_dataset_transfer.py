"""
Cross-dataset black-box transferability (CNN14 -> ProxyAudioCNN) on SWARM.

Re-crafts FGSM adversarial waveforms against the white-box CNN14 source
model on the held-out SWARM corpus, converts them to log-mel
spectrograms with the EXACT transform used to train ProxyAudioCNN,
and reports cross-dataset transfer ASR with per-source breakdowns.

Mirrors the FGSM branch of scripts/run_blackbox_transfer.py but uses
the SWARM manifest as the evaluation corpus.

Inputs:
  data/metadata/swarm_test_manifest.csv      (build_swarm_manifest.py)
  outputs/checkpoints/best_model_cnn14.pt    (source / white-box)
  outputs/checkpoints/best_model.pt          (target / black-box)

Outputs:
  outputs/results/cross_dataset_transfer.csv          flat
  outputs/results/cross_dataset_transfer.json         nested
  outputs/results/cross_dataset_transfer_per_sample.csv

Usage:
  python scripts/run_cross_dataset_transfer.py
  python scripts/run_cross_dataset_transfer.py --epsilons 0.001,0.01,0.05
"""
from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.attacks.fgsm import fgsm_attack, compute_perturbation_metrics
from src.data.dataset import DroneAudioDataset
from src.features.spectrograms import LogMelSpectrogram
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.models.pann_proxy import ProxyAudioCNN
from src.utils.seed import set_seed


DEFAULT_EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.05]
MANIFEST         = "data/metadata/swarm_test_manifest.csv"


# ---------------------------------------------------------------------------
# One pass through the loader for a given epsilon
# ---------------------------------------------------------------------------

def _transfer_pass(
    source: torch.nn.Module,
    target: torch.nn.Module,
    mel: LogMelSpectrogram,
    loader: DataLoader,
    epsilon: float,
    device: torch.device,
) -> list[dict]:
    """Returns one row per *source-correct* clip with the target
    model's adversarial prediction (raw -> mel -> ProxyAudioCNN)."""
    rows: list[dict] = []
    snr_accum: list[float] = []

    for batch in loader:
        wav   = batch["features"].to(device)
        y     = batch["label"].to(device)
        files = list(batch["filepath"])

        with torch.no_grad():
            src_preds = source(wav).argmax(dim=1)
        correct_mask = src_preds == y
        if correct_mask.sum() == 0:
            continue

        wav_correct = wav[correct_mask].detach()
        y_correct   = y[correct_mask]
        idx_correct = correct_mask.nonzero(as_tuple=False).squeeze(1).cpu().tolist()

        # Craft on CNN14
        adv, _ = fgsm_attack(source, wav_correct, y_correct, epsilon, device)

        m = compute_perturbation_metrics(wav_correct, adv)
        snr_accum.append(m["snr_db"])

        # Evaluate on ProxyAudioCNN via mel-spectrogram
        with torch.no_grad():
            adv_mel  = mel(adv)
            tgt_preds = target(adv_mel).argmax(dim=1).cpu().tolist()
        y_list = y_correct.cpu().tolist()

        for i, batch_i in enumerate(idx_correct):
            rows.append({
                "filepath":   files[batch_i],
                "label_int":  y_list[i],
                "adv_pred":   tgt_preds[i],
                "attack":     "FGSM_transfer",
                "epsilon":    epsilon,
                "snr_db":     m["snr_db"],
            })

    print(
        f"    FGSM_transfer eps={epsilon:<6}  "
        f"n_correct={len(rows):>5d}  avg_snr={np.mean(snr_accum):>6.2f} dB",
        flush=True,
    )
    return rows


# ---------------------------------------------------------------------------
# Aggregation (same logic as run_cross_dataset_attacks.py)
# ---------------------------------------------------------------------------

def _binom_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _slice(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "asr": float("nan"),
                "asr_ci95": [float("nan"), float("nan")]}
    flipped = int((df["adv_pred"] != df["label_int"]).sum())
    lo, hi = _binom_ci(flipped, n)
    return {"n": int(n), "asr": float(flipped / n),
            "asr_ci95": [float(lo), float(hi)]}


def _aggregate(per_sample: pd.DataFrame, manifest: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    per_sample = per_sample.copy()
    per_sample["filepath"] = per_sample["filepath"].str.replace("\\", "/", regex=False)
    mf = manifest.copy()
    mf["filepath"] = mf["filepath"].str.replace("\\", "/", regex=False)
    df = per_sample.merge(
        mf[["filepath", "source_dataset", "drone_type", "label"]],
        on="filepath", how="left",
    )

    flat: list[dict] = []
    nested: dict = {"by_eps": {}}
    for eps, sub in df.groupby("epsilon"):
        entry: dict = {
            "epsilon":     float(eps),
            "overall":     _slice(sub),
            "by_label":    {},
            "by_source":   {},
            "by_drone_type": {},
        }
        flat.append({"epsilon": eps, "scope": "overall", "slice": "all",
                     **_slice(sub)})
        for lbl, s in sub.groupby("label"):
            entry["by_label"][str(lbl)] = _slice(s)
            flat.append({"epsilon": eps, "scope": "label",
                         "slice": str(lbl), **_slice(s)})
        for src, s in sub.groupby("source_dataset"):
            entry["by_source"][str(src)] = _slice(s)
            flat.append({"epsilon": eps, "scope": "source_dataset",
                         "slice": str(src), **_slice(s)})
        for dt, s in sub[sub["label"] == "drone"].groupby("drone_type"):
            entry["by_drone_type"][str(dt)] = _slice(s)
            flat.append({"epsilon": eps, "scope": "drone_type",
                         "slice": str(dt), **_slice(s)})
        nested["by_eps"][f"eps{eps}"] = entry

    flat_df = pd.DataFrame(flat)
    flat_df["asr_ci_lo"] = flat_df["asr_ci95"].apply(lambda x: x[0])
    flat_df["asr_ci_hi"] = flat_df["asr_ci95"].apply(lambda x: x[1])
    flat_df = flat_df.drop(columns=["asr_ci95"])
    return nested, flat_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-dataset CNN14 -> ProxyAudioCNN FGSM transfer.",
    )
    p.add_argument(
        "--epsilons",
        default=",".join(str(e) for e in DEFAULT_EPSILONS),
        help="Comma-separated epsilons.",
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="DataLoader batch size.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip epsilons already in the per-sample CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    epsilons = [float(e) for e in args.epsilons.split(",") if e.strip()]

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Epsilons: {epsilons}  batch: {args.batch_size}\n")

    if not Path(MANIFEST).exists():
        print(f"ERROR: {MANIFEST} not found. "
              f"Run scripts/build_swarm_manifest.py first.")
        sys.exit(1)

    print("Loading source CNN14...")
    source = CNN14ProxyClassifier(
        num_classes=2,
        pretrained_path="outputs/checkpoints/Cnn14_16k_mAP=0.438.pth",
    ).to(device)
    source.load_state_dict(torch.load(
        "outputs/checkpoints/best_model_cnn14.pt",
        map_location=device, weights_only=False,
    ))
    source.eval()

    print("Loading target ProxyAudioCNN...")
    target = ProxyAudioCNN(input_channels=1, num_classes=2, dropout=0.3).to(device)
    target.load_state_dict(torch.load(
        "outputs/checkpoints/best_model.pt",
        map_location=device, weights_only=False,
    ))
    target.eval()

    mel = LogMelSpectrogram().to(device)

    test_dataset = DroneAudioDataset(
        metadata_csv=MANIFEST,
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    print(f"SWARM test samples: {len(test_dataset)}\n")

    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_csv = out_dir / "cross_dataset_transfer_per_sample.csv"
    flat_csv       = out_dir / "cross_dataset_transfer.csv"
    nested_json    = out_dir / "cross_dataset_transfer.json"

    manifest_df = pd.read_csv(MANIFEST)
    all_rows: list[dict] = []
    done: set[float] = set()
    if args.resume and per_sample_csv.exists():
        prev = pd.read_csv(per_sample_csv)
        all_rows = prev.to_dict("records")
        done = {float(r["epsilon"]) for r in all_rows}
        print(f"[resume] already done: {sorted(done)}")

    print("Running CNN14 -> ProxyAudioCNN FGSM transfer ...\n")
    for eps in epsilons:
        if eps in done:
            print(f"    FGSM_transfer eps={eps}  [skipped]")
            continue
        rows = _transfer_pass(source, target, mel, loader, eps, device)
        all_rows.extend(rows)
        # Kill-resilient save
        pd.DataFrame(all_rows).to_csv(per_sample_csv, index=False)
        nested, flat = _aggregate(pd.DataFrame(all_rows), manifest_df)
        flat.to_csv(flat_csv, index=False)
        with open(nested_json, "w") as f:
            json.dump(nested, f, indent=2)

    # Final report -----------------------------------------------------------
    nested, flat = _aggregate(pd.DataFrame(all_rows), manifest_df)
    print("\n" + "=" * 78)
    print("CROSS-DATASET CNN14 -> ProxyAudioCNN FGSM TRANSFER ASR")
    print("=" * 78)
    print(f"  {'eps':<8} {'overall':<24} "
          f"{'alemadi':<22} {'trident':<22} {'wildlife_XC':<22}")
    empty = {"n": 0, "asr": float("nan"),
             "asr_ci95": [float("nan"), float("nan")]}
    for eps in epsilons:
        e = nested["by_eps"].get(f"eps{eps}")
        if e is None:
            continue
        src = e["by_source"]
        def _fmt(s: dict) -> str:
            if s["n"] == 0:
                return "  --"
            return (f"{s['asr']:.3f} ({s['n']})"
                    f" CI[{s['asr_ci95'][0]:.2f},{s['asr_ci95'][1]:.2f}]")
        print(f"  {eps:<8.4f} {_fmt(e['overall']):<24} "
              f"{_fmt(src.get('alemadi', empty)):<22} "
              f"{_fmt(src.get('trident', empty)):<22} "
              f"{_fmt(src.get('wildlife_xenocanto', empty)):<22}")

    # Side-by-side comparison with white-box if it exists
    wb_csv = out_dir / "cross_dataset_attacks.csv"
    if wb_csv.exists():
        wb = pd.read_csv(wb_csv)
        wb_fgsm = wb[(wb["attack"] == "FGSM") & (wb["scope"] == "overall")]
        print("\n  White-box (CNN14)  vs  Black-box transfer (ProxyAudioCNN)")
        print(f"  {'eps':<8} {'WB ASR':>10}   {'BB ASR':>10}   {'ratio':>8}")
        for eps in epsilons:
            wb_row = wb_fgsm[wb_fgsm["epsilon"].round(4) == round(eps, 4)]
            e = nested["by_eps"].get(f"eps{eps}")
            if e is None or wb_row.empty:
                continue
            wb_asr = float(wb_row.iloc[0]["asr"])
            bb_asr = e["overall"]["asr"]
            ratio = bb_asr / wb_asr if wb_asr > 0 else float("nan")
            print(f"  {eps:<8.4f} {wb_asr:>10.3f}   {bb_asr:>10.3f}   "
                  f"{ratio:>8.2f}")

    print(f"\nSaved nested JSON -> {nested_json}")
    print(f"Saved flat CSV    -> {flat_csv}")
    print(f"Saved per-sample  -> {per_sample_csv}")


if __name__ == "__main__":
    main()
