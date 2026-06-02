"""
PGD black-box transfer only (CNN14 -> ProxyAudioCNN).

Usage:
  python scripts/run_blackbox_pgd_transfer.py
  python scripts/run_blackbox_pgd_transfer.py --epsilons 0.01,0.02,0.05 --resume
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import DroneAudioDataset
from src.features.spectrograms import LogMelSpectrogram
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.models.pann_proxy import ProxyAudioCNN
from src.utils.seed import set_seed

_transfer_path = os.path.join(os.path.dirname(__file__), "run_blackbox_transfer.py")
_spec = importlib.util.spec_from_file_location("run_blackbox_transfer", _transfer_path)
_transfer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_transfer_mod)
evaluate_transfer_pgd = _transfer_mod.evaluate_transfer_pgd

DEFAULT_EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.05]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PGD black-box transfer (CNN14 -> ProxyAudioCNN).")
    p.add_argument(
        "--epsilons",
        default=",".join(str(e) for e in DEFAULT_EPSILONS),
        help="Comma-separated epsilons (default: all five).",
    )
    p.add_argument(
        "--num-steps", type=int, default=40,
        help="PGD iterations (default: 40).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Keep existing rows in blackbox_pgd_transfer.csv; skip done epsilons.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    epsilons = [float(e) for e in args.epsilons.split(",") if e.strip()]

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PGD transfer epsilons: {epsilons}  steps: {args.num_steps}\n")

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
        metadata_csv="data/metadata/split_metadata.csv",
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}\n")

    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "blackbox_pgd_transfer.csv"

    results: list[dict] = []
    done: set[float] = set()
    if args.resume and csv_path.exists():
        prev = pd.read_csv(csv_path)
        results = prev.to_dict("records")
        done = {float(r["epsilon"]) for r in results}
        print(f"[resume] keeping {len(done)} epsilons: {sorted(done)}")

    print("=" * 60)
    print("PGD TRANSFER (CNN14 -> ProxyAudioCNN)")
    print("=" * 60)
    print(f"{'Epsilon':<10} {'Transfer ASR':<16} {'Conf Drop':<12} {'SNR(dB)'}")
    print("-" * 55)

    for eps in epsilons:
        if eps in done:
            row = next(r for r in results if float(r["epsilon"]) == eps)
            print(
                f"{eps:<10.3f} "
                f"{row['transfer_asr']:<16.4f} "
                f"{row['avg_confidence_drop']:<12.4f} "
                f"{row['avg_snr_db']:.2f}  [skipped]"
            )
            continue

        r = evaluate_transfer_pgd(
            source, target, loader, mel,
            epsilon=eps, num_steps=args.num_steps, device=device,
        )
        # Replace if re-running same epsilon without resume bookkeeping
        results = [x for x in results if float(x["epsilon"]) != eps]
        results.append(r)
        results.sort(key=lambda x: float(x["epsilon"]))

        pd.DataFrame(results).to_csv(csv_path, index=False)
        json_path = out_dir / "blackbox_transfer_results.json"
        payload: dict = {}
        if json_path.exists():
            with open(json_path) as f:
                payload = json.load(f)
        payload["pgd_transfer"] = results
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(
            f"{r['epsilon']:<10.3f} "
            f"{r['transfer_asr']:<16.4f} "
            f"{r['avg_confidence_drop']:<12.4f} "
            f"{r['avg_snr_db']:.2f}  [saved]"
        )

    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
