from __future__ import annotations

import argparse
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

DEFAULT_EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.05]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="In-distribution EOT-PGD on CNN14 test set.")
    p.add_argument(
        "--epsilons",
        default=",".join(str(e) for e in DEFAULT_EPSILONS),
        help="Comma-separated epsilons (default: all five).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip epsilons already present in the output CSV.",
    )
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--num-eot-samples", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main():
    args = _parse_args()
    epsilons = [float(e) for e in args.epsilons.split(",") if e.strip()]

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    print("Precomputing RIR bank (20 rooms, seed=42)...")
    rir_bank = build_rir_bank(n=20, sample_rate=16000, seed=42)
    rir_kernels = build_rir_bank_tensors(rir_bank, max_len=512, device=device)
    print(f"RIR kernels ready: {rir_kernels.shape}")

    test_dataset = DroneAudioDataset(
        metadata_csv="data/metadata/split_metadata.csv",
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    # batch_size=8 fits in 8 GB VRAM and roughly halves the CNN14 fwd+bwd
    # count compared to the original batch_size=4.
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    print(f"Test samples: {len(test_dataset)}\n")

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eot_pgd_results_cnn14.csv"
    json_path = output_dir / "eot_pgd_results_cnn14.json"

    all_results: list[dict] = []
    done: set[float] = set()
    if args.resume and csv_path.exists():
        prev = pd.read_csv(csv_path)
        all_results = prev.to_dict("records")
        done = {float(r["epsilon"]) for r in all_results}
        print(f"[resume] already done: {sorted(done)}")

    print("EOT-PGD Results (CNN14, 20 steps, 5 EOT samples):")
    print(
        f"{'Epsilon':<10} {'Digital ASR':<14} {'OTA ASR':<12} "
        f"{'Conf Drop':<12} {'SNR(dB)':<10}"
    )
    print("-" * 62)

    for eps in epsilons:
        if eps in done:
            print(f"\nSkipping epsilon={eps} [already in CSV]", flush=True)
            continue
        print(f"\nRunning epsilon={eps}...", flush=True)
        results = evaluate_eot_pgd(
            model,
            test_loader,
            epsilon=eps,
            num_steps=args.num_steps,
            num_eot_samples=args.num_eot_samples,
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

        # Save after every epsilon so a Ctrl+C mid-run doesn't lose progress.
        pd.DataFrame(all_results).to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [saved partial results: {len(all_results)}/{len(epsilons)} epsilons]")

    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()