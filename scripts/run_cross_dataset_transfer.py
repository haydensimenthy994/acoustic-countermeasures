"""run_blackbox_transfer.py, but on the SWARM cross-dataset manifest.

Same CNN14 -> ProxyAudioCNN setup; reports transfer ASR with per-source
breakdowns so the in-dist vs cross-dataset gap is visible directly.
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
from src.attacks.pgd import pgd_attack
from src.attacks.eot_pgd import (
    eot_pgd_attack,
    build_rir_bank,
    build_rir_bank_tensors,
)
from src.data.dataset import DroneAudioDataset
from src.features.spectrograms import LogMelSpectrogram
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.models.pann_proxy import ProxyAudioCNN
from src.utils.seed import set_seed


DEFAULT_EPSILONS    = [0.001, 0.005, 0.01, 0.02, 0.05]
DEFAULT_PGD_STEPS   = 20
DEFAULT_EOT_STEPS   = 20
DEFAULT_EOT_SAMPLES = 5
DEFAULT_BATCH_SIZE  = 8
MANIFEST            = "data/metadata/swarm_test_manifest.csv"


def _transfer_pass(
    source: torch.nn.Module,
    target: torch.nn.Module,
    mel: LogMelSpectrogram,
    loader: DataLoader,
    attack: str,
    epsilon: float,
    device: torch.device,
    pgd_steps: int = DEFAULT_PGD_STEPS,
    eot_steps: int = DEFAULT_EOT_STEPS,
    num_eot_samples: int = DEFAULT_EOT_SAMPLES,
    rir_kernels: torch.Tensor | None = None,
) -> list[dict]:
    """One pass over the loader at a single (attack, eps).

    One row per source-correct clip; `adv_pred` is the target's prediction
    after raw -> mel -> ProxyAudioCNN.
    """
    attack_key = f"{attack}_transfer"
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

        if attack == "FGSM":
            adv, _ = fgsm_attack(source, wav_correct, y_correct, epsilon, device)
        elif attack == "PGD":
            adv, _ = pgd_attack(
                source, wav_correct, y_correct,
                epsilon=epsilon, alpha=epsilon / 10,
                num_steps=pgd_steps, device=device,
                random_start=False,
            )
        elif attack == "EOT-PGD":
            adv, _ = eot_pgd_attack(
                source, wav_correct, y_correct,
                epsilon=epsilon, alpha=epsilon / 10,
                num_steps=eot_steps, num_eot_samples=num_eot_samples,
                device=device, rir_kernels=rir_kernels,
                random_start=False,
            )
        else:
            raise ValueError(f"unknown attack: {attack}")

        m = compute_perturbation_metrics(wav_correct, adv)
        snr_accum.append(m["snr_db"])

        with torch.no_grad():
            adv_mel   = mel(adv)
            tgt_preds = target(adv_mel).argmax(dim=1).cpu().tolist()
        y_list = y_correct.cpu().tolist()

        for i, batch_i in enumerate(idx_correct):
            rows.append({
                "filepath":   files[batch_i],
                "label_int":  y_list[i],
                "adv_pred":   tgt_preds[i],
                "attack":     attack_key,
                "epsilon":    epsilon,
                "snr_db":     m["snr_db"],
            })

    print(
        f"    {attack_key} eps={epsilon:<6}  "
        f"n_correct={len(rows):>5d}  avg_snr={np.mean(snr_accum):>6.2f} dB",
        flush=True,
    )
    return rows


# Aggregation mirrors run_cross_dataset_attacks.py — kept here so this
# script stays runnable without importing from the other one.

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
    nested: dict = {"by_attack_eps": {}}
    for (attack, eps), sub in df.groupby(["attack", "epsilon"]):
        key = f"{attack}_eps{eps}"
        entry: dict = {
            "attack":      attack,
            "epsilon":     float(eps),
            "overall":     _slice(sub),
            "by_label":    {},
            "by_source":   {},
            "by_drone_type": {},
        }
        flat.append({"attack": attack, "epsilon": eps, "scope": "overall",
                       "slice": "all", **_slice(sub)})
        for lbl, s in sub.groupby("label"):
            entry["by_label"][str(lbl)] = _slice(s)
            flat.append({"attack": attack, "epsilon": eps, "scope": "label",
                         "slice": str(lbl), **_slice(s)})
        for src, s in sub.groupby("source_dataset"):
            entry["by_source"][str(src)] = _slice(s)
            flat.append({"attack": attack, "epsilon": eps, "scope": "source_dataset",
                         "slice": str(src), **_slice(s)})
        for dt, s in sub[sub["label"] == "drone"].groupby("drone_type"):
            entry["by_drone_type"][str(dt)] = _slice(s)
            flat.append({"attack": attack, "epsilon": eps, "scope": "drone_type",
                         "slice": str(dt), **_slice(s)})
        nested["by_attack_eps"][key] = entry

    flat_df = pd.DataFrame(flat)
    flat_df["asr_ci_lo"] = flat_df["asr_ci95"].apply(lambda x: x[0])
    flat_df["asr_ci_hi"] = flat_df["asr_ci95"].apply(lambda x: x[1])
    flat_df = flat_df.drop(columns=["asr_ci95"])
    return nested, flat_df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-dataset CNN14 -> ProxyAudioCNN transfer.",
    )
    p.add_argument(
        "--attacks", default="FGSM",
        help='Comma-separated subset of {"FGSM","PGD","EOT-PGD"}.',
    )
    p.add_argument(
        "--epsilons",
        default=",".join(str(e) for e in DEFAULT_EPSILONS),
        help="Comma-separated epsilons.",
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="DataLoader batch size.",
    )
    p.add_argument(
        "--pgd-steps", type=int, default=DEFAULT_PGD_STEPS,
        help=f"PGD iterations (default: {DEFAULT_PGD_STEPS}).",
    )
    p.add_argument(
        "--eot-steps", type=int, default=DEFAULT_EOT_STEPS,
        help=f"EOT-PGD iterations (default: {DEFAULT_EOT_STEPS}).",
    )
    p.add_argument(
        "--num-eot-samples", type=int, default=DEFAULT_EOT_SAMPLES,
        help=f"EOT samples per PGD step (default: {DEFAULT_EOT_SAMPLES}).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip (attack, eps) pairs already in the per-sample CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    attacks  = [a.strip().upper().replace("EOT_PGD", "EOT-PGD") for a in args.attacks.split(",") if a.strip()]
    epsilons = [float(e) for e in args.epsilons.split(",") if e.strip()]
    for a in attacks:
        if a not in ("FGSM", "PGD", "EOT-PGD"):
            print(f"ERROR: unknown attack '{a}'", file=sys.stderr)
            sys.exit(1)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Attacks: {attacks}  epsilons: {epsilons}  batch: {args.batch_size}\n")

    rir_kernels = None
    if "EOT-PGD" in attacks:
        print("Precomputing RIR bank (20 rooms, seed=42)...")
        rir_bank = build_rir_bank(n=20, sample_rate=16000, seed=42)
        rir_kernels = build_rir_bank_tensors(rir_bank, max_len=512, device=device)
        print(f"RIR kernels ready: {rir_kernels.shape}\n")

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
    done: set[tuple[str, float]] = set()
    if args.resume and per_sample_csv.exists():
        prev = pd.read_csv(per_sample_csv)
        all_rows = prev.to_dict("records")
        done = {(r["attack"], float(r["epsilon"])) for r in all_rows}
        print(f"[resume] already done: {sorted(done)}")

    print("Running CNN14 -> ProxyAudioCNN transfer ...\n")
    for attack in attacks:
        attack_key = f"{attack}_transfer"
        print(f"  {attack}")
        for eps in epsilons:
            if (attack_key, eps) in done:
                print(f"    {attack_key} eps={eps}  [skipped]")
                continue
            rows = _transfer_pass(
                source, target, mel, loader, attack, eps, device,
                pgd_steps=args.pgd_steps,
                eot_steps=args.eot_steps,
                num_eot_samples=args.num_eot_samples,
                rir_kernels=rir_kernels,
            )
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(per_sample_csv, index=False)
            nested, flat = _aggregate(pd.DataFrame(all_rows), manifest_df)
            flat.to_csv(flat_csv, index=False)
            with open(nested_json, "w") as f:
                json.dump(nested, f, indent=2)

    nested, flat = _aggregate(pd.DataFrame(all_rows), manifest_df)
    print("\n" + "=" * 78)
    print("CROSS-DATASET CNN14 -> ProxyAudioCNN TRANSFER ASR")
    print("=" * 78)
    empty = {"n": 0, "asr": float("nan"),
             "asr_ci95": [float("nan"), float("nan")]}
    for attack in attacks:
        attack_key = f"{attack}_transfer"
        print(f"\n  {attack_key}")
        print(f"  {'eps':<8} {'overall':<24} "
              f"{'alemadi':<22} {'trident':<22} {'wildlife_XC':<22}")
        for eps in epsilons:
            e = nested["by_attack_eps"].get(f"{attack_key}_eps{eps}")
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

    # If the white-box cross-dataset run exists, print BB vs WB side-by-side.
    wb_csv = out_dir / "cross_dataset_attacks.csv"
    if wb_csv.exists() and "FGSM" in attacks:
        wb = pd.read_csv(wb_csv)
        wb_fgsm = wb[(wb["attack"] == "FGSM") & (wb["scope"] == "overall")]
        print("\n  White-box (CNN14) vs Black-box transfer (ProxyAudioCNN) — FGSM")
        print(f"  {'eps':<8} {'WB ASR':>10}   {'BB ASR':>10}   {'ratio':>8}")
        for eps in epsilons:
            wb_row = wb_fgsm[wb_fgsm["epsilon"].round(4) == round(eps, 4)]
            e = nested["by_attack_eps"].get(f"FGSM_transfer_eps{eps}")
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
