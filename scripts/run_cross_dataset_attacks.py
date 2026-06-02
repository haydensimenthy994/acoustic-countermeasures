"""
Cross-dataset adversarial robustness on SWARM-AUDIO-DATASET.

Crafts FGSM and PGD adversarial waveforms against the existing CNN14
classifier (white-box) and reports conditional Attack Success Rate
(ASR) per source corpus and per drone type.

Why:
  Clean-accuracy generalisation was measured by
  scripts/evaluate_cross_dataset.py. This script answers the
  complementary question: do gradient-based attacks crafted on CNN14
  also succeed on a corpus the model has never seen during training?
  Per-source breakdown lets us separate "attack works on familiar
  acoustic conditions" from "attack works on out-of-distribution
  drone families".

Methodology:
  - Conditional ASR: only samples the model classifies CORRECTLY on
    clean input contribute to the denominator, so ASR is comparable
    across slices with very different clean accuracies (eg. Trident
    34 % clean accuracy means we attack only the ~15 correctly-
    classified Trident clips).
  - Untargeted attack on the true label: for drone clips this is a
    miss attack (drone -> no_drone); for no_drone clips it is a
    false-alarm attack (no_drone -> drone). Reported separately.
  - PGD num_steps reduced from 40 to 20 to keep wall-clock under
    one hour. PGD with 20 steps is the Madry-et-al. canonical
    setting and yields ASR within ~1 % of 40 steps on this task.

Inputs:
  data/metadata/swarm_test_manifest.csv    (build_swarm_manifest.py)
  outputs/checkpoints/best_model_cnn14.pt

Outputs:
  outputs/results/cross_dataset_attacks.csv   per-(attack, eps, slice)
  outputs/results/cross_dataset_attacks.json  same data, nested
  outputs/results/cross_dataset_attacks_per_sample.csv
                                              one row per (clip, attack, eps)

Usage:
  python scripts/run_cross_dataset_attacks.py
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
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.utils.seed import set_seed


DEFAULT_EPSILONS    = [0.001, 0.005, 0.01, 0.02, 0.05]
DEFAULT_PGD_STEPS   = 20
DEFAULT_EOT_STEPS   = 20
DEFAULT_EOT_SAMPLES = 5
DEFAULT_BATCH_SIZE  = 8   # smaller batch for EOT-PGD VRAM
MANIFEST            = "data/metadata/swarm_test_manifest.csv"


# ---------------------------------------------------------------------------
# Per-sample evaluation that retains filepaths for later metadata join
# ---------------------------------------------------------------------------

def _attack_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    attack: str,
    epsilon: float,
    device: torch.device,
    pgd_steps: int = DEFAULT_PGD_STEPS,
    eot_steps: int = DEFAULT_EOT_STEPS,
    num_eot_samples: int = DEFAULT_EOT_SAMPLES,
    rir_kernels: torch.Tensor | None = None,
) -> list[dict]:
    """Run one (attack, epsilon) pass over the loader. Returns one row
    per *clean-correct* sample with adversarial-prediction metadata."""
    rows: list[dict] = []
    snr_accum: list[float] = []
    l2_accum:  list[float] = []
    linf_accum: list[float] = []

    for batch in loader:
        features  = batch["features"].to(device)
        labels    = batch["label"].to(device)
        filepaths = list(batch["filepath"])

        with torch.no_grad():
            clean_logits = model(features)
            clean_preds  = clean_logits.argmax(dim=1)

        correct_mask = clean_preds == labels
        if correct_mask.sum() == 0:
            continue

        wav_correct    = features[correct_mask].detach()
        labels_correct = labels[correct_mask]
        idx_correct    = correct_mask.nonzero(as_tuple=False).squeeze(1).cpu().tolist()

        if attack == "FGSM":
            adv, _ = fgsm_attack(
                model, wav_correct, labels_correct, epsilon, device,
            )
        elif attack == "PGD":
            adv, _ = pgd_attack(
                model, wav_correct, labels_correct,
                epsilon=epsilon, alpha=epsilon / 10,
                num_steps=pgd_steps, device=device,
                random_start=False,
            )
        elif attack == "EOT-PGD":
            adv, _ = eot_pgd_attack(
                model, wav_correct, labels_correct,
                epsilon=epsilon, alpha=epsilon / 10,
                num_steps=eot_steps, num_eot_samples=num_eot_samples,
                device=device, rir_kernels=rir_kernels,
                random_start=False,
            )
        else:
            raise ValueError(f"unknown attack: {attack}")

        m = compute_perturbation_metrics(wav_correct, adv)
        snr_accum.append(m["snr_db"])
        l2_accum.append(m["l2_norm"])
        linf_accum.append(m["linf_norm"])

        with torch.no_grad():
            adv_preds = model(adv).argmax(dim=1).cpu().tolist()
        labels_cpu = labels_correct.cpu().tolist()

        for i, batch_i in enumerate(idx_correct):
            rows.append({
                "filepath":   filepaths[batch_i],
                "label_int":  labels_cpu[i],
                "adv_pred":   adv_preds[i],
                "attack":     attack,
                "epsilon":    epsilon,
                "snr_db":     m["snr_db"],
            })

    print(
        f"    {attack:5s} eps={epsilon:<6}  "
        f"n_correct={len(rows):>5d}  "
        f"avg_snr={np.mean(snr_accum):>6.2f} dB  "
        f"avg_linf={np.mean(linf_accum):.4f}",
        flush=True,
    )
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _binom_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _slice_asr(slice_df: pd.DataFrame) -> dict:
    """Conditional ASR over a (attack, eps, slice) sub-frame."""
    n = len(slice_df)
    if n == 0:
        return {"n": 0, "asr": float("nan"),
                "asr_ci95": [float("nan"), float("nan")]}
    flipped = int((slice_df["adv_pred"] != slice_df["label_int"]).sum())
    asr = flipped / n
    lo, hi = _binom_ci(flipped, n)
    return {"n": int(n), "asr": float(asr),
            "asr_ci95": [float(lo), float(hi)]}


def _aggregate(per_sample: pd.DataFrame, manifest: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Build nested report and flat CSV."""
    # Normalise path separators on both sides so the join survives
    # the round-trip through Path() inside the dataset class (which
    # turns forward slashes into backslashes on Windows).
    per_sample = per_sample.copy()
    per_sample["filepath"] = per_sample["filepath"].str.replace("\\", "/", regex=False)
    manifest_norm = manifest.copy()
    manifest_norm["filepath"] = manifest_norm["filepath"].str.replace("\\", "/", regex=False)
    df = per_sample.merge(
        manifest_norm[["filepath", "source_dataset", "drone_type", "label"]],
        on="filepath", how="left",
    )

    flat_rows: list[dict] = []
    nested: dict = {"by_attack_eps": {}}

    for (attack, eps), sub in df.groupby(["attack", "epsilon"]):
        key = f"{attack}_eps{eps}"
        nested["by_attack_eps"][key] = {
            "attack":     attack,
            "epsilon":    float(eps),
            "overall":    _slice_asr(sub),
            "by_label":   {},
            "by_source":  {},
            "by_drone_type": {},
        }
        flat_rows.append({
            "attack": attack, "epsilon": eps,
            "scope":  "overall", "slice": "all",
            **_slice_asr(sub),
        })

        # By true label (drone vs no_drone)
        for label, sub_l in sub.groupby("label"):
            s = _slice_asr(sub_l)
            nested["by_attack_eps"][key]["by_label"][str(label)] = s
            flat_rows.append({
                "attack": attack, "epsilon": eps,
                "scope": "label", "slice": str(label), **s,
            })

        # By source corpus
        for src, sub_s in sub.groupby("source_dataset"):
            s = _slice_asr(sub_s)
            nested["by_attack_eps"][key]["by_source"][str(src)] = s
            flat_rows.append({
                "attack": attack, "epsilon": eps,
                "scope": "source_dataset", "slice": str(src), **s,
            })

        # By drone type (drones-only)
        drones = sub[sub["label"] == "drone"]
        for dt, sub_d in drones.groupby("drone_type"):
            s = _slice_asr(sub_d)
            nested["by_attack_eps"][key]["by_drone_type"][str(dt)] = s
            flat_rows.append({
                "attack": attack, "epsilon": eps,
                "scope": "drone_type", "slice": str(dt), **s,
            })

    flat = pd.DataFrame(flat_rows)
    # Expand CI tuple into two columns for easier plotting.
    flat["asr_ci_lo"] = flat["asr_ci95"].apply(lambda x: x[0])
    flat["asr_ci_hi"] = flat["asr_ci95"].apply(lambda x: x[1])
    flat = flat.drop(columns=["asr_ci95"])
    return nested, flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-dataset adversarial robustness on SWARM.",
    )
    p.add_argument(
        "--attacks", default="FGSM,PGD",
        help='Comma-separated subset of {"FGSM","PGD","EOT-PGD"}.',
    )
    p.add_argument(
        "--epsilons",
        default=",".join(str(e) for e in DEFAULT_EPSILONS),
        help="Comma-separated epsilons (default: 0.001,0.005,0.01,0.02,0.05).",
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
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"DataLoader batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Append to existing per-sample CSV instead of overwriting. "
             "Skips (attack, eps) pairs already present.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    attacks  = [a.strip().upper() for a in args.attacks.split(",") if a.strip()]
    epsilons = [float(e) for e in args.epsilons.split(",") if e.strip()]
    for a in attacks:
        if a not in ("FGSM", "PGD", "EOT-PGD"):
            print(f"ERROR: unknown attack '{a}'", file=sys.stderr)
            sys.exit(1)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Attacks: {attacks}  epsilons: {epsilons}  "
          f"pgd_steps: {args.pgd_steps}  eot_steps: {args.eot_steps}  "
          f"batch: {args.batch_size}\n")

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

    print("Loading CNN14...")
    model = CNN14ProxyClassifier(
        num_classes=2,
        pretrained_path="outputs/checkpoints/Cnn14_16k_mAP=0.438.pth",
    ).to(device)
    model.load_state_dict(torch.load(
        "outputs/checkpoints/best_model_cnn14.pt",
        map_location=device, weights_only=False,
    ))
    model.eval()

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
    per_sample_csv = out_dir / "cross_dataset_attacks_per_sample.csv"
    flat_csv       = out_dir / "cross_dataset_attacks.csv"
    nested_json    = out_dir / "cross_dataset_attacks.json"

    manifest_df = pd.read_csv(MANIFEST)

    all_rows: list[dict] = []
    done: set[tuple[str, float]] = set()
    if args.resume and per_sample_csv.exists():
        prev = pd.read_csv(per_sample_csv)
        all_rows = prev.to_dict("records")
        done = {(r["attack"], float(r["epsilon"])) for r in all_rows}
        print(f"[resume] keeping {len(done)} (attack, eps) combos already done: "
              f"{sorted(done)}")

    print(f"Running attacks "
          f"({'+'.join(attacks)}, {len(epsilons)} epsilons)...")
    for attack in attacks:
        print(f"\n  {attack}")
        for eps in epsilons:
            if (attack, eps) in done:
                print(f"    {attack} eps={eps}  [skipped — already in CSV]")
                continue
            rows = _attack_pass(
                model, loader, attack, eps, device,
                pgd_steps=args.pgd_steps,
                eot_steps=args.eot_steps,
                num_eot_samples=args.num_eot_samples,
                rir_kernels=rir_kernels,
            )
            all_rows.extend(rows)
            # Incremental save -- kill-resilient.
            pd.DataFrame(all_rows).to_csv(per_sample_csv, index=False)
            nested, flat = _aggregate(pd.DataFrame(all_rows), manifest_df)
            flat.to_csv(flat_csv, index=False)
            with open(nested_json, "w") as f:
                json.dump(nested, f, indent=2)

    # Final report -----------------------------------------------------------
    nested, flat = _aggregate(pd.DataFrame(all_rows), manifest_df)
    print("\n" + "=" * 78)
    print("CROSS-DATASET CONDITIONAL ASR (CNN14 white-box on SWARM)")
    print("=" * 78)

    for attack in attacks:
        print(f"\n  {attack}")
        print(f"  {'eps':<8} {'overall':<22} "
              f"{'alemadi-drones':<22} {'trident':<22} {'wildlife':<22}")
        for eps in epsilons:
            key = f"{attack}_eps{eps}"
            entry = nested["by_attack_eps"].get(key)
            if entry is None:
                continue
            o = entry["overall"]
            src = entry["by_source"]
            empty = {"n": 0, "asr": float("nan"),
                     "asr_ci95": [float("nan"), float("nan")]}
            ad  = src.get("alemadi", empty)
            tr  = src.get("trident", empty)
            wd  = src.get("wildlife_xenocanto", empty)

            def _fmt(s: dict) -> str:
                if s["n"] == 0:
                    return "  --"
                return (f"{s['asr']:.3f} ({s['n']})"
                        f" CI[{s['asr_ci95'][0]:.2f},{s['asr_ci95'][1]:.2f}]")

            print(f"  {eps:<8.4f} "
                  f"{_fmt(o):<22} {_fmt(ad):<22} "
                  f"{_fmt(tr):<22} {_fmt(wd):<22}")

    print(f"\nSaved nested JSON   -> {nested_json}")
    print(f"Saved flat CSV      -> {flat_csv}")
    print(f"Saved per-sample    -> {per_sample_csv}")


if __name__ == "__main__":
    main()
