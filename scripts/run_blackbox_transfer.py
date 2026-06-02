"""Black-box transfer evaluation: craft adversaries on CNN14 (raw waveform),
then evaluate them through the matched LogMelSpectrogram on ProxyAudioCNN.

ProxyAudioCNN was trained from scratch — it never saw CNN14's gradients, so
this is a real black-box transfer setting.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.dataset import DroneAudioDataset
from src.features.spectrograms import LogMelSpectrogram
from src.models.cnn14_proxy import CNN14ProxyClassifier
from src.models.pann_proxy import ProxyAudioCNN
from src.attacks.fgsm import fgsm_attack, compute_perturbation_metrics
from src.attacks.pgd import pgd_attack
from src.attacks.eot_pgd import (
    eot_pgd_attack,
    build_rir_bank,
    build_rir_bank_tensors,
    apply_eot_transforms_batched,
)
from src.utils.seed import set_seed


def evaluate_transfer_fgsm(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    loader: DataLoader,
    mel_transform: LogMelSpectrogram,
    epsilon: float,
    device: torch.device,
) -> dict:
    """FGSM on the source model (CNN14), evaluated through mel on the target."""
    source_model.eval()
    target_model.eval()

    total = 0
    correct_adv = 0
    confidence_drops: list[float] = []
    l2_norms: list[float] = []
    linf_norms: list[float] = []
    snr_values: list[float] = []

    for batch in loader:
        waveforms = batch["features"].to(device)
        labels    = batch["label"].to(device)

        # Conditional ASR: only attack source-correct clips.
        with torch.no_grad():
            src_preds = source_model(waveforms).argmax(dim=1)
        correct_mask = src_preds == labels
        if correct_mask.sum() == 0:
            continue

        wav_correct    = waveforms[correct_mask].detach()
        labels_correct = labels[correct_mask]

        adv_wav, _ = fgsm_attack(
            source_model, wav_correct, labels_correct, epsilon, device
        )

        metrics = compute_perturbation_metrics(wav_correct, adv_wav)
        l2_norms.append(metrics["l2_norm"])
        linf_norms.append(metrics["linf_norm"])
        snr_values.append(metrics["snr_db"])

        with torch.no_grad():
            adv_mel   = mel_transform(adv_wav)
            clean_mel = mel_transform(wav_correct)

            clean_probs_tgt = torch.softmax(target_model(clean_mel), dim=1)
            adv_logits_tgt  = target_model(adv_mel)
            adv_probs_tgt   = torch.softmax(adv_logits_tgt, dim=1)
            adv_preds_tgt   = adv_logits_tgt.argmax(dim=1)

        correct_adv += (adv_preds_tgt == labels_correct).sum().item()
        total       += labels_correct.shape[0]

        for i in range(len(labels_correct)):
            tc = labels_correct[i].item()
            confidence_drops.append(
                clean_probs_tgt[i, tc].item() - adv_probs_tgt[i, tc].item()
            )

    asr = (total - correct_adv) / total if total > 0 else 0.0
    conf_drops = np.array(confidence_drops)
    return {
        "attack":              "FGSM",
        "epsilon":             epsilon,
        "total_samples":       total,
        "transfer_asr":        asr,
        "adv_accuracy":        correct_adv / total if total > 0 else 0.0,
        "avg_confidence_drop": float(conf_drops.mean()) if len(conf_drops) else 0.0,
        "std_confidence_drop": float(conf_drops.std())  if len(conf_drops) else 0.0,
        "avg_l2_norm":         float(np.mean(l2_norms))   if l2_norms  else 0.0,
        "avg_linf_norm":       float(np.mean(linf_norms)) if linf_norms else 0.0,
        "avg_snr_db":          float(np.mean(snr_values)) if snr_values else 0.0,
    }


def evaluate_transfer_pgd(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    loader: DataLoader,
    mel_transform: LogMelSpectrogram,
    epsilon: float,
    num_steps: int = 40,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """PGD on the source model (CNN14), evaluated through mel on the target."""
    source_model.eval()
    target_model.eval()
    alpha = epsilon / 10

    total = 0
    correct_adv = 0
    confidence_drops: list[float] = []
    l2_norms: list[float] = []
    linf_norms: list[float] = []
    snr_values: list[float] = []

    for batch in loader:
        waveforms = batch["features"].to(device)
        labels    = batch["label"].to(device)

        with torch.no_grad():
            src_preds = source_model(waveforms).argmax(dim=1)
        correct_mask = src_preds == labels
        if correct_mask.sum() == 0:
            continue

        wav_correct    = waveforms[correct_mask].detach()
        labels_correct = labels[correct_mask]

        # random_start=False — keeps transfer ASR monotonic in epsilon.
        adv_wav, _ = pgd_attack(
            source_model, wav_correct, labels_correct,
            epsilon=epsilon, alpha=alpha, num_steps=num_steps, device=device,
            random_start=False,
        )

        metrics = compute_perturbation_metrics(wav_correct, adv_wav)
        l2_norms.append(metrics["l2_norm"])
        linf_norms.append(metrics["linf_norm"])
        snr_values.append(metrics["snr_db"])

        with torch.no_grad():
            adv_mel   = mel_transform(adv_wav)
            clean_mel = mel_transform(wav_correct)

            clean_probs_tgt = torch.softmax(target_model(clean_mel), dim=1)
            adv_logits_tgt  = target_model(adv_mel)
            adv_probs_tgt   = torch.softmax(adv_logits_tgt, dim=1)
            adv_preds_tgt   = adv_logits_tgt.argmax(dim=1)

        correct_adv += (adv_preds_tgt == labels_correct).sum().item()
        total       += labels_correct.shape[0]

        for i in range(len(labels_correct)):
            tc = labels_correct[i].item()
            confidence_drops.append(
                clean_probs_tgt[i, tc].item() - adv_probs_tgt[i, tc].item()
            )

    asr = (total - correct_adv) / total if total > 0 else 0.0
    conf_drops = np.array(confidence_drops)
    return {
        "attack":              "PGD",
        "epsilon":             epsilon,
        "num_steps":           num_steps,
        "total_samples":       total,
        "transfer_asr":        asr,
        "adv_accuracy":        correct_adv / total if total > 0 else 0.0,
        "avg_confidence_drop": float(conf_drops.mean()) if len(conf_drops) else 0.0,
        "std_confidence_drop": float(conf_drops.std())  if len(conf_drops) else 0.0,
        "avg_l2_norm":         float(np.mean(l2_norms))   if l2_norms  else 0.0,
        "avg_linf_norm":       float(np.mean(linf_norms)) if linf_norms else 0.0,
        "avg_snr_db":          float(np.mean(snr_values)) if snr_values else 0.0,
    }


def evaluate_transfer_eot_pgd(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    loader: DataLoader,
    mel_transform: LogMelSpectrogram,
    epsilon: float,
    rir_kernels: torch.Tensor,
    num_steps: int = 20,
    num_eot_samples: int = 5,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """EOT-PGD on the source, mel-transformed onto the target. Digital + OTA ASR."""
    source_model.eval()
    target_model.eval()
    alpha    = epsilon / 10
    num_rirs = rir_kernels.shape[0]

    total               = 0
    correct_adv_digital = 0
    correct_adv_ota     = 0
    confidence_drops: list[float] = []
    snr_values: list[float] = []

    for batch_idx, batch in enumerate(loader):
        waveforms = batch["features"].to(device)
        labels    = batch["label"].to(device)

        with torch.no_grad():
            src_preds = source_model(waveforms).argmax(dim=1)
        correct_mask = src_preds == labels
        if correct_mask.sum() == 0:
            continue

        wav_correct    = waveforms[correct_mask].detach()
        labels_correct = labels[correct_mask]

        print(
            f"  EOT batch {batch_idx}: {wav_correct.shape[0]} samples, eps={epsilon}",
            flush=True,
        )

        adv_wav, _ = eot_pgd_attack(
            source_model, wav_correct, labels_correct,
            epsilon=epsilon, alpha=alpha,
            num_steps=num_steps, num_eot_samples=num_eot_samples,
            device=device, rir_kernels=rir_kernels,
            verbose=(batch_idx == 0),
            random_start=False,
        )

        metrics = compute_perturbation_metrics(wav_correct, adv_wav)
        snr_values.append(metrics["snr_db"])

        # Digital target evaluation.
        with torch.no_grad():
            adv_mel   = mel_transform(adv_wav)
            clean_mel = mel_transform(wav_correct)

            clean_probs_tgt = torch.softmax(target_model(clean_mel), dim=1)
            adv_logits_tgt  = target_model(adv_mel)
            adv_probs_tgt   = torch.softmax(adv_logits_tgt, dim=1)
            adv_preds_tgt   = adv_logits_tgt.argmax(dim=1)

        correct_adv_digital += (adv_preds_tgt == labels_correct).sum().item()

        # OTA: run the adv waveform through the same RIR/gain/noise the
        # attack used, then convert to mel for the target — 5-vote majority.
        ota_votes = torch.zeros(len(labels_correct), dtype=torch.long, device=device)
        for _ in range(5):
            with torch.no_grad():
                transformed_wav = apply_eot_transforms_batched(
                    adv_wav,
                    rir_kernels=rir_kernels,
                    with_noise=True,
                )
                transformed_mel = mel_transform(transformed_wav)
                ota_preds = target_model(transformed_mel).argmax(dim=1)
            ota_votes += (ota_preds != labels_correct).long()

        ota_success      = ota_votes >= 3
        correct_adv_ota += (~ota_success).sum().item()
        total           += labels_correct.shape[0]

        for i in range(len(labels_correct)):
            tc = labels_correct[i].item()
            confidence_drops.append(
                clean_probs_tgt[i, tc].item() - adv_probs_tgt[i, tc].item()
            )

    digital_asr = (total - correct_adv_digital) / total if total > 0 else 0.0
    ota_asr     = (total - correct_adv_ota)     / total if total > 0 else 0.0
    conf_drops  = np.array(confidence_drops)

    return {
        "attack":               "EOT-PGD",
        "epsilon":              epsilon,
        "num_steps":            num_steps,
        "num_eot_samples":      num_eot_samples,
        "total_samples":        total,
        "transfer_asr_digital": digital_asr,
        "transfer_asr_ota":     ota_asr,
        "adv_accuracy_digital": correct_adv_digital / total if total > 0 else 0.0,
        "adv_accuracy_ota":     correct_adv_ota     / total if total > 0 else 0.0,
        "avg_confidence_drop":  float(conf_drops.mean()) if len(conf_drops) else 0.0,
        "std_confidence_drop":  float(conf_drops.std())  if len(conf_drops) else 0.0,
        "avg_snr_db":           float(np.mean(snr_values)) if snr_values else 0.0,
    }


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Source: white-box CNN14, only used to craft perturbations.
    print("Loading source model (CNN14)...")
    source_model = CNN14ProxyClassifier(
        num_classes=2,
        pretrained_path="outputs/checkpoints/Cnn14_16k_mAP=0.438.pth",
    ).to(device)
    source_model.load_state_dict(torch.load(
        "outputs/checkpoints/best_model_cnn14.pt",
        map_location=device,
        weights_only=False,
    ))
    source_model.eval()
    print("  CNN14 loaded (val acc 98.29%)")

    # Target: black-box ProxyAudioCNN. Gradients never flow through this one.
    print("Loading target model (ProxyAudioCNN)...")
    target_model = ProxyAudioCNN(
        input_channels=1,
        num_classes=2,
        dropout=0.3,
    ).to(device)
    target_model.load_state_dict(torch.load(
        "outputs/checkpoints/best_model.pt",
        map_location=device,
        weights_only=False,
    ))
    target_model.eval()
    print("  ProxyAudioCNN loaded (val acc 97.07%)")

    # Must be the same class/defaults used during ProxyAudioCNN training.
    print("Building mel transform (LogMelSpectrogram)...")
    mel_transform = LogMelSpectrogram().to(device)
    print("  Mel transform ready")

    print("\nLoading test dataset...")
    test_dataset = DroneAudioDataset(
        metadata_csv="data/metadata/split_metadata.csv",
        config_path="configs/data.yaml",
        split="test",
        fixed_duration_sec=5.0,
        use_raw_waveform=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=0,
    )
    print(f"  Test samples: {len(test_dataset)}")

    print("\nPrecomputing RIR bank (20 rooms, seed=42)...")
    rir_bank    = build_rir_bank(n=20, sample_rate=16000, seed=42)
    rir_kernels = build_rir_bank_tensors(rir_bank, max_len=512, device=device)
    print(f"  RIR kernels ready: {rir_kernels.shape}")

    epsilons_fgsm_pgd = [0.001, 0.005, 0.01, 0.02, 0.05]
    epsilons_eot      = [0.001, 0.005, 0.01, 0.02, 0.05]

    fgsm_results = []
    pgd_results  = []
    eot_results  = []

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save_partial():
        # Called after every epsilon so Ctrl+C never wipes a run.
        if pgd_results:
            pd.DataFrame(pgd_results).to_csv(output_dir / "blackbox_pgd_transfer.csv", index=False)
        if eot_results:
            pd.DataFrame(eot_results).to_csv(output_dir / "blackbox_eot_transfer.csv", index=False)
        if fgsm_results:
            pd.DataFrame(fgsm_results).to_csv(output_dir / "blackbox_fgsm_transfer.csv", index=False)
        with open(output_dir / "blackbox_transfer_results.json", "w") as f:
            json.dump({
                "fgsm_transfer": fgsm_results,
                "pgd_transfer":  pgd_results,
                "eot_transfer":  eot_results,
            }, f, indent=2)

    # Run PGD/EOT first — FGSM mutates requires_grad on the batch tensors,
    # so doing it last avoids cross-contamination between attacks.
    print("\n" + "=" * 60)
    print("PGD TRANSFER (CNN14 → ProxyAudioCNN)")
    print("=" * 60)
    print(f"{'Epsilon':<10} {'Transfer ASR':<16} {'Conf Drop':<12} {'SNR(dB)'}")
    print("-" * 55)
    for eps in epsilons_fgsm_pgd:
        r = evaluate_transfer_pgd(
            source_model, target_model, test_loader,
            mel_transform, eps, num_steps=40, device=device,
        )
        pgd_results.append(r)
        _save_partial()
        print(
            f"{r['epsilon']:<10.3f} "
            f"{r['transfer_asr']:<16.4f} "
            f"{r['avg_confidence_drop']:<12.4f} "
            f"{r['avg_snr_db']:.2f}  [saved]"
        )

    print("\n" + "=" * 60)
    print("EOT-PGD TRANSFER (CNN14 → ProxyAudioCNN)")
    print("=" * 60)
    print(f"{'Epsilon':<10} {'Digital ASR':<14} {'OTA ASR':<12} {'SNR(dB)'}")
    print("-" * 50)
    for eps in epsilons_eot:
        print(f"\n  Running EOT epsilon={eps}...", flush=True)
        r = evaluate_transfer_eot_pgd(
            source_model, target_model, test_loader,
            mel_transform, eps,
            rir_kernels=rir_kernels,
            num_steps=20, num_eot_samples=5,
            device=device,
        )
        eot_results.append(r)
        _save_partial()
        print(
            f"\n{r['epsilon']:<10.3f} "
            f"{r['transfer_asr_digital']:<14.4f} "
            f"{r['transfer_asr_ota']:<12.4f} "
            f"{r['avg_snr_db']:.2f}  [saved]"
        )

    print("\n" + "=" * 60)
    print("FGSM TRANSFER (CNN14 → ProxyAudioCNN)")
    print("=" * 60)
    print(f"{'Epsilon':<10} {'Transfer ASR':<16} {'Conf Drop':<12} {'SNR(dB)'}")
    print("-" * 55)
    for eps in epsilons_fgsm_pgd:
        r = evaluate_transfer_fgsm(
            source_model, target_model, test_loader,
            mel_transform, eps, device,
        )
        fgsm_results.append(r)
        _save_partial()
        print(
            f"{r['epsilon']:<10.3f} "
            f"{r['transfer_asr']:<16.4f} "
            f"{r['avg_confidence_drop']:<12.4f} "
            f"{r['avg_snr_db']:.2f}  [saved]"
        )

    all_results = {
        "fgsm_transfer": fgsm_results,
        "pgd_transfer":  pgd_results,
        "eot_transfer":  eot_results,
    }

    # Pull white-box numbers from the CSVs so the summary stays in sync
    # if the upstream experiments are re-run.
    print("\n" + "=" * 60)
    print("SUMMARY — Transfer ASR vs White-box ASR (ε = 0.001)")
    print("=" * 60)
    print(f"{'Attack':<12} {'White-box ASR':<18} {'Transfer ASR'}")
    print("-" * 45)

    def _wb_asr(csv_path: str, asr_col: str = "attack_success_rate") -> str:
        path = Path(csv_path)
        if not path.exists():
            return "n/a"
        df = pd.read_csv(path)
        row = df[df["epsilon"].round(4) == 0.001]
        if row.empty:
            return "n/a"
        return f"{row.iloc[0][asr_col] * 100:.2f}%"

    fgsm_wb = _wb_asr("outputs/results/fgsm_results_cnn14.csv")
    pgd_wb = _wb_asr("outputs/results/pgd_results_cnn14.csv")
    eot_wb_dig = _wb_asr("outputs/results/eot_pgd_results_cnn14.csv", "digital_asr")
    eot_wb_ota = _wb_asr("outputs/results/eot_pgd_results_cnn14.csv", "ota_asr")

    print(f"{'FGSM':<12} {fgsm_wb:<18} {fgsm_results[0]['transfer_asr']*100:.2f}%")
    print(f"{'PGD':<12} {pgd_wb:<18} {pgd_results[0]['transfer_asr']*100:.2f}%")
    print(f"{'EOT-PGD':<12} {eot_wb_dig + ' (dig)':<18} {eot_results[0]['transfer_asr_digital']*100:.2f}% (dig)")
    print(f"{'EOT-PGD':<12} {eot_wb_ota + ' (OTA)':<18} {eot_results[0]['transfer_asr_ota']*100:.2f}% (OTA)")
    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()