from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


def jamming_attack(
    waveform: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Mix broadband Gaussian noise into the waveform at the given SNR."""
    signal_power = waveform.pow(2).mean()
    if signal_power < 1e-8:
        return waveform
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return waveform + noise


def evaluate_jamming(
    model: nn.Module,
    loader: DataLoader,
    snr_levels: list,
    device: torch.device = torch.device("cpu"),
) -> list[dict]:
    """Sweep classifier accuracy across jamming SNR levels (lower = louder noise).

    Reports both `conditional_asr` (flip rate over clean-correct clips, the
    apples-to-apples version used in Figure 5) and the legacy population-level
    `asr` so older result CSVs still round-trip.
    """
    model.eval()
    results = []

    for snr_db in snr_levels:
        total_all = 0
        correct_all = 0
        clean_correct = 0        # denominator for conditional ASR
        flipped = 0              # numerator
        confidence_list = []

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(device)
                labels = batch["label"].to(device)

                clean_logits = model(features)
                clean_preds = clean_logits.argmax(dim=1)
                clean_correct_mask = clean_preds == labels

                jammed = torch.stack([
                    jamming_attack(features[i], snr_db)
                    for i in range(features.shape[0])
                ])

                logits = model(jammed)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                correct_all += (preds == labels).sum().item()
                total_all += len(labels)

                # Conditional ASR only counts clips that were clean-correct.
                clean_correct += clean_correct_mask.sum().item()
                flipped_mask = clean_correct_mask & (preds != labels)
                flipped += flipped_mask.sum().item()

                for i in range(len(labels)):
                    true_class = labels[i].item()
                    confidence_list.append(probs[i, true_class].item())

        accuracy = correct_all / total_all if total_all > 0 else 0.0
        population_asr = 1.0 - accuracy
        conditional_asr = flipped / clean_correct if clean_correct > 0 else 0.0

        results.append({
            "snr_db": snr_db,
            "accuracy": accuracy,
            "asr": population_asr,                # legacy
            "conditional_asr": conditional_asr,   # comparable with gradient attacks
            "clean_correct": clean_correct,
            "flipped": flipped,
            "total": total_all,
            "avg_confidence": float(np.mean(confidence_list)),
        })

        print(
            f"  SNR={snr_db:>6.1f} dB | Acc={accuracy:.4f} | "
            f"pop ASR={population_asr:.4f} | cond ASR={conditional_asr:.4f} "
            f"({flipped}/{clean_correct})"
        )

    return results


def evaluate_drone_recall(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Per-class recall on the test set. Not an attack — just a sanity check.

    Earlier result files saved this under `spoofing_asr`, which was wrong
    (it's a true positive, not a false alarm). Kept here under the correct
    name so the old number is still reproducible.
    """
    model.eval()

    drone_features = []
    no_drone_features = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"]
            labels = batch["label"]
            for i in range(len(labels)):
                if labels[i].item() == 1:
                    drone_features.append(features[i])
                else:
                    no_drone_features.append(features[i])

    if len(drone_features) == 0 or len(no_drone_features) == 0:
        print("Not enough samples for drone-recall sanity check")
        return {}

    drone_tensor = torch.stack(drone_features).to(device)
    no_drone_tensor = torch.stack(no_drone_features).to(device)

    with torch.no_grad():
        drone_recall = (model(drone_tensor).argmax(dim=1) == 1).float().mean().item()
        no_drone_recall = (model(no_drone_tensor).argmax(dim=1) == 0).float().mean().item()

    print(f"  drone_recall    = {drone_recall:.4f}  (was reported as spoofing_asr)")
    print(f"  no_drone_recall = {no_drone_recall:.4f}")

    return {
        "drone_recall": drone_recall,
        "no_drone_recall": no_drone_recall,
        "num_drone_samples": len(drone_features),
        "num_no_drone_samples": len(no_drone_features),
    }


def evaluate_spoofing(
    model: nn.Module,
    loader: DataLoader,
    snr_levels_db: list,
    device: torch.device = torch.device("cpu"),
    seed: int | None = 42,
) -> list[dict]:
    """Acoustic-spoofing baseline: an attacker plays drone audio over a no-drone clip.

    SNR_dB is the no_drone:drone power ratio — high SNR means the drone is
    faint/far away, low SNR means it's loud/close. `spoofing_asr` is the flip
    rate on clips the model was clean-correct on, so it's comparable to the
    gradient attacks.
    """
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    correct_no_drone = []
    drone_clips = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            preds = model(features).argmax(dim=1)
            for i in range(len(labels)):
                if labels[i].item() == 0 and preds[i].item() == 0:
                    correct_no_drone.append(features[i].cpu())
                if labels[i].item() == 1:
                    drone_clips.append(features[i].cpu())

    if not correct_no_drone or not drone_clips:
        print("Not enough samples for spoofing evaluation")
        return []

    print(
        f"  spoofing pool: {len(correct_no_drone)} clean-correct no_drone, "
        f"{len(drone_clips)} drone clips available to play"
    )

    drone_stack = torch.stack(drone_clips)

    results = []
    for snr_db in snr_levels_db:
        flipped = 0
        clean_conf_sum = 0.0
        mix_conf_sum = 0.0
        n = 0

        for clean_wav in correct_no_drone:
            # Pick a random drone clip and scale it to hit the target SNR.
            drone_wav = drone_stack[np.random.randint(len(drone_stack))]

            sig_pow = clean_wav.pow(2).mean()
            spoof_pow = drone_wav.pow(2).mean()
            if sig_pow < 1e-8 or spoof_pow < 1e-8:
                continue

            target_spoof_pow = sig_pow / (10 ** (snr_db / 10))
            scale = torch.sqrt(target_spoof_pow / spoof_pow)
            mixed = (clean_wav + scale * drone_wav).to(device)

            with torch.no_grad():
                clean_logits = model(clean_wav.unsqueeze(0).to(device))
                mix_logits = model(mixed.unsqueeze(0))
                clean_probs = torch.softmax(clean_logits, dim=1)
                mix_probs = torch.softmax(mix_logits, dim=1)
                mix_pred = mix_logits.argmax(dim=1).item()

            clean_conf_sum += clean_probs[0, 0].item()
            mix_conf_sum += mix_probs[0, 0].item()
            n += 1
            if mix_pred == 1:
                flipped += 1

        asr = flipped / n if n > 0 else 0.0
        clean_conf = clean_conf_sum / n if n > 0 else 0.0
        mix_conf = mix_conf_sum / n if n > 0 else 0.0

        results.append({
            "snr_db": snr_db,
            "spoofing_asr": asr,
            "flipped": flipped,
            "clean_correct_no_drone": n,
            "avg_clean_no_drone_confidence": clean_conf,
            "avg_mix_no_drone_confidence": mix_conf,
        })

        print(
            f"  SNR={snr_db:>6.1f} dB | spoofing ASR={asr:.4f} "
            f"({flipped}/{n}) | "
            f"P(no_drone): {clean_conf:.3f} -> {mix_conf:.3f}"
        )

    return results
