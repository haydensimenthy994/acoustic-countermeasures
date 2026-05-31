from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Jamming
# ---------------------------------------------------------------------------

def jamming_attack(
    waveform: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """
    Simulate acoustic jamming by adding broadband Gaussian noise
    at a specified SNR level.
    waveform: [samples]
    """
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
    """
    Evaluate classifier robustness under jamming noise at varying SNR levels.

    For each SNR level, reports BOTH:
      - conditional_asr: (correct_clean AND wrong_jammed) / correct_clean
            — same denominator as the gradient-based attacks, so directly
            comparable on Figure 5.
      - asr: 1 - accuracy on all samples
            — kept for backwards compatibility with older result files.

    Lower SNR = stronger jamming (more noise).
    """
    model.eval()
    results = []

    for snr_db in snr_levels:
        total_all = 0
        correct_all = 0           # for population accuracy
        clean_correct = 0         # denominator for conditional ASR
        flipped = 0               # numerator   for conditional ASR
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

                # Population accuracy / ASR
                correct_all += (preds == labels).sum().item()
                total_all += len(labels)

                # Conditional ASR: only count flips on clips that were
                # correctly classified on the clean signal
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
            "asr": population_asr,                       # legacy field
            "conditional_asr": conditional_asr,          # comparable with gradient attacks
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


# ---------------------------------------------------------------------------
# Drone-recall sanity check (was previously mis-named `evaluate_spoofing`)
# ---------------------------------------------------------------------------

def evaluate_drone_recall(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Sanity check on the trained classifier — NOT an attack.

    Reports:
      - drone_recall   : P(pred = drone | label = drone)
      - no_drone_recall: P(pred = no_drone | label = no_drone)

    This was previously published as `spoofing_asr` in earlier result files,
    which was a misnomer — a drone clip predicted as drone is a true
    positive, not a false alarm. Kept here under its correct name so the
    historical number is reproducible.
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


# ---------------------------------------------------------------------------
# Real acoustic-spoofing baseline
# ---------------------------------------------------------------------------

def evaluate_spoofing(
    model: nn.Module,
    loader: DataLoader,
    snr_levels_db: list,
    device: torch.device = torch.device("cpu"),
    seed: int | None = 42,
) -> list[dict]:
    """
    Real acoustic-spoofing baseline.

    Scenario: an attacker plays drone audio near a microphone that is
    monitoring a real (no-drone) environment. We measure how often the
    drone signal, mixed into the no-drone background at a controlled
    SNR, induces a `drone` prediction on a sample the model previously
    classified correctly as `no_drone`.

    SNR_dB here is the no_drone:drone power ratio. High SNR = quiet
    drone (far away), low SNR = loud drone (close to the mic).

    For each SNR level we report:
      - spoofing_asr : P(pred = drone after mix | clean pred = no_drone, label = no_drone)
                       computed over the clean-correct no_drone subset.
      - confidence drop on the no_drone class.
    """
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Collect clips that the model classifies correctly when clean
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
            # Pick a random drone clip and scale it so the
            # no_drone:drone power ratio matches snr_db.
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

            clean_conf_sum += clean_probs[0, 0].item()    # P(no_drone) before
            mix_conf_sum += mix_probs[0, 0].item()        # P(no_drone) after
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
