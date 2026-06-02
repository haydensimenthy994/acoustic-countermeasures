from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


def compute_perturbation_metrics(original: torch.Tensor, perturbed: torch.Tensor) -> dict:
    """L2, L-inf and SNR (dB) of the perturbation, averaged over the batch."""
    perturbation = perturbed - original

    l2 = perturbation.norm(p=2, dim=-1).mean().item()
    linf = perturbation.abs().max(dim=-1).values.mean().item()

    signal_power = original.pow(2).mean(dim=-1)
    noise_power = perturbation.pow(2).mean(dim=-1)

    # Silent clips would blow up the SNR — leave them out.
    valid_mask = signal_power > 1e-6
    if valid_mask.sum() > 0:
        snr_per_sample = 10 * torch.log10(
            signal_power[valid_mask] / (noise_power[valid_mask] + 1e-10)
        )
        snr = snr_per_sample.mean().item()
    else:
        snr = float('nan')

    return {"l2_norm": l2, "linf_norm": linf, "snr_db": snr}


def fgsm_attack(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    # Clone — otherwise requires_grad sticks on whatever tensor was passed in.
    features = features.detach().clone().to(device).requires_grad_(True)
    labels = labels.to(device)

    criterion = nn.CrossEntropyLoss()
    logits = model(features)
    loss = criterion(logits, labels)
    grad = torch.autograd.grad(loss, features)[0]

    perturbation = epsilon * grad.sign()
    adv_features = features + perturbation

    return adv_features.detach(), perturbation.detach()


def evaluate_fgsm(
    model: nn.Module,
    loader,
    epsilon: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> dict:
    model.eval()
    total = 0
    correct_adv = 0
    confidence_drops = []
    l2_norms = []
    linf_norms = []
    snr_values = []

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            clean_logits = model(features)
            clean_probs = torch.softmax(clean_logits, dim=1)
            clean_preds = clean_logits.argmax(dim=1)

        correct_mask = clean_preds == labels
        if correct_mask.sum() == 0:
            continue

        features_correct = features[correct_mask]
        labels_correct = labels[correct_mask]

        adv_features, _ = fgsm_attack(model, features_correct, labels_correct, epsilon, device)

        metrics = compute_perturbation_metrics(features_correct, adv_features)
        l2_norms.append(metrics["l2_norm"])
        linf_norms.append(metrics["linf_norm"])
        snr_values.append(metrics["snr_db"])

        with torch.no_grad():
            adv_logits = model(adv_features)
            adv_probs = torch.softmax(adv_logits, dim=1)
            adv_preds = adv_logits.argmax(dim=1)

        correct_adv += (adv_preds == labels_correct).sum().item()
        total += correct_mask.sum().item()

        for i in range(len(labels_correct)):
            true_class = labels_correct[i].item()
            clean_conf = clean_probs[correct_mask][i, true_class].item()
            adv_conf = adv_probs[i, true_class].item()
            confidence_drops.append(clean_conf - adv_conf)

    asr = (total - correct_adv) / total if total > 0 else 0.0
    conf_drops = np.array(confidence_drops)

    return {
        "epsilon": epsilon,
        "total_samples": total,
        "attack_success_rate": asr,
        "adv_accuracy": correct_adv / total if total > 0 else 0.0,
        "avg_confidence_drop": float(conf_drops.mean()) if len(conf_drops) > 0 else 0.0,
        "std_confidence_drop": float(conf_drops.std()) if len(conf_drops) > 0 else 0.0,
        "avg_l2_norm": float(np.mean(l2_norms)),
        "avg_linf_norm": float(np.mean(linf_norms)),
        "avg_snr_db": float(np.mean(snr_values)),
    }