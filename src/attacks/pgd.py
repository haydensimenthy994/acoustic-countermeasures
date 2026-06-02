from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from src.attacks.fgsm import compute_perturbation_metrics


def pgd_attack(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.01,
    alpha: float = 0.001,
    num_steps: int = 40,
    device: torch.device = torch.device("cpu"),
    random_start: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PGD attack inside an L-inf ball of radius epsilon.

    Set `random_start=False` for black-box transfer sweeps so ASR scales
    monotonically with epsilon (otherwise the random init muddies the curve).
    """
    model.eval()
    features = features.detach().to(device)
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss()

    if random_start:
        delta = torch.empty_like(features).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(features)
    delta = delta.to(device)

    for _ in range(num_steps):
        delta = delta.detach().requires_grad_(True)
        adv_input = features + delta

        logits = model(adv_input)
        loss = criterion(logits, labels)
        # Only take grad w.r.t. delta — don't zero model grads, the caller
        # may want them for downstream evaluation.
        grad = torch.autograd.grad(loss, delta)[0]

        delta = delta + alpha * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon).detach()

    adv_features = (features + delta).detach()
    perturbation = delta.detach()
    return adv_features, perturbation


def evaluate_pgd(
    model: nn.Module,
    loader,
    epsilon: float = 0.01,
    alpha: float = None,
    num_steps: int = 40,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run PGD over a loader. Alpha defaults to epsilon/10."""
    if alpha is None:
        alpha = epsilon / 10

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

        # Only attack samples the model already gets right.
        correct_mask = clean_preds == labels
        if correct_mask.sum() == 0:
            continue

        features_correct = features[correct_mask]
        labels_correct = labels[correct_mask]

        adv_features, _ = pgd_attack(
            model, features_correct, labels_correct,
            epsilon=epsilon, alpha=alpha, num_steps=num_steps, device=device
        )

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
        "alpha": alpha,
        "num_steps": num_steps,
        "total_samples": total,
        "attack_success_rate": asr,
        "adv_accuracy": correct_adv / total if total > 0 else 0.0,
        "avg_confidence_drop": float(conf_drops.mean()) if len(conf_drops) > 0 else 0.0,
        "std_confidence_drop": float(conf_drops.std()) if len(conf_drops) > 0 else 0.0,
        "avg_l2_norm": float(np.mean(l2_norms)),
        "avg_linf_norm": float(np.mean(linf_norms)),
        "avg_snr_db": float(np.mean(snr_values)),
    }