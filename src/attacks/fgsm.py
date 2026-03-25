from __future__ import annotations

import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features = features.to(device).requires_grad_(True)
    labels = labels.to(device)

    criterion = nn.CrossEntropyLoss()
    logits = model(features)
    loss = criterion(logits, labels)
    model.zero_grad()
    loss.backward()

    perturbation = epsilon * features.grad.sign()
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
    avg_conf_drop = sum(confidence_drops) / len(confidence_drops) if confidence_drops else 0.0

    return {
        "epsilon": epsilon,
        "total_samples": total,
        "attack_success_rate": asr,
        "avg_confidence_drop": avg_conf_drop,
        "clean_accuracy": 1.0 - asr,
        "adv_accuracy": correct_adv / total if total > 0 else 0.0,
    }