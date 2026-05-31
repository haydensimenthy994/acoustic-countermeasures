from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import pyroomacoustics as pra

from src.attacks.fgsm import compute_perturbation_metrics


def simulate_rir(
    sample_rate: int = 16000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a random Room Impulse Response using pyroomacoustics.

    If `rng` is provided, all randomness comes from it and the bank
    is bit-reproducible. If None, falls back to global numpy state
    (legacy behaviour for back-compat).
    """
    r = rng if rng is not None else np.random
    length = r.uniform(3.0, 10.0)
    width = r.uniform(3.0, 8.0)
    height = r.uniform(2.5, 4.0)

    absorption = r.uniform(0.1, 0.5)
    room = pra.ShoeBox(
        [length, width, height],
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=10,
    )
    room.add_source([
        r.uniform(0.5, length - 0.5),
        r.uniform(0.5, width - 0.5),
        r.uniform(0.5, height - 0.5),
    ])
    room.add_microphone([
        r.uniform(0.5, length - 0.5),
        r.uniform(0.5, width - 0.5),
        r.uniform(0.5, height - 0.5),
    ])
    room.compute_rir()
    rir = room.rir[0][0]
    rir = rir / (np.abs(rir).max() + 1e-8)
    return rir.astype(np.float32)


def build_rir_bank(
    n: int = 20,
    sample_rate: int = 16000,
    seed: int | None = 42,
) -> list[np.ndarray]:
    """Convenience helper — seeded RIR bank for reproducible runs."""
    rng = np.random.default_rng(seed) if seed is not None else None
    return [simulate_rir(sample_rate=sample_rate, rng=rng) for _ in range(n)]


def build_rir_bank_tensors(
    rir_bank: list,
    max_len: int = 512,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convert list of RIR numpy arrays into a batched GPU tensor for fast convolution.
    Returns: [num_rirs, 1, max_len]
    """
    tensors = []
    for rir in rir_bank:
        r = rir[:max_len]
        if len(r) < max_len:
            r = np.pad(r, (0, max_len - len(r)))
        t = torch.tensor(r, dtype=torch.float32, device=device).flip(0)
        tensors.append(t)
    return torch.stack(tensors).unsqueeze(1)  # [N, 1, max_len]


def _apply_rir_differentiable(
    waveform: torch.Tensor,
    rir_kernel: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable RIR convolution that keeps the computation graph alive.
    waveform:   [samples] — must have requires_grad lineage
    rir_kernel: [1, 1, kernel_len] — pre-flipped, treated as a fixed filter
    Returns:    [samples] with grad flowing back to waveform
    """
    wav = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
    pad = rir_kernel.shape[-1] - 1
    convolved = torch.nn.functional.conv1d(wav, rir_kernel, padding=pad)
    convolved = convolved[0, 0, : waveform.shape[-1]]

    # Normalise energy to match the input.  .clamp keeps grads finite.
    peak = convolved.abs().max().clamp(min=1e-8)
    ref = waveform.abs().max().clamp(min=1e-8)
    convolved = convolved / peak * ref
    return convolved


def _apply_rir_no_grad(
    waveform: torch.Tensor,
    rir_kernel: torch.Tensor,
) -> torch.Tensor:
    """Non-differentiable RIR convolution for OTA evaluation."""
    wav = waveform.unsqueeze(0).unsqueeze(0)
    pad = rir_kernel.shape[-1] - 1
    convolved = torch.nn.functional.conv1d(wav, rir_kernel, padding=pad)
    convolved = convolved[0, 0, : waveform.shape[-1]]
    peak = convolved.abs().max()
    if peak > 1e-8:
        convolved = convolved / peak * waveform.abs().max()
    return convolved


# ---------------------------------------------------------------------------
# Batched (vectorised) EOT transform stack
# ---------------------------------------------------------------------------

def apply_eot_transforms_batched(
    waveforms: torch.Tensor,
    rir_kernels: torch.Tensor | None,
    with_noise: bool = True,
    gain_range: tuple[float, float] = (0.7, 1.3),
    snr_range_db: tuple[float, float] = (20.0, 40.0),
) -> torch.Tensor:
    """
    Apply the EOT transform stack (random RIR + gain + optional Gaussian
    noise) to a whole batch of waveforms in one shot.

    This replaces the previous per-sample Python loop, which serialised
    400 single-element conv1d + GPU sync calls per batch and made
    EOT-PGD ~20x slower than necessary.

    Args:
        waveforms:    [B, samples] — may carry grad lineage from delta.
        rir_kernels:  [N, 1, K] pre-flipped RIRs on the same device, or None.
        with_noise:   if True, add Gaussian noise at SNR uniformly sampled
                      from `snr_range_db` per batch element.
        gain_range:   uniform sample range for the per-sample gain.
        snr_range_db: uniform sample range for the per-sample SNR (dB).

    Returns:
        [B, samples] transformed waveforms; gradient flows back through
        `waveforms` if it had requires_grad set.

    Random parameters (which RIR, gain, SNR) are drawn from torch's
    global RNG (which `src.utils.seed.set_seed(42)` seeds at startup),
    so calls are bit-reproducible across runs with the same seed.
    """
    B, T = waveforms.shape
    device = waveforms.device

    # 1) RIR convolution — pick one RIR per batch element, apply via
    #    grouped conv1d so the whole batch is convolved in a single call.
    if rir_kernels is not None and rir_kernels.shape[0] > 0:
        N = rir_kernels.shape[0]
        K = rir_kernels.shape[-1]
        idx = torch.randint(0, N, (B,), device=device)
        selected = rir_kernels[idx]                       # [B, 1, K]
        x = waveforms.unsqueeze(0)                        # [1, B, T]
        pad = K - 1
        convolved = torch.nn.functional.conv1d(
            x, selected, padding=pad, groups=B,
        )                                                 # [1, B, T+K-1]
        convolved = convolved[:, :, :T].squeeze(0)        # [B, T]

        # Peak-normalise per sample to match the input's amplitude.
        peak = convolved.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        ref = waveforms.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        x = convolved / peak * ref
    else:
        x = waveforms

    # 2) Random per-sample gain (broadcast multiply).
    scales = torch.empty(B, 1, device=device).uniform_(
        gain_range[0], gain_range[1],
    )
    x = x * scales

    # 3) Additive Gaussian noise at random per-sample SNR.
    if with_noise:
        snr_db = torch.empty(B, 1, device=device).uniform_(
            snr_range_db[0], snr_range_db[1],
        )
        sig_pow = x.detach().pow(2).mean(dim=-1, keepdim=True)          # [B, 1]
        noise_pow = torch.where(
            sig_pow > 1e-8,
            sig_pow / (10 ** (snr_db / 10)),
            torch.zeros_like(sig_pow),
        )
        x = x + torch.randn_like(x) * torch.sqrt(noise_pow)

    return x


# ---------------------------------------------------------------------------
# EOT-PGD attack
# ---------------------------------------------------------------------------

def eot_pgd_attack(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.005,
    alpha: float = 0.0005,
    num_steps: int = 20,
    num_eot_samples: int = 5,
    sample_rate: int = 16000,
    device: torch.device = torch.device("cpu"),
    rir_kernels: torch.Tensor = None,
    verbose: bool = False,
    random_start: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PGD with Expectation-over-Transformation (EOT).

    All acoustic transformations (RIR convolution, volume scaling, additive
    noise) are applied **inside** the computation graph so that gradients
    propagate back through delta.  Random *parameters* (which RIR, scale
    factor, SNR) are sampled without grad — the transforms themselves are
    differentiable.

    Args:
        rir_kernels: [N, 1, kernel_len] pre-flipped RIR tensor on *device*.
        random_start: see pgd_attack.random_start.
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
    num_rirs = rir_kernels.shape[0] if rir_kernels is not None else 0

    for step in range(num_steps):
        delta = delta.detach().requires_grad_(True)
        accumulated_grad = torch.zeros_like(features)

        for _ in range(num_eot_samples):
            adv_input = features + delta

            # Batched RIR + gain + noise — see apply_eot_transforms_batched.
            # Replaces a per-sample Python loop; grad still flows from
            # `transformed` back through `adv_input` (= features + delta).
            transformed = apply_eot_transforms_batched(
                adv_input,
                rir_kernels=rir_kernels,
                with_noise=True,
            )

            logits = model(transformed)
            loss = criterion(logits, labels)
            grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
            accumulated_grad = accumulated_grad + grad.detach()

        avg_grad = accumulated_grad / num_eot_samples

        if verbose and step == 0:
            print(f"    [step 0] grad norm = {avg_grad.norm().item():.6f}")

        # PGD update
        delta = delta + alpha * avg_grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon).detach()

    return (features + delta).detach(), delta.detach()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_eot_pgd(
    model: nn.Module,
    loader,
    epsilon: float = 0.005,
    alpha: float = None,
    num_steps: int = 20,
    num_eot_samples: int = 5,
    sample_rate: int = 16000,
    device: torch.device = torch.device("cpu"),
    rir_kernels: torch.Tensor = None,
) -> dict:
    """Evaluate EOT-PGD — reports digital ASR and OTA ASR."""
    if alpha is None:
        alpha = epsilon / 10

    model.eval()
    total = 0
    correct_adv_digital = 0
    correct_adv_ota = 0
    confidence_drops: list[float] = []
    l2_norms: list[float] = []
    linf_norms: list[float] = []
    snr_values: list[float] = []
    num_rirs = rir_kernels.shape[0] if rir_kernels is not None else 0

    for batch_idx, batch in enumerate(loader):
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

        print(
            f"  batch {batch_idx}: {features_correct.shape[0]} correctly-classified samples, "
            f"attacking with eps={epsilon} ...",
            flush=True,
        )

        adv_features, _ = eot_pgd_attack(
            model,
            features_correct,
            labels_correct,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            num_eot_samples=num_eot_samples,
            sample_rate=sample_rate,
            device=device,
            rir_kernels=rir_kernels,
            verbose=(batch_idx == 0),  # print grad norm for first batch
        )

        metrics = compute_perturbation_metrics(features_correct, adv_features)
        l2_norms.append(metrics["l2_norm"])
        linf_norms.append(metrics["linf_norm"])
        snr_values.append(metrics["snr_db"])

        # ----- Digital evaluation -----
        with torch.no_grad():
            adv_logits = model(adv_features)
            adv_probs = torch.softmax(adv_logits, dim=1)
            adv_preds = adv_logits.argmax(dim=1)

        correct_adv_digital += (adv_preds == labels_correct).sum().item()

        # ----- OTA evaluation — majority vote over 5 transforms -----
        # Same batched transform stack the attack saw: RIR + gain +
        # Gaussian noise at SNR 20-40 dB. Previously this block omitted
        # the noise (made OTA ASR optimistic) and ran a per-sample loop
        # (made it slow); both fixes go through apply_eot_transforms_batched.
        ota_misclassified_votes = torch.zeros(
            len(labels_correct), dtype=torch.long, device=device,
        )
        for _ in range(5):
            with torch.no_grad():
                transformed = apply_eot_transforms_batched(
                    adv_features,
                    rir_kernels=rir_kernels,
                    with_noise=True,
                )
                ota_preds = model(transformed).argmax(dim=1)
            ota_misclassified_votes += (ota_preds != labels_correct).long()

        ota_success = ota_misclassified_votes >= 3  # majority misclassified
        correct_adv_ota += (~ota_success).sum().item()
        total += correct_mask.sum().item()

        for i in range(len(labels_correct)):
            true_class = labels_correct[i].item()
            clean_conf = clean_probs[correct_mask][i, true_class].item()
            adv_conf = adv_probs[i, true_class].item()
            confidence_drops.append(clean_conf - adv_conf)

    digital_asr = (total - correct_adv_digital) / total if total > 0 else 0.0
    ota_asr = (total - correct_adv_ota) / total if total > 0 else 0.0
    conf_drops = np.array(confidence_drops)

    return {
        "epsilon": epsilon,
        "alpha": alpha,
        "num_steps": num_steps,
        "num_eot_samples": num_eot_samples,
        "total_samples": total,
        "digital_asr": digital_asr,
        "ota_asr": ota_asr,
        "adv_accuracy_digital": correct_adv_digital / total if total > 0 else 0.0,
        "adv_accuracy_ota": correct_adv_ota / total if total > 0 else 0.0,
        "avg_confidence_drop": float(conf_drops.mean()) if len(conf_drops) > 0 else 0.0,
        "std_confidence_drop": float(conf_drops.std()) if len(conf_drops) > 0 else 0.0,
        "avg_l2_norm": float(np.mean(l2_norms)) if l2_norms else 0.0,
        "avg_linf_norm": float(np.mean(linf_norms)) if linf_norms else 0.0,
        "avg_snr_db": float(np.mean(snr_values)) if snr_values else 0.0,
    }