"""Cross-dataset Figure 6 — mel-spectrogram comparison on one SWARM clip."""
from __future__ import annotations

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

from src.viz.style import apply_style, save_fig


def compute_mel(waveform, sr=16000, n_mels=64, n_fft=400, hop_length=160):
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    w = torch.as_tensor(waveform, dtype=torch.float32)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft,
        win_length=int(0.025 * sr), hop_length=hop_length,
        n_mels=n_mels,
    )(w)
    return torch.log(mel + 1e-8).squeeze(0).numpy()


def plot_cross_spectrogram_comparison(
    npz_path: str = "outputs/results/cross_pgd_samples_eps0.001.npz",
    outname: str = "fig6_cross_spectrogram_comparison",
    outdir: str = "outputs/figures_cross",
    sr: int = 16000,
):
    apply_style()
    data = np.load(npz_path, allow_pickle=True)
    clean = np.asarray(data["example_clean"]).squeeze()
    adv = np.asarray(data["example_adv"]).squeeze()
    eps = float(data["epsilon"])
    fpath = str(data.get("example_filepath", ""))

    if clean.size == 0 or adv.size == 0:
        raise ValueError(
            "No example waveforms in cross npz — run run_cross_pgd_with_samples.py"
        )

    abs_clean = np.abs(clean)
    threshold = abs_clean.max() * 0.01
    nonzero = np.where(abs_clean > threshold)[0]
    if len(nonzero) > 0:
        pad = int(0.01 * sr)
        start = max(0, nonzero[0] - pad)
        end = min(len(clean), nonzero[-1] + pad)
        clean = clean[start:end]
        adv = adv[start:end]

    perturb = adv - clean
    mel_clean = compute_mel(clean, sr=sr)
    mel_adv = compute_mel(adv, sr=sr)
    mel_perturb = compute_mel(perturb, sr=sr)

    combined = np.concatenate([mel_clean.ravel(), mel_adv.ravel()])
    vmin = float(np.percentile(combined, 2))
    vmax = float(np.percentile(combined, 99))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    titles = [
        "(a) Clean SWARM clip",
        f"(b) PGD adversarial (ε = {eps})",
        "(c) Perturbation (adv − clean)",
    ]
    mels = [mel_clean, mel_adv, mel_perturb]
    vmins = [vmin, vmin, None]
    vmaxs = [vmax, vmax, None]

    for ax, mel, title, lo, hi in zip(axes, mels, titles, vmins, vmaxs):
        im = ax.imshow(
            mel, aspect="auto", origin="lower",
            vmin=lo, vmax=hi, cmap="magma",
        )
        ax.set_title(title)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Mel bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    subtitle = fpath if fpath else "SWARM test clip"
    fig.suptitle(
        f"Figure 6 (cross-dataset) — Spectrogram comparison\n{subtitle}",
        fontsize=12, y=1.05,
    )
    fig.tight_layout()
    paths = save_fig(fig, outname, outdir=outdir)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in plot_cross_spectrogram_comparison():
        print(f"saved {p}")
