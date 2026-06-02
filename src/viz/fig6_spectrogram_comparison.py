"""Figure 6 — clean vs PGD-adversarial mel-spectrograms, plus the residual.

Reads the example clip pair saved by scripts/run_pgd_with_samples.py.
"""
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

from src.viz.style import apply_style, save_fig


def compute_mel(waveform, sr=16000, n_mels=64, n_fft=400, hop_length=160):
    """Log-mel matching src.features.spectrograms.LogMelSpectrogram."""
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    w = torch.as_tensor(waveform, dtype=torch.float32)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft,
        win_length=int(0.025 * sr), hop_length=hop_length,
        n_mels=n_mels,
    )(w)
    log_mel = torch.log(mel + 1e-8)
    return log_mel.squeeze(0).numpy()


def plot_spectrogram_comparison(
    npz_path="outputs/results/pgd_samples_eps0.001.npz",
    outname="fig6_spectrogram_comparison",
    sr=16000,
):
    apply_style()
    data  = np.load(npz_path, allow_pickle=True)
    clean = np.asarray(data["example_clean"]).squeeze()
    adv   = np.asarray(data["example_adv"  ]).squeeze()
    eps   = float(data["epsilon"])

    if clean.size == 0 or adv.size == 0:
        raise ValueError(
            "No example waveforms were saved in the .npz. "
            "Re-run scripts/run_pgd_with_samples.py — no drone sample flipped."
        )

    # Trim off the zero-padding that DroneAudioDataset adds to short clips;
    # otherwise the silent region dominates the colour scale.
    abs_clean = np.abs(clean)
    threshold = abs_clean.max() * 0.01
    nonzero = np.where(abs_clean > threshold)[0]
    if len(nonzero) > 0:
        pad = int(0.01 * sr)  # leave 10 ms either side
        start = max(0, nonzero[0] - pad)
        end   = min(len(clean), nonzero[-1] + pad)
        clean = clean[start:end]
        adv   = adv  [start:end]
    duration_s = len(clean) / sr

    perturb = adv - clean

    mel_clean   = compute_mel(clean,   sr=sr)
    mel_adv     = compute_mel(adv,     sr=sr)
    mel_perturb = compute_mel(perturb, sr=sr)

    # Same vmin/vmax across (a) and (b) so the comparison is honest.
    combined = np.concatenate([mel_clean.ravel(), mel_adv.ravel()])
    vmin = float(np.percentile(combined,  2))
    vmax = float(np.percentile(combined, 99))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    extent = [0, duration_s, 0, mel_clean.shape[0]]

    im0 = axes[0].imshow(mel_clean, aspect="auto", origin="lower",
                          cmap="viridis", extent=extent,
                          vmin=vmin, vmax=vmax)
    axes[0].set_title("(a) Clean drone audio")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Mel bin")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="log-mel")

    im1 = axes[1].imshow(mel_adv, aspect="auto", origin="lower",
                          cmap="viridis", extent=extent,
                          vmin=vmin, vmax=vmax)
    axes[1].set_title(f"(b) PGD adversarial (ε = {eps})")
    axes[1].set_xlabel("Time (s)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="log-mel")

    im2 = axes[2].imshow(mel_perturb, aspect="auto", origin="lower",
                          cmap="RdBu_r", extent=extent)
    axes[2].set_title("(c) Perturbation (adv − clean)")
    axes[2].set_xlabel("Time (s)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="log-mel (diff)")

    for ax in axes:
        ax.grid(False)

    # Recompute the L2/L-inf/SNR on the cropped region, not the padded clip.
    l2 = float(np.linalg.norm(perturb))
    linf = float(np.max(np.abs(perturb)))
    sig_power = float(np.mean(clean ** 2) + 1e-12)
    noise_power = float(np.mean(perturb ** 2) + 1e-12)
    snr_db = 10 * np.log10(sig_power / noise_power)

    fig.suptitle(
        f"Figure 6 — PGD perturbation is inaudible in time and frequency "
        f"(L₂ = {l2:.3f},  L∞ = {linf:.4f},  SNR = {snr_db:.1f} dB)",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    paths = save_fig(fig, outname)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in plot_spectrogram_comparison():
        print(f"saved {p}")