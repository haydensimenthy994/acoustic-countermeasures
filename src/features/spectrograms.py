from __future__ import annotations

import torch
import torchaudio


class LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        frame_ms: int = 25,
        hop_ms: int = 10,
        f_min: float = 0.0,
        f_max: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        n_fft = int(sample_rate * frame_ms / 1000)
        hop_length = int(sample_rate * hop_ms / 1000)
        win_length = n_fft

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            center=True,
        )
        self.eps = eps

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [1, num_samples] or [batch, num_samples]
        returns:  [n_mels, time] or [batch, n_mels, time]
        """
        mel = self.mel(waveform)
        log_mel = torch.log(mel + self.eps)
        return log_mel