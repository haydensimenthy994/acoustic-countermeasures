from __future__ import annotations

from pathlib import Path

import soundfile as sf
import pandas as pd
import torch
import torchaudio
import yaml
from torch.utils.data import Dataset

from src.features.spectrograms import LogMelSpectrogram


LABEL_MAP = {
    "drone": 1,
    "no_drone": 0,
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DroneAudioDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str,
        config_path: str = "configs/data.yaml",
        split: str | None = None,
        fixed_duration_sec: float | None = 5.0,
        use_raw_waveform: bool = False,
    ) -> None:
        self.cfg = load_config(config_path)
        self.sample_rate = int(self.cfg["audio"]["sample_rate"])
        self.n_mels = int(self.cfg["features"]["n_mels"])
        self.frame_ms = int(self.cfg["features"]["frame_ms"])
        self.hop_ms = int(self.cfg["features"]["hop_ms"])
        self.use_raw_waveform = use_raw_waveform

        self.df = pd.read_csv(metadata_csv)

        if split is not None:
            self.df = self.df[self.df["split"] == split].copy()

        self.df = self.df[self.df["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)

        self.fixed_duration_sec = fixed_duration_sec
        self.target_num_samples = (
            int(self.sample_rate * fixed_duration_sec) if fixed_duration_sec is not None else None
        )

        if not use_raw_waveform:
            self.feature_extractor = LogMelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                frame_ms=self.frame_ms,
                hop_ms=self.hop_ms,
            )

    def __len__(self) -> int:
        return len(self.df)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.target_num_samples is None:
            return waveform
        num_samples = waveform.shape[-1]
        if num_samples > self.target_num_samples:
            waveform = waveform[..., :self.target_num_samples]
        elif num_samples < self.target_num_samples:
            pad_amount = self.target_num_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        audio_path = Path(row["filepath"])

        data, sr = sf.read(str(audio_path), always_2d=True)
        waveform = torch.tensor(data.T, dtype=torch.float32)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        waveform = self._fix_length(waveform)
        label = LABEL_MAP[row["label"]]

        if self.use_raw_waveform:
            # CNN14 wants raw waveform; ProxyAudioCNN wants mel.
            features = waveform.squeeze(0)
        else:
            features = self.feature_extractor(waveform).squeeze(0)

        return {
            "features": features,
            "label": torch.tensor(label, dtype=torch.long),
            "filepath": str(audio_path),
            "dataset": row.get("dataset", "unknown"),
        }