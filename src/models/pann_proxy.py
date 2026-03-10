from __future__ import annotations

import torch
import torch.nn as nn


class ProxyAudioCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n_mels, time]
        returns: [batch, num_classes]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [batch, n_mels, time], got {x.shape}")

        x = x.unsqueeze(1)  # [batch, 1, n_mels, time]
        x = self.features(x)
        x = self.classifier(x)
        return x