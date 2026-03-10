from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pann_proxy import ProxyAudioCNN


def main() -> None:
    model = ProxyAudioCNN(input_channels=1, num_classes=2, dropout=0.3)
    x = torch.randn(4, 64, 500)  # [batch, n_mels, time]
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()