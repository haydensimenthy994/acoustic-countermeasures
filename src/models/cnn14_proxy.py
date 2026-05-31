from __future__ import annotations

import sys
import os
import importlib.util
import torch
import torch.nn as nn

# PANNs repo path
PANNS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../audioset_tagging_cnn/pytorch')
)

# Add to sys.path so pytorch_utils and other PANNs deps resolve correctly
if PANNS_DIR not in sys.path:
    sys.path.insert(0, PANNS_DIR)

# Load PANNs models.py directly by file path to avoid naming conflict with src/models/
spec = importlib.util.spec_from_file_location("panns_models", os.path.join(PANNS_DIR, "models.py"))
panns_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panns_models)
Cnn14 = panns_models.Cnn14


class CNN14ProxyClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained_path: str | None = None,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()

        self.base = Cnn14(
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        )

        if pretrained_path is not None:
            checkpoint = torch.load(
                pretrained_path,
                map_location='cpu',
                weights_only=False
            )
            self.base.load_state_dict(checkpoint['model'], strict=False)
            print(f"Loaded pretrained CNN14 weights from {pretrained_path}")

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False
            print("Base CNN14 frozen — only classifier head will train")

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, samples] — raw waveform at 16kHz
        returns: [batch, num_classes]
        """
        output_dict = self.base(x)
        embedding = output_dict['embedding']  # [batch, 2048]
        logits = self.classifier(embedding)
        return logits