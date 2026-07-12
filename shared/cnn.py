"""
2-D CNN for log-mel-spectrogram input.

Input shape:  (B, 1, n_mels=128, n_frames=100)
Output:       (B, 15) raw logits (no sigmoid -- pair with BCEWithLogitsLoss)

Architecture: four conv blocks doubling channels (32 -> 64 -> 128 -> 256)
              with BatchNorm + ReLU + MaxPool, followed by global average
              pooling and a single linear classifier.

Why this size: ~700K params is the smallest network that decisively beats
the MLP on aggregated features. Larger CNNs overfit on 100K-segment training
sets at our class imbalance. Smaller CNNs underfit on rare classes.

Hardware: trains in ~30 s/epoch on Apple M-series MPS (with grad clipping),
~2-3 min/epoch on CPU. GPU strongly preferred.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MelCNN(nn.Module):
    """4-block CNN for 128 x 100 log-mel patches.

    Parameters
    ----------
    n_classes : int
        Number of output logits (15 for this project).
    dropout : float
        Dropout before the final linear layer.
    base_ch : int
        Channel count in the first conv block; doubled in each subsequent block.
    """

    def __init__(self, n_classes: int = 15, dropout: float = 0.4,
                 base_ch: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.features = nn.Sequential(
            self._block(1,  c1),     # (B, 32,  64, 50)
            self._block(c1, c2),     # (B, 64,  32, 25)
            self._block(c2, c3),     # (B, 128, 16, 12)
            self._block(c3, c4),     # (B, 256, 8,  6)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, 256, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, n_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.gap(z)
        return self.head(z)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
