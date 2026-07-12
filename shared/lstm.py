"""
Convolutional Recurrent Neural Network (CRNN) for log-mel spectrograms.

Why CRNN, not pure LSTM
-----------------------
A pure LSTM over 100 mel-frames of 128-D vectors works, but it spends most
of its capacity learning per-frame frequency patterns that a 2-D conv can
extract trivially. The standard architecture for spectrogram-input SED
(DCASE-style) is therefore CRNN: small CNN front-end -> RNN over time ->
classifier. We follow that convention.

Input shape:  (B, 1, n_mels=128, n_frames=100)
Output:       (B, 15) raw logits

Pipeline:
  (B, 1,   128, 100)  --conv block 1 (pool 4x4)--> (B, 64,  32, 25)
  (B, 64,  32, 25)    --conv block 2 (pool 4x1)--> (B, 128, 8,  25)
  reshape                                          (B, 25, 1024)
  BiLSTM (2 layers, hidden=128)                    (B, 25, 256)
  mean-pool over time                              (B, 256)
  Dropout + Linear                                 (B, 15)

~1.4M parameters. Trains ~6-8 min/epoch on Apple M-series MPS at batch 128.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Conv front-end + bidirectional LSTM + linear head."""

    def __init__(
        self,
        n_classes:  int = 15,
        cnn_ch:     tuple[int, int] = (64, 128),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout:    float = 0.4,
    ):
        super().__init__()
        c1, c2 = cnn_ch

        # Conv block 1: reduce freq by 4, time by 4
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 4)),       # (B, c1, 32, 25)
        )

        # Conv block 2: reduce freq by 4 more, keep time
        self.cnn2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)),       # (B, c2, 8, 25)
        )

        # After conv: each time step has c2 * 8 = 1024 features
        seq_dim = c2 * 8
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 128, 100)
        z = self.cnn1(x)               # (B, c1, 32, 25)
        z = self.cnn2(z)               # (B, c2,  8, 25)
        B, C, F, T = z.shape
        z = z.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # (B, T, C*F)
        z, _ = self.lstm(z)            # (B, T, 2 * lstm_hidden)
        z = z.mean(dim=1)              # mean-pool over time
        return self.head(z)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
