"""Depthwise-separable CNN for keyword spotting (DS-CNN).

Architecture follows the DS-CNN family from "Hello Edge: Keyword Spotting on
Microcontrollers" (Zhang et al., 2017, arXiv:1711.07128): one regular conv
stem, then a stack of depthwise-separable conv blocks, global average
pooling, and a single linear classifier.

A depthwise-separable conv factorizes a standard KxK conv into
  (1) a depthwise KxK conv: one filter per channel, no cross-channel mixing
  (2) a pointwise 1x1 conv: mixes channels, no spatial extent
which costs roughly C*(K*K + C) parameters instead of C*C*K*K — about 8-9x
fewer for K=3. That parameter efficiency is the whole point of the
"edge/TinyML" framing.
"""

import torch
import torch.nn as nn


class DSBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride, padding=1,
            groups=channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    """Input: log-mel spectrogram (B, 1, 64, 101). Output: logits (B, 12)."""

    def __init__(self, n_classes: int = 12, width: int = 160, n_blocks: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.config = {"n_classes": n_classes, "width": width,
                       "n_blocks": n_blocks, "dropout": dropout}
        # Stem: a tall-in-frequency kernel (10x4), stride 2 in both axes,
        # as in the DS-CNN paper — quickly reduces the 64x101 input.
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=(10, 4), stride=(2, 2),
                      padding=(5, 2), bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        blocks = [DSBlock(width, stride=2)]  # one more 2x downsample
        blocks += [DSBlock(width) for _ in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(width, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = DSCNN()
    x = torch.randn(2, 1, 64, 101)
    y = m(x)
    n = count_parameters(m)
    print(f"output shape: {tuple(y.shape)}")
    print(f"parameters:   {n:,} ({n * 4 / 1e6:.2f} MB fp32)")
