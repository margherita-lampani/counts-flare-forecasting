"""
Reusable neural-network building blocks for flare forecasting models.

This module contains shared architectural components used across the
forecasting networks implemented in the project. These blocks provide
modular and configurable feature-extraction layers that can be combined
to construct both regression and classification models.

Implemented components:
    - ResidualBlock: convolutional residual block with skip connections,
      batch normalization, and ReLU activations.

The residual architecture improves gradient propagation and facilitates
the training of deeper convolutional networks on solar magnetogram data.
"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN + skip → ReLU."""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))
