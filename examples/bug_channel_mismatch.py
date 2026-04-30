"""Bug example: a Conv2d ``in_channels`` declared off-by-one
relative to the producer.  The producer has 64 output channels,
so consumer must accept 64; this consumer mistakenly says 32.

Running the model triggers a runtime ``RuntimeError: Given groups=1,
weight of size [128, 32, 3, 3], expected input[1, 64, 16, 16] to have
32 channels, but got 64 channels instead`` -- a *symptom* deep in
``F.conv2d``.  TensorGuard's static check localises the *cause* to
the constructor at line ``self.conv2 = nn.Conv2d(32, 128, ...)``.
"""
import torch.nn as nn


class ChannelMismatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)   # outputs 64 channels
        self.bn1 = nn.BatchNorm2d(64)
        # BUG: in_channels should be 64 (matching conv1.out_channels), not 32
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):                             # x: (B, 3, H, W)
        x = self.bn1(self.conv1(x))                   # (B, 64, H, W)
        x = self.bn2(self.conv2(x))                   # CRASH: expected 32
        return x
