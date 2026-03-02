"""Deliberate shape error that TensorGuard catches.

The Linear layer expects 256 input features, but the preceding
Conv2d + AdaptiveAvgPool2d produces 128 features. TensorGuard
reports a shape mismatch at the fc layer.
"""
import torch.nn as nn


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # BUG: expects 256 features but conv outputs 128 channels
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv(x)          # [B, 128, H, W]
        x = self.pool(x)          # [B, 128, 1, 1]
        x = x.flatten(1)          # [B, 128]
        x = self.fc(x)            # ERROR: fc expects [B, 256]
        return x
