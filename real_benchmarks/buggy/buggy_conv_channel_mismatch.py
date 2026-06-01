"""
TensorGuard benchmark corpus -- BUGGY model (buggy_conv_channel_mismatch).

GitHub Issue: https://github.com/pytorch/pytorch/issues/179931
Expected Error: channels

conv2 expects 8 input channels but conv1 emits 16.
"""

INPUT_SHAPES = {'x': (8, 3, 32, 32)}


import torch
import torch.nn as nn
import torch.nn.functional as F


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 32, 3, padding=1)  # BUG: in_channels should be 16

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))
