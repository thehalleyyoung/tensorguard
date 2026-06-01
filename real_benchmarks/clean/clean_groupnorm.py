"""
TensorGuard benchmark corpus -- CLEAN model (clean_groupnorm).

Conv followed by GroupNorm whose channel count divides evenly.
"""

INPUT_SHAPES = {'x': (2, 16, 8, 8)}


import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.gn = nn.GroupNorm(4, 16)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)))
