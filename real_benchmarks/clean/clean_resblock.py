"""
TensorGuard benchmark corpus -- CLEAN model (clean_resblock).

Residual block; the skip connection shape matches the main path.
"""

INPUT_SHAPES = {'x': (4, 64, 16, 16)}


import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(h + x)
