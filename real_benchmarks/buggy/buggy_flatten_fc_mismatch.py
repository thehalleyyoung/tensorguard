"""
TensorGuard benchmark corpus -- BUGGY model (buggy_flatten_fc_mismatch).

GitHub Issue: https://github.com/pytorch/pytorch/issues/172739
Expected Error: mat1 and mat2 shapes cannot be multiplied

Flattened conv features are 16*8*8=1024 but the head expects 999.
"""

INPUT_SHAPES = {'x': (8, 3, 8, 8)}


import torch
import torch.nn as nn
import torch.nn.functional as F


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(999, 10)  # BUG: flattened size is 16*8*8 = 1024

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc(x)
