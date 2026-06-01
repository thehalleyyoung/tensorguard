"""
TensorGuard benchmark corpus -- BUGGY model (buggy_linear_inout_mismatch).

GitHub Issue: https://github.com/pytorch/pytorch/issues/179789
Expected Error: mat1 and mat2 shapes cannot be multiplied

fc2 expects 128 in-features but receives 256 from fc1.
"""

INPUT_SHAPES = {'x': (32, 784)}


import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)  # BUG: should be Linear(256, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
