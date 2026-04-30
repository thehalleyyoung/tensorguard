"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/179573
Expected Error: must match the size of tensor
"""

import torch
import torch.nn as nn


INPUT_SHAPES = {"x": (16, 49, 256)}


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(128, 2))

    def forward(self, x):
        return x * self.scale


if __name__ == '__main__':
    try:
        m = BuggyModule()
        x = torch.randn(16, 49, 256)
        m(x)
    except RuntimeError as e:
        print(f"Error: {e}")
