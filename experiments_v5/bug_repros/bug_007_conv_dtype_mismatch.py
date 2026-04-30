"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/180548
Expected Error: should be the same
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SHAPES = {"x": (1, 1, 4, 4), "weight": (1, 1, 1, 1)}


class BuggyModule(nn.Module):
    def forward(self, x, weight):
        return F.conv2d(x, weight)


if __name__ == '__main__':
    try:
        m = BuggyModule()
        x = torch.randn(1, 1, 4, 4, dtype=torch.bfloat16)
        weight = torch.randn(1, 1, 1, 1, dtype=torch.float32)
        m(x, weight)
    except RuntimeError as e:
        print(f"Error: {e}")
