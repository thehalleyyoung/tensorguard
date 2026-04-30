"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/176375
Expected Error: is invalid for input of size
"""

import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (2048, 384)}


class BuggyModule(nn.Module):
    def forward(self, x):
        return x.view(2048, 640)


if __name__ == '__main__':
    try:
        m = BuggyModule()
        x = torch.randn(2048, 384)
        m(x)
    except RuntimeError as e:
        print(f"Error: {e}")
