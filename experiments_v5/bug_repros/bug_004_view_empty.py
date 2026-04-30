"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/174379
Expected Error: is invalid for input of size 0
"""

import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (0,)}


class BuggyModule(nn.Module):
    def forward(self, x):
        return x.view(328, 1000)


if __name__ == '__main__':
    try:
        m = BuggyModule()
        x = torch.empty(0)
        m(x)
    except RuntimeError as e:
        print(f"Error: {e}")
