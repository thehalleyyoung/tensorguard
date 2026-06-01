"""
TensorGuard benchmark corpus -- BUGGY model (buggy_view_total_size).

GitHub Issue: https://github.com/pytorch/pytorch/issues/177691
Expected Error: is invalid for input of size

view target has a different total element count than the input.
"""

INPUT_SHAPES = {'x': (2048, 384)}


import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def forward(self, x):
        return x.view(2048, 640)  # BUG: 2048*640 != 2048*384
