"""
TensorGuard benchmark corpus -- BUGGY model (buggy_matmul_inner_mismatch).

GitHub Issue: https://github.com/pytorch/pytorch/issues/176230
Expected Error: mat1 and mat2 shapes cannot be multiplied

Inner dimensions of the two matmul operands do not agree (8 vs 16).
"""

INPUT_SHAPES = {'x': (4, 8), 'y': (16, 4)}


import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)  # BUG: (4,8) @ (16,4) inner dims 8 != 16
