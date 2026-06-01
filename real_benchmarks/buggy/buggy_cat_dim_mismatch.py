"""
TensorGuard benchmark corpus -- BUGGY model (buggy_cat_dim_mismatch).

GitHub Issue: https://github.com/pytorch/pytorch/issues/175683
Expected Error: size of tensor

Concatenation along dim 0 then added to one branch whose dim-0 differs.
"""

INPUT_SHAPES = {'x': (3, 8)}


import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 4)
        self.b = nn.Linear(8, 6)

    def forward(self, x):
        ha = self.a(x)
        hb = self.b(x)
        # cat along dim 0 -> (6, 4) but ha is (3, 4): add broadcasts incorrectly
        return torch.cat([ha, hb], dim=0) + ha
