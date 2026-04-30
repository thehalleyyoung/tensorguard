# input_shape: (4, 16, 8)
# bug: matmul of (16,8) @ (16,8) — second tensor not transposed.
# cause_line: 14
# expected: 8, 16
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(16, 8))

    def forward(self, x):
        return x @ self.w  # WRONG: needs self.w.t()
