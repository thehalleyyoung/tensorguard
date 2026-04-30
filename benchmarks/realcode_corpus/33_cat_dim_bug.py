# input_shape: (2, 3, 64, 64)
# bug: cat of two tensors with mismatched channel counts on dim=1.
# cause_line: 17
# expected: 32, 64
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, x):
        a = self.c1(x)
        b = self.c2(x)
        return torch.cat([a, b], dim=2)  # WRONG: cat-dim mismatched
