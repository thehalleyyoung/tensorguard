"""Targeted: Conv2d where padding turns a would-be-negative output positive.
With kernel=3, padding=1, input H=2: h_out = (2 + 2 - 2 - 1)/1 + 1 = 2 (positive).
Mutating `2*padding` to `2-padding` (via `+` -> `-`) yields:
  h_out = (2 - 2 - 2 - 1)/1 + 1 = -2 (non-positive)."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    M()(torch.randn(1, 3, 2, 2))
