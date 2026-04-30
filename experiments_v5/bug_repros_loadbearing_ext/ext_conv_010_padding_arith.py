"""Targeted: Conv2d with padding such that arithmetic-swap mutation flips outcome.
With kernel=5, padding=2, input H=3: h_out = (3 + 4 - 4 - 1)/1 + 1 = 3 (positive).
A `+` -> `-` swap on `2*padding[0]` makes h_out = (3 - 4 - 4 - 1)+1 = -5 (negative).
This bug is NOT itself a violation (correct verdict V), so mutation
flipping V -> RP also kills."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=5, padding=2)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    M()(torch.randn(1, 3, 3, 3))
