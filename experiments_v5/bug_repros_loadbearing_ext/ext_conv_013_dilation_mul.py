"""Targeted: clean Conv2d where mutating the `*` in `dilation*(ks-1)` to `+`
flips the spatial-dim arithmetic enough to make the `<= 0` guard fire.
With dilation=3, ks=2, input H=4, no padding:
  correct: h_out = (4 + 0 - 3*(2-1) - 1)/1 + 1 = 1 (positive, V).
  mutated `*` -> `+`: h_out = (4 + 0 - (3+(2-1)) - 1)/1 + 1 = 0 (non-positive, RP).
This bug is therefore CLEAN; mutations that flip V to RP kill."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=2, dilation=3)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    M()(torch.randn(1, 3, 4, 4))
