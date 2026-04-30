"""Targeted: Conv2d spatial dim that is exactly zero under correct arithmetic
but becomes positive (or non-positive) under a `+`/`-` swap on the
spatial-dim formula.
With kernel=4, stride=1, pad=0, dil=1: h_out = (4 + 0 - 3 - 1) / 1 + 1 = 1 (ok).
But with input H=3: h_out = (3 + 0 - 3 - 1)/1 + 1 = 0 → non-positive."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=4)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    try:
        M()(torch.randn(1, 3, 3, 3))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
