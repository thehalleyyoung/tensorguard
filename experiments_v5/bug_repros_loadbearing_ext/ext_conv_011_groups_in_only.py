"""Targeted: Conv2d groups divides out_channels but not in_channels.
With Conv2d(7, 6, 3, groups=2): 7%2=1 (bad in), 6%2=0 (ok out).
Exercises the in-channel divisibility check (line ~4935) in isolation."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(7, 6, 3, groups=2)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    try:
        M()(torch.randn(1, 7, 16, 16))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
