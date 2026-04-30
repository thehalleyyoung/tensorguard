"""Targeted: Conv2d groups does not divide out_channels.
Exercises line ~4940 (`layer.out_channels % groups != 0`)."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 10, 3, groups=3)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    try:
        M()(torch.randn(1, 6, 16, 16))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
