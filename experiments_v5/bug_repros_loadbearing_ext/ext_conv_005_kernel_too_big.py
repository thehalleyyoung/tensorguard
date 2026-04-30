"""Targeted: Conv2d output non-positive (kernel > input).
Exercises lines ~4995 (spatial-dim arithmetic) and ~4997 (`<= 0` guard)."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=5)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    try:
        M()(torch.randn(1, 3, 3, 3))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
