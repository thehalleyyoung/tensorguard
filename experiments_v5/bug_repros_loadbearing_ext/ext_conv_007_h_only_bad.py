"""Targeted: Conv2d output non-positive only on H (asymmetric).
Designed so that mutating the second `<= 0` (w_out check) to `>= 0`
or the `or` to `and` will silence detection."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=(7, 1))

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    try:
        M()(torch.randn(1, 3, 3, 32))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
