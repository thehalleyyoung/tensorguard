"""Targeted: Conv2d effective receptive field exceeds input via dilation.
Exercises the dilation arithmetic on line ~4995."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, dilation=4)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    try:
        M()(torch.randn(1, 3, 6, 6))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
