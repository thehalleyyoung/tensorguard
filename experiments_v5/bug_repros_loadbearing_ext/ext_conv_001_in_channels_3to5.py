"""Targeted: Conv2d in_channels mismatch (declared 3, supplied 5).
Designed to exercise src/model_checker.py:_propagate_conv2d line ~4925
(`c_in.value != layer.in_channels`)."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    try:
        M()(torch.randn(2, 5, 32, 32))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
