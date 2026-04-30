"""Targeted: Conv2d in_channels mismatch (declared 4, supplied 8)."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 12, kernel_size=3)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    try:
        M()(torch.randn(1, 8, 16, 16))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
