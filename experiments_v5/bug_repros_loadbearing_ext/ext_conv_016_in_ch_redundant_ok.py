"""Targeted: Conv2d clean baseline where mutating `==` on isinstance check
or the in_channels presence check would otherwise leave the analyser
believing channel-mismatch had occurred. Pure-clean module."""
import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    M()(torch.randn(2, 8, 16, 16))
