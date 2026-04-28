"""
L5-lowconf stress case 02: division by zero when stride defaults to 0.

Target feature: low-confidence violations (L5).
Bug: stride=0 default in a convolutional stride operation causes
ZeroDivisionError when computing output size.

Expected:
  WITHOUT L5: Verified
  WITH    L5: Refuted (division_by_zero from flow-sensitive analysis)
"""
import torch
import torch.nn as nn


class ZeroStrideConv(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 16, stride: int = 0):
        super().__init__()
        # BUG: stride=0 → ZeroDivisionError in output size calculation
        self.out_h = 224 // stride
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=max(stride, 1), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


FEATURE = "L5_lowconf"
INPUT_SHAPES = {"x": ("batch", 3, 224, 224)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
