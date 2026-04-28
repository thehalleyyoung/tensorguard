"""
L5-lowconf stress case 04: division by zero from window_size=0 in local attention.

Target feature: low-confidence violations (L5).
Bug: window_size=0 default → ZeroDivisionError when computing number of
windows: seq_len // window_size. Flow-sensitive analyser detects this.

Expected:
  WITHOUT L5: Verified
  WITH    L5: Refuted (division_by_zero on window_size)
"""
import torch
import torch.nn as nn


class LocalWindowAttention(nn.Module):
    def __init__(self, d_model: int = 64, window_size: int = 0):
        super().__init__()
        # BUG: window_size=0 → ZeroDivisionError
        self.num_windows = 512 // window_size
        self.window_size = window_size
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


FEATURE = "L5_lowconf"
INPUT_SHAPES = {"x": ("batch", 512, 64)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
