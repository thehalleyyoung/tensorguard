"""
L5-lowconf stress case 03: division by zero when scale_factor defaults to 0.

Target feature: low-confidence violations (L5).
Bug: scale_factor=0 → ZeroDivisionError when normalising attention scores.
The flow-sensitive analyser flags scale_factor as an unguarded zero divisor.

Expected:
  WITHOUT L5: Verified
  WITH    L5: Refuted (division_by_zero)
"""
import torch
import torch.nn as nn


class ScaledDotProductAttn(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, scale_factor: int = 0):
        super().__init__()
        # BUG: scale_factor=0 → ZeroDivisionError
        self.scale = d_model // scale_factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        return (q @ k.transpose(-2, -1)) / self.scale


FEATURE = "L5_lowconf"
INPUT_SHAPES = {"x": ("batch", "seq", 64)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
