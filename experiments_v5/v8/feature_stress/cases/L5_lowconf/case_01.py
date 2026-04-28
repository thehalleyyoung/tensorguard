"""
L5-lowconf stress case 01: division by zero when num_heads defaults to 0.

Target feature: low-confidence violations (L5).
Bug: num_heads=0 default makes head_dim = d_model // num_heads raise
ZeroDivisionError at object construction time. The flow-sensitive analyser
(analyze()) detects this as a potential division-by-zero with confidence 0.80,
but the constraint-based verify_model does not flag it.

Expected:
  WITHOUT L5 (high_confidence_only=True):  Verified  (flow-sensitive not run)
  WITH    L5 (high_confidence_only=False):  Refuted   (division_by_zero found)
"""
import torch
import torch.nn as nn


class ZeroHeadAttention(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 0):
        super().__init__()
        # BUG: num_heads=0 → ZeroDivisionError
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) / self.head_dim


FEATURE = "L5_lowconf"
INPUT_SHAPES = {"x": ("batch", "seq", 64)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
