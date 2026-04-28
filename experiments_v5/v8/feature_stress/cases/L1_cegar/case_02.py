"""
L1-CEGAR stress case 02: d_model % n_heads != 0.

Target feature: CEGAR (L1).
Bug: d_model=100 is not divisible by n_heads=3.
Same class of CEGAR contract violation as case_01.

Honest: CEGAR returns SAFE (real_bugs=[]) — non-discriminating in practice.
"""
import torch
import torch.nn as nn


class MultiHeadDotProduct(nn.Module):
    def __init__(self, d_model: int = 100, n_heads: int = 3):
        super().__init__()
        # BUG: 100 % 3 != 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads   # 33, not exact
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.proj(x)
        # contract: D == n_heads * head_dim — violated here
        h = h.view(B, S, self.n_heads, self.head_dim)
        return h.view(B, S, -1)


FEATURE = "L1_cegar"
INPUT_SHAPES = {"x": ("batch", "seq", 100)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
