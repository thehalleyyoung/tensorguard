"""
L1-CEGAR stress case 05: total_heads % groups != 0.

Target feature: CEGAR (L1).
Bug: total_heads=10, groups=3 → 10 % 3 = 1 != 0.

Honest: CEGAR returns SAFE — non-discriminating in practice.
"""
import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int = 90, total_heads: int = 10, groups: int = 3):
        super().__init__()
        # BUG: 10 % 3 = 1 != 0
        self.d_model = d_model
        self.total_heads = total_heads
        self.groups = groups
        self.heads_per_group = total_heads // groups  # 3, not exact
        self.head_dim = d_model // total_heads        # 9
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model // groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x)
        # contract: total_heads % groups == 0 — violated
        q = q.reshape(B, S, self.groups, self.heads_per_group, self.head_dim)
        kv = self.kv_proj(x)
        return q.reshape(B, S, D)


FEATURE = "L1_cegar"
INPUT_SHAPES = {"x": ("batch", "seq", 90)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
