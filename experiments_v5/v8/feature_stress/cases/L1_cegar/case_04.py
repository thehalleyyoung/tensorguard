"""
L1-CEGAR stress case 04: embed_dim % n_slots != 0.

Target feature: CEGAR (L1).
Bug: embed_dim=128, n_slots=9 → 128 % 9 = 2 != 0.

Honest: CEGAR returns SAFE — non-discriminating in practice.
"""
import torch
import torch.nn as nn


class SlotAttention(nn.Module):
    def __init__(self, embed_dim: int = 128, n_slots: int = 9):
        super().__init__()
        # BUG: 128 % 9 = 2 != 0
        self.embed_dim = embed_dim
        self.n_slots = n_slots
        self.slot_dim = embed_dim // n_slots   # 14, not exact
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        k = self.k_proj(x)
        v = self.v_proj(x)
        # contract: D == n_slots * slot_dim — violated
        k = k.reshape(B, N, self.n_slots, self.slot_dim)
        v = v.reshape(B, N, self.n_slots, self.slot_dim)
        attn = (k * v).sum(-1)
        return attn.mean(dim=2)


FEATURE = "L1_cegar"
INPUT_SHAPES = {"x": ("batch", "N", 128)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
