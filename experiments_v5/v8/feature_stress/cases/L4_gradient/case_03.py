"""
L4-gradient stress case 03: detach before projection severs encoder gradient.

Target feature: gradient-flow check (L4, B1 type).
Bug: The attention output is detached before the output projection.
self.out_proj.weight and self.out_proj.bias will not receive gradients.

Expected:
  WITHOUT L4: Verified
  WITH    L4: Refuted (GRADIENT-BROKEN on attention output)
"""
import torch
import torch.nn as nn


class DetachedAttentionOut(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        # BUG: detach severs gradient before projection
        attn_out = attn_out.detach()
        return self.out_proj(attn_out)


FEATURE = "L4_gradient"
INPUT_SHAPES = {"x": ("batch", "seq", 128)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
