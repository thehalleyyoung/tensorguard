# bug: key tensor is detached before the attention dot-product;
#      gradients cannot flow back through the key projection, so the
#      key projection weights receive no gradient signal.
# source: self-attention implementation where the author detaches keys
#         to implement a "stop-gradient on keys" trick but applies it
#         to the wrong variable.
import torch
import torch.nn as nn


class BrokenSelfAttention(nn.Module):
    """Self-attention with gradient accidentally severed on keys."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out    = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x).detach()   # BUG: severs grad to k_proj weights
        v = self.v_proj(x)
        attn = (q @ k.transpose(-2, -1))
        return self.out(attn @ v)
