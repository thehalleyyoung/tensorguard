"""
L4-gradient stress case 05: double detach chain — gradient stops at first detach.

Target feature: gradient-flow check (L4, B1 type).
Bug: Both the encoder output and the projection output are detached. No
gradient reaches either self.encoder or self.proj weights.

Expected:
  WITHOUT L4: Verified
  WITH    L4: Refuted (GRADIENT-BROKEN on first detached tensor)
"""
import torch
import torch.nn as nn


class DoubleDetachChain(nn.Module):
    def __init__(self, dim: int = 48):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        # BUG: first detach — encoder gradient severed
        h = h.detach()
        p = self.proj(h)
        # BUG: second detach — proj gradient also severed
        p = p.detach()
        return self.head(p)


FEATURE = "L4_gradient"
INPUT_SHAPES = {"x": ("batch", 48)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
