"""
L4-gradient stress case 02: .detach() on layer output inside residual connection
severs gradient flow through the main branch.

Target feature: gradient-flow check (L4, B1 type).
Bug: After applying self.norm and self.fc, the output is detached before
adding the residual. This means the gradient is broken for the output of fc.

Expected:
  WITHOUT L4: Verified
  WITH    L4: Refuted (GRADIENT-BROKEN)
"""
import torch
import torch.nn as nn


class BrokenResidual(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        # BUG: detach severs gradient from self.fc output
        h = self.fc(h).detach()
        return h + x


FEATURE = "L4_gradient"
INPUT_SHAPES = {"x": ("batch", "seq", 64)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"

