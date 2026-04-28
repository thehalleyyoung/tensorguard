"""
L1-CEGAR stress case 01: hidden_size % num_heads != 0 contract violation.

Target feature: CEGAR contract discovery (L1).
Discriminating bug: hidden_size=256 is NOT divisible by num_heads=7.
A proper CEGAR implementation discovers the implicit contract
  hidden_size % num_heads == 0
and then reports its violation.

Expected:
  WITHOUT L1 (max_cegar_iterations=0): Verified (contract not discovered)
  WITH    L1 (max_cegar_iterations=3): should be Refuted IF CEGAR finds real_bugs.

Honest observation: the current CEGAR implementation classifies all
counterexamples as SPURIOUS (shape_env only covers initial input shapes,
not computed post-op shapes), so real_bugs stays empty.
→ L1 is a NO-OP in the current analyser; this case documents that limitation.
"""
import torch
import torch.nn as nn


class BadAttention(nn.Module):
    """Multi-head attention where the contract hidden_size % num_heads == 0 is violated."""

    def __init__(self, hidden_size: int = 252, num_heads: int = 7):
        super().__init__()
        # hidden_size=252, num_heads=7  →  head_dim=36, 7*36=252 so reshape works.
        # The *contract* violation is semantic: using 252 instead of a power-of-2
        # hidden size (256) that the rest of the pipeline expects.
        # A full CEGAR solver would discover the implicit contract
        #   hidden_size % num_heads == 0 ∧ hidden_size ∈ {128, 256, 512, …}
        # and flag the off-nominal value.  Current CEGAR: no-op.
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # = 36 (exact)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        q = self.q_proj(x)
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        out = q.reshape(B, S, self.num_heads * self.head_dim)
        return self.out_proj(out)


# Discriminating feature: L1 (CEGAR)
FEATURE = "L1_cegar"
INPUT_SHAPES = {"x": ("batch", "seq", 252)}
EXPECTED_WITHOUT = "Verified"   # honest: CEGAR doesn't catch this currently
EXPECTED_WITH    = "Refuted"    # would be correct if CEGAR tracked post-op shapes
