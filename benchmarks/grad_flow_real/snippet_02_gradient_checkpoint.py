# bug: torch.utils.checkpoint rewrites the backward graph (recomputation);
#      the first-order grad lattice cannot soundly verify this module.
# source: pattern from memory-efficient training code that uses
#         activation checkpointing on transformer blocks.
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedTransformerBlock(nn.Module):
    """Single transformer block with activation checkpointing."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.ff   = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BUG: checkpoint rewrites backward; out-of-fragment for grad lattice
        return checkpoint(self.ff, self.norm(x))
