# bug: the first layer is wrapped in torch.utils.checkpoint, which
#      performs activation recomputation during backward; the first-order
#      grad lattice cannot soundly verify this out-of-fragment construct.
# source: multi-layer network adapted from a memory-efficient training
#         recipe where checkpointing was applied selectively.
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedDeepNet(nn.Module):
    """Three-layer MLP with the first layer activation-checkpointed."""

    def __init__(self, in_dim: int = 16, hidden: int = 16, out_dim: int = 4):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BUG: checkpoint on l1 rewrites the backward graph
        h = checkpoint(self.l1, x)
        h = torch.relu(self.l2(h))
        return self.l3(h)
