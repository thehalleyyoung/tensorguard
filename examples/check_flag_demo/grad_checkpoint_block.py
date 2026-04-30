"""Real-source example: gradient-checkpointed module out of fragment.

``torch.utils.checkpoint.checkpoint`` performs a second-order graph
rewrite (forward discards activations, backward recomputes them) that
the first-order grad lattice cannot soundly verify.  The intended
behaviour is REFUTED-PROOF (out-of-fragment) when ``check_gradients``
is on, so that a runtime grad-flag check cannot silently report this
module as VERIFIED.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedBlock(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, x):
        # Activation checkpointing rewrites the backward graph
        return checkpoint(self.lin, x)
