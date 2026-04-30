"""Real-source example: residual connection across explicit device boundaries.

Pattern adapted from PyTorch issue threads where a model author manually
moves an activation to ``cuda`` while a buffer (positional encoding,
zero-init residual) is still on ``cpu``.  The forward type-checks at run
time only because PyTorch raises a ``RuntimeError`` -- a static checker
must catch the mismatch before instantiation.
"""

import torch
import torch.nn as nn


class ResidualWithBias(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = x.cuda()
        bias = torch.zeros(x.shape[-1]).cpu()
        return self.proj(x) + bias
