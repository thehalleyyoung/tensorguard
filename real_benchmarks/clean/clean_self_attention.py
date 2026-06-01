"""
TensorGuard benchmark corpus -- CLEAN model (clean_self_attention).

Single-head scaled dot-product attention with consistent projections.
"""

INPUT_SHAPES = {'x': (2, 10, 64)}


import torch
import torch.nn as nn


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64, 64)
        self.k = nn.Linear(64, 64)
        self.v = nn.Linear(64, 64)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
