"""
TensorGuard benchmark corpus -- CLEAN model (clean_layernorm_mlp).

Token-wise MLP with LayerNorm over the feature dimension.
"""

INPUT_SHAPES = {'x': (16, 32, 128)}


import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        return x + self.fc2(h)
