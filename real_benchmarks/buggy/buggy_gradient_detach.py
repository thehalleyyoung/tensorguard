"""
TensorGuard benchmark corpus -- BUGGY model (buggy_gradient_detach).

Provenance: canonical_pattern

An intermediate activation is detached, silently severing the gradient path to fc1 -- the canonical 'parameters never update' bug. This is a SILENT bug: it raises no runtime exception, so runtime testing misses it; TensorGuard flags it statically.
"""

INPUT_SHAPES = {'x': (4, 8)}


import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        h = self.fc1(x)
        h = h.detach()  # BUG: severs gradient flow to fc1
        return self.fc2(h)
