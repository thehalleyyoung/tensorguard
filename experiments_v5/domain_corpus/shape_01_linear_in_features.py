"""
Domain: SHAPE (control — caught by the base L0 shape view)
Bug class: Linear in_features mismatch.
Real-world error: "mat1 and mat2 shapes cannot be multiplied" /
                  "size mismatch" RuntimeError at the first nn.Linear.
Provenance: canonical shape-mismatch pattern; equivalent to thousands of
            "RuntimeError: size mismatch" reports in PyTorch issue history.
Expected: refuted by the base shape view alone (no domain flags required).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SHAPES = {"x": (4, 768)}


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(512, 10)

    def forward(self, x):
        return self.lin(x)
