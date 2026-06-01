"""
TensorGuard benchmark corpus -- BUGGY model (buggy_device_mismatch).

Provenance: canonical_pattern
Expected Error: Expected all tensors to be on the same device

A CUDA buffer is added to a CPU activation -- the canonical 'Expected all tensors to be on the same device' failure. The device mismatch only raises at runtime on a CUDA-enabled host; TensorGuard detects it statically without any GPU.
"""

INPUT_SHAPES = {'x': (4, 8)}


import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.zeros(8, device="cuda"))
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc(x) + self.bias  # BUG: cuda buffer added to cpu activation
