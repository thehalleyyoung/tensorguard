"""
L2-device stress case 04: register_buffer offset tensor mixed with CUDA linear output.

Target feature: device-consistency check (L2).
Bug: self.offset is a registered CPU buffer. When self.linear(x) produces
a CUDA output, adding self.offset (CPU) causes a device mismatch.

Expected:
  WITHOUT L2: Verified
  WITH    L2: Refuted (DEVICE-MISMATCH on offset + linear output)
"""
import torch
import torch.nn as nn


class OffsetLayer(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        # BUG: offset buffer stays on CPU
        self.register_buffer("offset", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # When x is on CUDA, self.linear(x) is CUDA, self.offset is CPU
        return self.linear(x) + self.offset


FEATURE = "L2_device"
INPUT_SHAPES = {"x": ("batch", 64)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"

