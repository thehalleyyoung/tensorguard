"""
L2-device stress case 05: register_buffer per-head scale vector on CPU vs CUDA attention.

Target feature: device-consistency check (L2).
Bug: self.head_scales is a registered CPU buffer. When linear output is on CUDA,
adding self.head_scales (CPU) to it causes a device mismatch.

Expected:
  WITHOUT L2: Verified
  WITH    L2: Refuted (DEVICE-MISMATCH on head_scales + linear output)
"""
import torch
import torch.nn as nn


class HeadScaledLinear(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        # BUG: per-head scale factors stay on CPU
        self.register_buffer("head_scales", torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        # head_scales is on CPU; h is on CUDA -> device mismatch
        return h + self.head_scales


FEATURE = "L2_device"
INPUT_SHAPES = {"x": ("batch", 64)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
