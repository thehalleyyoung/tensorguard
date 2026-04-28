"""
L2-device stress case 03: running_mean buffer in manual BatchNorm stays on CPU.

Target feature: device-consistency check (L2).
Bug: running_mean is registered on CPU; when input x is on CUDA, the
subtraction x - self.running_mean raises a device mismatch.

Expected:
  WITHOUT L2: Verified
  WITH    L2: Refuted (DEVICE-MISMATCH)
"""
import torch
import torch.nn as nn


class ManualBatchNorm(nn.Module):
    """Simplified manual batch norm with explicit running statistics buffer."""

    def __init__(self, num_features: int = 64):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias_param = nn.Parameter(torch.zeros(num_features))
        # BUG: running stats registered as CPU buffers
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Device mismatch when x is on CUDA
        x_norm = (x - self.running_mean) / (self.running_var + 1e-5).sqrt()
        return self.weight * x_norm + self.bias_param


FEATURE = "L2_device"
INPUT_SHAPES = {"x": ("batch", 64)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
