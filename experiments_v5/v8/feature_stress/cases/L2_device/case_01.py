"""
L2-device stress case 01: register_buffer stays on CPU while inputs go to CUDA.

Target feature: device-consistency check (L2).
Bug: self.bias is registered as a CPU buffer; when the model moves to GPU and
forward() receives a CUDA input, the addition self.linear(x) + self.bias
causes a device mismatch at runtime.

Expected:
  WITHOUT L2 (check_devices=False): Verified (device bugs filtered out)
  WITH    L2 (check_devices=True):  Refuted (DEVICE-MISMATCH violation)
"""
import torch
import torch.nn as nn


class BiasedLinear(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # BUG: buffer registered on CPU; not moved with model.cuda()
        self.register_buffer("bias", torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # When x is on CUDA, self.bias is still on CPU → RuntimeError
        return self.linear(x) + self.bias


FEATURE = "L2_device"
INPUT_SHAPES = {"x": ("batch", 64)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
