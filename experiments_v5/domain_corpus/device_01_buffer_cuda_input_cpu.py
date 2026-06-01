"""
Domain: DEVICE (missed by the shape view; caught only with --device check)
Bug class: a cuda buffer combined with a cpu input.
Real-world error: "Expected all tensors to be on the same device, but found
                  at least two devices, cuda:0 and cpu!"
Provenance: canonical device-placement pattern; one of the most common
            real-world PyTorch runtime failures (registered buffer left on
            cuda while the module runs on cpu inputs, or vice versa).
Expected: NOT refuted by the base shape view; refuted only when the device
          domain is enabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SHAPES = {"x": (4, 10)}


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.zeros(10, device="cuda"))

    def forward(self, x):
        return x + self.bias
