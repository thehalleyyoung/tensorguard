"""
Domain: DEVICE (missed by the shape view; caught only with --device check)
Bug class: an in-forward `.cuda()` move applied to a cpu-resident buffer that
           is then multiplied against a cpu input.
Real-world error: "Expected all tensors to be on the same device, but found
                  at least two devices, cuda:0 and cpu!"
Provenance: canonical device-placement pattern; mirrors the common mistake of
            calling `.cuda()` inside forward on one operand only.
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
        self.register_buffer("scale", torch.ones(10))

    def forward(self, x):
        return x * self.scale.cuda()
