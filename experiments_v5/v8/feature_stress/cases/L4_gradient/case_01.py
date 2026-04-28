"""
L4-gradient stress case 01: .detach() on linear layer output breaks gradient.

Target feature: gradient-flow check (L4, B1 type).
Bug: self.fc(x).detach() breaks the backward path; no gradient flows to
self.fc.weight or self.fc.bias. The loss.backward() call would be a no-op.

Expected:
  WITHOUT L4 (check_gradients=False): Verified (gradient bugs filtered out)
  WITH    L4 (check_gradients=True):  Refuted (GRADIENT-BROKEN violation)
"""
import torch
import torch.nn as nn


class DetachedLinear(nn.Module):
    def __init__(self, in_f: int = 32, out_f: int = 16):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BUG: .detach() kills gradients; fc.weight.grad will be None after backward
        h = self.fc(x).detach()
        return h


FEATURE = "L4_gradient"
INPUT_SHAPES = {"x": ("batch", 32)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
