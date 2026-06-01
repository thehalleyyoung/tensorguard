"""
Domain: GRADIENT (missed by the shape view; caught only with --grad check)
Bug class: the forward returns a `.detach()`ed tensor as its result, so the
           whole module output has no grad_fn.
Real-world error (at backward): "element 0 of tensors does not require grad
                  and does not have a grad_fn".
Provenance: canonical gradient-flow pattern; the returned value is detached,
            a common refactor mistake when extracting a feature head.
Expected: NOT refuted by the base shape view; refuted only when the gradient
          domain is enabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SHAPES = {"x": (4, 10)}


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10)

    def forward(self, x):
        h = self.lin(x)
        return h.detach()
