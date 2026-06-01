"""
Domain: GRADIENT (missed by the shape view; caught only with --grad check)
Bug class: `.detach()` on the only path to the output, silently severing the
           gradient so the parameter never trains.
Real-world error (at backward): "element 0 of tensors does not require grad
                  and does not have a grad_fn".
Provenance: canonical gradient-flow pattern; a frequent silent training bug
            where a `.detach()` (often copied from an inference snippet) ends
            up on the trainable path.
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
        return self.lin(x).detach() * 2
