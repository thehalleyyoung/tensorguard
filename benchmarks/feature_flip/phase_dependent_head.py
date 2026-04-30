"""Real-source example: phase-dependent shape divergence.

A pattern seen in HuggingFace fine-tuning forks where an EVAL-mode head
is added with a shape that does not match the body's output, while the
TRAIN-mode head is correct.  PyTorch only complains when the user
switches the model to ``.eval()``; static checkers that do not
multi-phase the conditional miss it.
"""

import torch
import torch.nn as nn


class PhaseDivergentHead(nn.Module):
    def __init__(self, hidden: int = 8, n_classes: int = 4):
        super().__init__()
        self.body = nn.Linear(hidden, 16)
        self.head_train = nn.Linear(16, n_classes)
        # BUG: eval head expects 32-dim input but body outputs 16
        self.head_eval = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.body(x)
        if self.training:
            return self.head_train(x)
        else:
            return self.head_eval(x)
