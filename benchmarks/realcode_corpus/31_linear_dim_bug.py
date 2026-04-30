# input_shape: (4, 32)
# bug: Linear in_features=64 but feeding tensor of last-dim 32.
# cause_line: 11
# expected: 32, 64
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 16)  # WRONG: should be 32

    def forward(self, x):
        return self.fc1(x)
