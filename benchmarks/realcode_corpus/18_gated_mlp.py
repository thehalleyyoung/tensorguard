# input_shape: (16, 64)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 256):
        super().__init__()
        self.gate = nn.Linear(dim, hidden)
        self.value = nn.Linear(dim, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        v = self.value(x)
        h = g * v
        return self.out(h)
