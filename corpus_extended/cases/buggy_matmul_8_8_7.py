import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(7, 5))

    def forward(self, x):
        return x @ self.w
