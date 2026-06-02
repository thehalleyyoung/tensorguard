import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 16)
        self.b = nn.Linear(8, 14)

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x)], dim=0)
