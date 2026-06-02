import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 4)
        self.b = nn.Linear(16, 2)

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x)], dim=0)
