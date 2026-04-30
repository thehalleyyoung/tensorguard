# input_shape: (32, 128)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        x = self.norm(x + h)
        return F.relu(self.fc3(x))
