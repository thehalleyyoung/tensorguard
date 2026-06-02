import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(96, 160)
        self.norm = nn.LayerNorm(160)
        self.fc2 = nn.Linear(160, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        return self.fc2(torch.relu(x))
