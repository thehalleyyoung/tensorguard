import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(48, 48)
        self.norm = nn.BatchNorm1d(48)
        self.fc2 = nn.Linear(48, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        return self.fc2(torch.relu(x))
