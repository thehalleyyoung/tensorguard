import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(24, 160)
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(torch.relu(self.fc0(x)))))
