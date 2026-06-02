import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(96, 300)
        self.fc1 = nn.Linear(300, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(torch.relu(self.fc0(x)))))
