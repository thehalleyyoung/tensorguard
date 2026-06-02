import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(24, 120)
        self.fc1 = nn.Linear(120, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(torch.relu(self.fc0(x)))))))
