import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(32, 48)
        self.fc1 = nn.Linear(48, 48)
        self.fc2 = nn.Linear(48, 48)
        self.fc3 = nn.Linear(48, 10)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(torch.relu(self.fc0(x)))))))
