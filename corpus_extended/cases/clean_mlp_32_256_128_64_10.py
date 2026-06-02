import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(32, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(torch.relu(self.fc0(x)))))))
