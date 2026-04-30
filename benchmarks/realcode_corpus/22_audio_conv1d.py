# input_shape: (4, 1, 16000)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.c1 = nn.Conv1d(1, 32, 80, stride=16)
        self.c2 = nn.Conv1d(32, 64, 3, stride=2, padding=1)
        self.c3 = nn.Conv1d(64, 128, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
