import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 12, 3, padding=1)
        self.fc = nn.Linear(1731, 10)

    def forward(self, x):
        x = F.relu(self.c(x))
        x = torch.flatten(x, 1)
        return self.fc(x)
