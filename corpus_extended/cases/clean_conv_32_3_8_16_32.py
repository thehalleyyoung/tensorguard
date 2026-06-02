import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 8, 3, padding=1)
        self.c1 = nn.Conv2d(8, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32768, 10)

    def forward(self, x):
        x = F.relu(self.c2(F.relu(self.c1(F.relu(self.c0(x))))))
        x = torch.flatten(x, 1)
        return self.fc(x)
