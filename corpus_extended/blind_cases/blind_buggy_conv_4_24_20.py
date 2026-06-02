import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(4, 24, 3, padding=1)
        self.c2 = nn.Conv2d(20, 16, 3, padding=1)

    def forward(self, x):
        return self.c2(F.relu(self.c1(x)))
