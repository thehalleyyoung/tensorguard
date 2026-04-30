# input_shape: (2, 16, 64, 64)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, ch: int = 16):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4)
        self.c4 = nn.Conv2d(ch, ch, 3, padding=8, dilation=8)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        return F.relu(self.c4(x))
