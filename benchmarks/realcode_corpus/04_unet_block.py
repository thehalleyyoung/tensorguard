# input_shape: (2, 3, 64, 64)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec = nn.Conv2d(base * 2, base, 3, padding=1)
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool(e1)))
        u = self.up(e2)
        d = F.relu(self.dec(torch.cat([u, e1], dim=1)))
        return self.out(d)
