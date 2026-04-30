# input_shape: (4, 3, 32, 32)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.dec1(x))
        return torch.sigmoid(self.dec2(x))
