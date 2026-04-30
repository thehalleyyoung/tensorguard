# input_shape: (8, 100)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, latent: int = 100, out_ch: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent, 256 * 4 * 4)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(64, out_ch, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        h = self.fc(x).reshape(-1, 256, 4, 4)
        h = F.relu(self.bn1(self.up1(h)))
        h = F.relu(self.bn2(self.up2(h)))
        return torch.tanh(self.up3(h))
