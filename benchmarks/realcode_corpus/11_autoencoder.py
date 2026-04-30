# input_shape: (32, 784)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, in_dim: int = 784, latent: int = 32):
        super().__init__()
        self.enc1 = nn.Linear(in_dim, 256)
        self.enc2 = nn.Linear(256, latent)
        self.dec1 = nn.Linear(latent, 256)
        self.dec2 = nn.Linear(256, in_dim)

    def forward(self, x):
        z = F.relu(self.enc1(x))
        z = self.enc2(z)
        h = F.relu(self.dec1(z))
        return torch.sigmoid(self.dec2(h))
