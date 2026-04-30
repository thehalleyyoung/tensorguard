# input_shape: (8, 3, 32, 32)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    """Single embedding tower used in a triplet network."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, embed_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return F.normalize(self.fc(x), dim=-1)
