# input_shape: (8, 3, 64, 64)
# bug: GAN discriminator final FC declared as Linear(256*4*4) but feature
#      map after three stride-2 convs from 64x64 is 8x8, so flatten yields
#      256*8*8=16384 features and the FC will mismatch.
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, ch: int = 3):
        super().__init__()
        self.c1 = nn.Conv2d(ch, 64, 4, stride=2, padding=1)
        self.c2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.c3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 4 * 4, 1)  # WRONG: should be 256*8*8

    def forward(self, x):
        x = F.leaky_relu(self.c1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.c2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.c3(x)), 0.2)
        return self.fc(x.flatten(1))

