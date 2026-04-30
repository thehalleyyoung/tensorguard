import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, num_channels=1, stem_channels=3):
        super().__init__()
        self.stem = nn.Conv2d(stem_channels, 32, 3)
        self.in_channels = num_channels
    def forward(self, x):
        return self.stem(x)
