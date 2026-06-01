"""
TensorGuard benchmark corpus -- CLEAN model (clean_conv_bn_pool).

Conv -> BatchNorm -> ReLU -> MaxPool -> flatten -> Linear, all sized correctly.
"""

INPUT_SHAPES = {'x': (4, 1, 28, 28)}


import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))
        x = torch.flatten(x, 1)
        return self.fc(x)
