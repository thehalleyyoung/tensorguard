"""
TensorGuard benchmark corpus -- CLEAN model (clean_mlp).

Two-layer MLP with matching feature dimensions.
"""

INPUT_SHAPES = {'x': (32, 784)}


import torch
import torch.nn as nn


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
