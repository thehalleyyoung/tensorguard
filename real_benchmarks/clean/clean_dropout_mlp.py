"""
TensorGuard benchmark corpus -- CLEAN model (clean_dropout_mlp).

Regression MLP using Dropout; correct in both train and eval phases.
"""

INPUT_SHAPES = {'x': (64, 100)}


import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
