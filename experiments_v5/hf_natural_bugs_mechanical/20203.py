import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (2, 8, 768)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(769, 768))
        self.bias = nn.Parameter(torch.zeros(768))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias
