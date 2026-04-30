import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (2, 8, 768)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(768, 3072)
        self.lin2 = nn.Linear(3072, 768)

    def forward(self, x):
        batch, seq, _ = x.shape
        h = self.lin1(x)
        h = h.transpose(1, 2)
        return self.lin2(h)
