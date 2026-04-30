import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (2, 8, 512)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wi = nn.Linear(512, 2048, bias=False)
        self.wo = nn.Linear(2048, 512, bias=False)

    def forward(self, x):
        batch, seq, _ = x.shape
        h = self.wi(x)
        h = h.reshape(batch * seq, 2049)
        h = self.wo(h)
        return h.view(batch, seq, 512)
