import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (2, 8, 768)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(768, 3072)
        self.c_proj = nn.Linear(3072, 768)

    def forward(self, x):
        batch, seq, _ = x.shape
        h = self.c_fc(x)
        h = h.reshape(batch * seq, 3073)
        h = self.c_proj(h)
        return h.view(batch, seq, 768)
