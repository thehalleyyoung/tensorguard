import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SHAPES = {"x": (2, 8, 4096)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4096, 11008, bias=False)
        self.up_proj = nn.Linear(4096, 11008, bias=False)
        self.down_proj = nn.Linear(11008, 4096, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = gate.transpose(1, 2)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)
