# input_shape: (8, 16, 64)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, dim: int = 64, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_experts)])

    def forward(self, x):
        scores = self.gate(x)
        gates = F.softmax(scores, dim=-1)
        # Use the top-1 expert for shape simplicity in static analysis
        out = self.experts[0](x) * gates[..., :1]
        for i in range(1, self.num_experts):
            out = out + self.experts[i](x) * gates[..., i:i + 1]
        return out
