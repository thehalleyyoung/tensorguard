import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (2, 8, 4096)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.q_proj = nn.Linear(4096, 4096, bias=False)
        self.k_proj = nn.Linear(4096, 1024, bias=False)
        self.v_proj = nn.Linear(4096, 1024, bias=False)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        return torch.matmul(q, k.transpose(-2, -1))
