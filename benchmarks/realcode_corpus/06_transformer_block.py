# input_shape: (16, 32, 128)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, ff: int = 512):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ff)
        self.ff2 = nn.Linear(ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        h = self.ff2(F.gelu(self.ff1(x)))
        return self.norm2(x + h)
