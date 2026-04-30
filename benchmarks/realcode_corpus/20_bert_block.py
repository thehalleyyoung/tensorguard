# input_shape: (4, 32, 256)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, hidden: int = 256, n_heads: int = 8, ff: int = 1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff1 = nn.Linear(hidden, ff)
        self.ff2 = nn.Linear(ff, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(a))
        h = self.ff2(F.gelu(self.ff1(x)))
        return self.norm2(x + self.dropout(h))
