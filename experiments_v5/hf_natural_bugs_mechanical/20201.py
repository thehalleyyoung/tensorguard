import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (2, 8, 768)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = 12
        self.head_dim = 64
        self.c_attn = nn.Linear(768, 768 * 3)
        self.c_proj = nn.Linear(768, 768)

    def forward(self, x):
        batch, seq, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(768, dim=2)
        q = q.view(batch, seq, self.n_head, self.head_dim + 1).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, self.head_dim + 1).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, self.head_dim + 1).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1))
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq, 768)
        return self.c_proj(y)
