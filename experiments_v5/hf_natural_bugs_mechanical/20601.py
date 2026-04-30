import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (2, 8, 1024)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 16
        self.head_dim = 64
        self.embed_dim = 1024
        self.q_proj = nn.Linear(1024, 1024)
        self.k_proj = nn.Linear(1024, 1024)
        self.v_proj = nn.Linear(1024, 1024)
        self.out_proj = nn.Linear(1024, 1024)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        att = torch.matmul(q, k.transpose(-2, -1))
        attn_out = torch.matmul(att, v)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(batch, seq, 1025)
        return self.out_proj(attn_out)
