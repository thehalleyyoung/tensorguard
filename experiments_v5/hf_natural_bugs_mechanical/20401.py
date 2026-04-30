import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (2, 8, 768)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 12
        self.attention_head_size = 64
        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.query(x).view(batch, seq, self.num_attention_heads, self.attention_head_size)
        k = self.key(x).view(batch, seq, self.num_attention_heads, self.attention_head_size)
        v = self.value(x).view(batch, seq, self.num_attention_heads, self.attention_head_size)
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        return torch.matmul(q, k)
