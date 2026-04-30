import torch
import torch.nn as nn

INPUT_SHAPES = {"x": (2, 8, 768)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = 12
        self.dim = 768
        self.dim_per_head = 64
        self.q_lin = nn.Linear(768, 768)
        self.k_lin = nn.Linear(768, 768)
        self.v_lin = nn.Linear(768, 768)
        self.out_lin = nn.Linear(768, 768)

    def forward(self, x):
        batch, seq, _ = x.shape

        def shape(t):
            return t.view(batch, seq, self.n_heads, self.dim_per_head + 1).transpose(1, 2)

        q = shape(self.q_lin(x))
        k = shape(self.k_lin(x))
        v = shape(self.v_lin(x))
        scores = torch.matmul(q, k.transpose(-1, -2))
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.dim)
        return self.out_lin(context)
