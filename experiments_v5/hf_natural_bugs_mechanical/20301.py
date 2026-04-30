import torch
import torch.nn as nn

INPUT_SHAPES = {"hidden_states": (2, 8, 512)}


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = 8
        self.key_value_proj_dim = 64
        self.q = nn.Linear(512, 512, bias=False)
        self.k = nn.Linear(512, 512, bias=False)
        self.v = nn.Linear(512, 512, bias=False)

    def forward(self, hidden_states):
        batch, seq, _ = hidden_states.shape
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        q = q.view(batch, seq, 9, self.key_value_proj_dim).transpose(1, 2)
        k = k.view(batch, seq, 9, self.key_value_proj_dim).transpose(1, 2)
        v = v.view(batch, seq, 9, self.key_value_proj_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2))
        return torch.matmul(scores, v)
