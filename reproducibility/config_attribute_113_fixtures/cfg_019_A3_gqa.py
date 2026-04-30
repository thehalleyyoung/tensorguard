import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, hidden_size=1024, num_q=16, num_kv=4):
        super().__init__()
        self.head_dim = hidden_size // num_q
        self.q = nn.Linear(hidden_size, num_q * self.head_dim)
        self.k = nn.Linear(hidden_size, num_kv * self.head_dim)
        self.num_q, self.num_kv = num_q, num_kv
    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.num_q, self.head_dim)
        k = self.k(x).view(b, t, self.num_kv, self.head_dim)
        return q @ k.transpose(-1, -2)
