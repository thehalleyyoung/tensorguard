import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, head_dim=64):
        super().__init__()
        self.proj = nn.Linear(num_heads * head_dim, hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k_, self.d_ = num_heads, head_dim
    def forward(self, x):
        b, t, _ = x.shape
        return self.proj(self.q(x).view(b, t, self.k_, self.d_).reshape(b, t, -1))
