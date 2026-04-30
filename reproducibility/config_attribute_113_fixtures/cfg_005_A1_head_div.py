import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, hidden_size=512, num_heads=7):
        super().__init__()
        self.h = hidden_size
        self.nh = num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
    def forward(self, x):
        b, t, _ = x.shape
        qkv = self.qkv(x)
        return qkv.view(b, t, 3, self.nh, self.h // self.nh)
