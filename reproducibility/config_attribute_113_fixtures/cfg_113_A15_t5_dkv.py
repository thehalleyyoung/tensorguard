import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, d_model=768, num_heads=12, d_kv=65):
        super().__init__()
        self.q = nn.Linear(d_model, num_heads * d_kv)
        self.o = nn.Linear(num_heads * 64, d_model)
    def forward(self, x):
        b, t, _ = x.shape
        return self.o(self.q(x).view(b, t, -1))
