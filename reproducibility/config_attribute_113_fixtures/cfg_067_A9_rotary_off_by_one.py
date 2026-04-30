import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, num_layers=24, cache_layers=24):
        super().__init__()
        self.cache = nn.Parameter(torch.zeros(cache_layers, 32, 64))
        self.proj = nn.Linear(64, 64)
        self.n = num_layers
    def forward(self, x):
        return self.proj(x + self.cache[self.n])
