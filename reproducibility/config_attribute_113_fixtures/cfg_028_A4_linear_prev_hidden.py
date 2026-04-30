import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, hidden_size=1024, prev_hidden=1024):
        super().__init__()
        self.l = nn.Linear(prev_hidden, hidden_size)
    def forward(self, x):
        return self.l(x)
