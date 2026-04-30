import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, hidden_size=4096, prev_hidden=5120):
        super().__init__()
        self.l = nn.Linear(prev_hidden, hidden_size)
    def forward(self, x):
        return self.l(x)
