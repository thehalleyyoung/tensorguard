import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, hidden_size=512, intermediate_size=2048,
                 gate_size=1024):
        super().__init__()
        self.up = nn.Linear(hidden_size, intermediate_size)
        self.gate = nn.Linear(hidden_size, gate_size)
        self.down = nn.Linear(intermediate_size, hidden_size)
    def forward(self, x):
        return self.down(self.up(x) * self.gate(x))
