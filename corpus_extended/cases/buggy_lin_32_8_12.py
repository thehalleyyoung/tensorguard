import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(32, 8)
        self.b = nn.Linear(12, 4)

    def forward(self, x):
        return self.b(self.a(x))
