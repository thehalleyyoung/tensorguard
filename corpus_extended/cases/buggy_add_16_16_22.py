import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 16)
        self.b = nn.Linear(16, 22)

    def forward(self, x):
        return self.a(x) + self.b(x)
