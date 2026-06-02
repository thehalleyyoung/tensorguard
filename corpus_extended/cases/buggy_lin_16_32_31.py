import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 32)
        self.b = nn.Linear(31, 4)

    def forward(self, x):
        return self.b(self.a(x))
