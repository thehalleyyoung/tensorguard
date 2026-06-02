import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(20, 6)
        self.b = nn.Linear(20, 9)

    def forward(self, x):
        return self.a(x) + self.b(x)
