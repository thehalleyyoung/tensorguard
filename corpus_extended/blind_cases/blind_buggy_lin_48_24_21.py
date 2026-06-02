import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(48, 24)
        self.b = nn.Linear(21, 4)

    def forward(self, x):
        return self.b(self.a(x))
