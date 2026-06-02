import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(20, 48)
        self.b = nn.Linear(53, 4)

    def forward(self, x):
        return self.b(self.a(x))
