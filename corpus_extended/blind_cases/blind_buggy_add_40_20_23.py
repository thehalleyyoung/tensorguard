import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(40, 20)
        self.b = nn.Linear(40, 23)

    def forward(self, x):
        return self.a(x) + self.b(x)
