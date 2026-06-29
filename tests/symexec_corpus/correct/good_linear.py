import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 5)
    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    m = M()
    x = torch.zeros(3, 8)
    m(x)
