import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, hidden_size=1024, lora_r=4):
        super().__init__()
        self.A = nn.Linear(hidden_size, lora_r, bias=False)
        self.B = nn.Linear(lora_r + 1, hidden_size, bias=False)
    def forward(self, x):
        return self.B(self.A(x))
