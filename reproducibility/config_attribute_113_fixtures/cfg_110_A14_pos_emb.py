import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, max_seq=1024, pos_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(pos_len, 64))
        self.s = max_seq
    def forward(self, x):
        return x + self.pe[:self.s]
