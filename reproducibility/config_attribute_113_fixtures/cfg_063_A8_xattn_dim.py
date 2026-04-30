import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, enc_hidden=4096, dec_hidden=4096):
        super().__init__()
        self.k = nn.Linear(enc_hidden, dec_hidden)
        self.v = nn.Linear(enc_hidden, dec_hidden)
        self.q = nn.Linear(dec_hidden, dec_hidden)
    def forward(self, dec, enc):
        return self.q(dec) @ self.k(enc).transpose(-1, -2)
