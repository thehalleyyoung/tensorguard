import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, latent_dim=32, target_dim=128):
        super().__init__()
        self.dec = nn.Linear(latent_dim, target_dim * 2)
        self.tail = nn.Linear(target_dim, 3)
    def forward(self, z):
        return self.tail(self.dec(z))
