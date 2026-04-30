# input_shape: (2, 3, 224, 224)
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 384, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # (B, C, H, W) → (B, E, H/p, W/p) → (B, N, E)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)
