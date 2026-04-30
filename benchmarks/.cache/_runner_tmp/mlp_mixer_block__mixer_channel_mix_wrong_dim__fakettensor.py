"""Adapted from lucidrains/mlp-mixer-pytorch (single Mixer block, einops-free)."""
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    """One Mixer block: token-mixing then channel-mixing."""

    def __init__(self, num_patches: int, dim: int, token_hidden: int, channel_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = FeedForward(num_patches, token_hidden, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = FeedForward(num_patches, channel_hidden, dropout)

    def forward(self, x):
        # x: [B, num_patches, dim]
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mix(y).transpose(1, 2)
        x = x + y
        z = self.norm2(x)
        z = self.channel_mix(z)
        return x + z
