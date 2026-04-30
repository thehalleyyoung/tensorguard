"""
Upstream-faithful real-bug repro: Diffusers UNet1D GaussianFourier embedding mismatch.
GitHub Issue: https://github.com/huggingface/diffusers/issues/12110
Fixed in PR : https://github.com/huggingface/diffusers/pull/12111
"""
import torch
import torch.nn as nn


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size: int = 8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_size), requires_grad=False)
        self.embedding_size = embedding_size

    def forward(self, x):
        x_proj = x[:, None] * self.weight[None, :] * 2 * 3.141592653589793
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class BuggyModule(nn.Module):
    def __init__(self, block_out_channels=(32, 32, 64), time_embed_dim=128):
        super().__init__()
        timestep_input_dim = 2 * block_out_channels[0]   # 64
        # BUG: hard-coded embedding_size=8 -> 2*8=16 NOT timestep_input_dim
        self.time_proj = GaussianFourierProjection(embedding_size=8)
        self.linear_1 = nn.Linear(timestep_input_dim, time_embed_dim)

    def forward(self, timesteps):
        emb = self.time_proj(timesteps)        # (batch, 16)
        return self.linear_1(emb)              # expects 64, gets 16


INPUT_SHAPES = {"timesteps": (1,)}
