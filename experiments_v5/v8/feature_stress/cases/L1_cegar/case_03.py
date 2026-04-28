"""
L1-CEGAR stress case 03: config.hidden_size % config.num_heads != 0.

Target feature: CEGAR (L1).
Bug: hidden=200, num_heads=8 → 200 % 8 = 8 != 0.

Honest: CEGAR returns SAFE — non-discriminating in practice.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Config:
    hidden_size: int = 200
    num_heads: int = 8  # 200 % 8 != 0 → BUG


class TransformerLayer(nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        config = config or Config()
        self.config = config
        # BUG: 200 % 8 = 8 (not 0) — head_dim is imprecise
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, S, 3, self.config.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        out = (attn.softmax(-1) @ v).reshape(B, S, H)
        return self.out(out)


FEATURE = "L1_cegar"
INPUT_SHAPES = {"x": ("batch", "seq", 200)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
