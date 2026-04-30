# input_shape: (16, 8)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    """Affine coupling layer of a normalizing flow (RealNVP-style).

    Splits input in half along the last dim and applies a learned
    shift+scale to one half conditioned on the other.
    """

    def __init__(self, dim: int = 8, hidden: int = 32):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        st = self.net(x1)
        s, t = st.chunk(2, dim=-1)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=-1)
