"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/177482
Expected Error: must match the size of tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SHAPES = {"q": (2, 8, 10, 14), "k": (2, 4, 12, 14), "v": (2, 4, 12, 16),
                "mask": (2, 10, 12)}


class BuggyModule(nn.Module):
    def forward(self, q, k, v, mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)


if __name__ == '__main__':
    try:
        m = BuggyModule()
        N, H, Hq, L, S, E, Ev = 2, 4, 8, 10, 12, 14, 16
        q = torch.randn(N, Hq, L, E)
        k = torch.randn(N, H, S, E)
        v = torch.randn(N, H, S, Ev)
        mask = torch.ones(N, L, S, dtype=torch.bool)
        m(q, k, v, mask)
    except RuntimeError as e:
        print(f"Error: {e}")
