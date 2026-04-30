"""Targeted: einsum batch label mismatch (2 vs 3)."""
import torch
import torch.nn as nn

class M(nn.Module):
    def forward(self, a, b):
        return torch.einsum("bij,bjk->bik", a, b)


if __name__ == "__main__":
    try:
        M()(torch.randn(2, 3, 4), torch.randn(3, 4, 5))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
