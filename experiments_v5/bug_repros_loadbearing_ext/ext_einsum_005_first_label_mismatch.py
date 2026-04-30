"""Targeted: einsum leading-label mismatch (4 vs 6)."""
import torch
import torch.nn as nn

class M(nn.Module):
    def forward(self, a, b):
        return torch.einsum("xy,yz->xz", a, b)


if __name__ == "__main__":
    try:
        M()(torch.randn(4, 7), torch.randn(6, 5))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
