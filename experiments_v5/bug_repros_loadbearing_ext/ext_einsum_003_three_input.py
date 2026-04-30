"""Targeted: einsum three-input chain with mismatched contracted dim."""
import torch
import torch.nn as nn

class M(nn.Module):
    def forward(self, a, b, c):
        return torch.einsum("ij,jk,kl->il", a, b, c)


if __name__ == "__main__":
    try:
        M()(torch.randn(2, 3), torch.randn(3, 4), torch.randn(5, 6))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
