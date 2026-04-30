"""Targeted: einsum 3D explicit-mode mismatch on shared label."""
import torch
import torch.nn as nn

class M(nn.Module):
    def forward(self, a, b):
        return torch.einsum("abc,acd->abd", a, b)


if __name__ == "__main__":
    try:
        M()(torch.randn(2, 3, 4), torch.randn(2, 5, 6))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
