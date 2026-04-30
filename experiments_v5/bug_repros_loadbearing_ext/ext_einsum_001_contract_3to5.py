"""Targeted: einsum contracted dim mismatch (3 vs 5).
Exercises src/model_checker.py EINSUM block line ~8278
(`prev.value != cur_dim.value`)."""
import torch
import torch.nn as nn

class M(nn.Module):
    def forward(self, a, b):
        return torch.einsum("ij,jk->ik", a, b)


if __name__ == "__main__":
    try:
        M()(torch.randn(3, 4), torch.randn(5, 6))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
