"""Bug example: ``view`` after ``transpose`` violates the
contiguity precondition.  Without an intervening ``.contiguous()``,
PyTorch raises ``RuntimeError: view size is not compatible with
input tensor's size and stride (at least one dimension spans across
two contiguous subspaces). Use .reshape(...) instead.`` -- a stride
error whose surface text does not name the responsible
``transpose`` call.

TensorGuard's permutation-history theory (Section "five-theory
product domain") flags the constructor-free statement
``y = y.view(B, -1)`` as ill-formed because ``\sigma_\Pi`` is no longer
the identity at that point.
"""
import torch.nn as nn


class TransposeViewBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 32)

    def forward(self, x):                  # x: (B, 8, 64)
        y = x.transpose(1, 2)              # (B, 64, 8) -- non-contiguous
        # BUG: view requires identity permutation history; .reshape or
        # .contiguous().view would be safe here.
        y = y.view(y.size(0), -1)          # CRASH: stride incompatibility
        return self.proj(y)
