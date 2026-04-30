# bug: residual connection formed by adding a detached shortcut to the
#      transformed output, then the sum itself is also detached; no
#      gradient flows to the transform parameters.
# source: residual blocks where the author mistakenly detaches both the
#         shortcut and the final output, believing it speeds up training.
import torch
import torch.nn as nn


class BrokenResBlock(nn.Module):
    """Residual block with doubly-broken gradient path."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.transform = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.detach()                        # BUG: kills grad on shortcut
        out      = self.transform(x) + shortcut
        return out.detach()                          # BUG: kills grad on transform output too
