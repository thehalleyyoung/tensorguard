"""Bug example: ``torch.cat`` along ``dim=1`` (the channel axis)
is requested for two feature maps with different *spatial* extents
(H, W).  The runtime error is ``RuntimeError: Sizes of tensors must
match except in dimension 1. Expected size 16 but got size 8 for
tensor number 1 in the list.`` -- the symptom localises to the
``torch.cat`` call but does not explain *which* upstream module is
producing the smaller spatial extent.

TensorGuard reports a per-axis disagreement directly attributable to
``self.down`` (a stride-2 conv) vs. the unstrided ``self.same``,
allowing the user to either (a) align spatial sizes via
``F.interpolate`` or (b) move the stride-2 conv onto both branches.
"""
import torch
import torch.nn as nn


class CatSpatialMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.same = nn.Conv2d(3, 16, 3, padding=1)            # keeps H, W
        self.down = nn.Conv2d(3, 16, 3, padding=1, stride=2)  # halves H, W

    def forward(self, x):                       # x: (B, 3, 16, 16)
        a = self.same(x)                        # (B, 16, 16, 16)
        b = self.down(x)                        # (B, 16,  8,  8)
        # BUG: spatial extents disagree on dims 2 and 3.
        return torch.cat([a, b], dim=1)         # CRASH
