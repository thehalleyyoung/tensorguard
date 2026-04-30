"""
Unfiltered post-freeze repro #15 (in-fragment, transpose dim swap):
HF diffusers - Cosmos2 vision tower transpose.

GitHub PR  : https://github.com/huggingface/diffusers/pull/13580  (merged 2026-04-25)
Repository : huggingface/diffusers
Buggy file : src/diffusers/models/transformers/transformer_cosmos2.py
            (pre-#13580, Cosmos2VisionTower._merge_patches)

Root cause: the vision tower transposes the (height, width) axes
before the patch projection, but the projection Linear's weight
was built assuming (width, height).  The transpose swap
produces a Linear-in-features mismatch on the very next op.

In-fragment, expected verdict: RP at >= 0.99.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden=512, h=14, w=10):
        super().__init__()
        # Linear is sized for (B, h * w, hidden) in the upstream-correct
        # path; the buggy transpose inverts this to (B, w * h, hidden)
        # and then a misordered .reshape() produces a different
        # in-features for the projection.
        self.h = h
        self.w = w
        self.hidden = hidden
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        # x: (B, hidden, h, w).  Upstream-correct: x.flatten(2).transpose(1, 2)
        # BUG (pre-#13580): transposed before flatten, producing a
        # different per-patch layout that the next view rejects.
        B, C, H, W = x.shape
        # The view target uses h * w, but the transposed input has shape
        # (B, hidden, w, h) so view to (B, h * w, hidden) fails when
        # h != w.
        bad = x.transpose(2, 3)              # (B, hidden, w, h)
        merged = bad.view(B, self.h * self.w, self.hidden)
        return self.proj(merged)


INPUT_SHAPES = {"x": (1, 512, 14, 10)}
