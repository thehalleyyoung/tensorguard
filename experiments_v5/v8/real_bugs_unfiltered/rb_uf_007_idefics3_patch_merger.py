"""
Unfiltered post-freeze repro #7 (in-fragment, divisibility-bound):
HF transformers - Idefics3 patch-merger view mismatch.

GitHub PR  : https://github.com/huggingface/transformers/pull/45602  (merged 2026-04-23)
Repository : huggingface/transformers
Buggy file : src/transformers/models/idefics3/modeling_idefics3.py
            (pre-#45602, Idefics3SimpleMLP merge step)

Root cause: scale_factor**2 patches are concatenated along the
hidden axis, but the projection Linear was sized for the
unmerged hidden_size.  The buggy module produces
(B, N//s**2, hidden * s**2) and feeds it to Linear(hidden, ...).

In-fragment, expected verdict: RP at >= 0.99.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden_size=1152, scale_factor=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        # BUG (pre-#45602): proj_in expects hidden_size but receives
        # hidden_size * scale_factor**2.
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x: (B, N, hidden_size).  N must be divisible by scale_factor**2.
        B, N, C = x.shape
        merged = x.view(B, N // (self.scale_factor ** 2), C * (self.scale_factor ** 2))
        return self.proj(merged)


INPUT_SHAPES = {"x": (2, 64, 1152)}
