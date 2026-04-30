"""
Post-freeze upstream-faithful real-bug repro: HF diffusers LongCatAudioDiT FFN
GitHub PR  : https://github.com/huggingface/diffusers/pull/13494  (merged 2026-04-16)
Repository : huggingface/diffusers
Buggy file : src/diffusers/models/transformers/transformer_longcat_audio_dit.py
            (pre-#13494, ff_mult hardcoded to 4.0)

Root cause (per PR body): the LongCat-AudioDiT-3.5B checkpoint uses
ff_mult=3.6, but the upstream FFN ctor hardcoded 4.0; loading produces
``size mismatch for blocks.0.ffn.ff.0.weight: copying a param with shape
torch.Size([9216, 2560]) from checkpoint, the shape in current model is
torch.Size([10240, 2560])``. Equivalently, the FFN's first Linear is
``Linear(dim, int(dim * ff_mult))`` and at forward time the chained
``Linear(int(dim * ff_mult_actual), dim)`` rejects the mismatched
intermediate.

This repro mirrors the buggy chain at forward time: a Linear chain whose
intermediate width is sized by the hardcoded ``ff_mult`` while the
input is preprocessed for the checkpoint's ``ff_mult_actual``.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, dim=2560, ff_mult_hardcoded=4.0, ff_mult_actual=3.6):
        super().__init__()
        # BUG (pre-#13494): ff_mult is the hardcoded 4.0; checkpoint
        # was trained with 3.6, so the second Linear expects the
        # 3.6 * dim intermediate.
        intermediate_hardcoded = int(dim * ff_mult_hardcoded)   # 10240
        intermediate_actual    = int(dim * ff_mult_actual)      # 9216
        self.up   = nn.Linear(dim, intermediate_hardcoded)
        self.down = nn.Linear(intermediate_actual, dim)

    def forward(self, x):
        # x: (B, dim).  up -> (B, 10240).  down expects (B, 9216).
        h = self.up(x)
        return self.down(h)


INPUT_SHAPES = {"x": (2, 2560)}
