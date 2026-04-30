"""
Post-freeze upstream-faithful real-bug repro: HF diffusers Qwen Image DreamBooth batch ordering
GitHub PR  : https://github.com/huggingface/diffusers/pull/13441  (merged 2026-04-10)
Repository : huggingface/diffusers
Buggy file : examples/dreambooth/train_dreambooth_lora_qwen_image.py (pre-#13441)

Root cause (per PR body): under ``--with_prior_preservation`` the
script ``repeat(...)``s prompt embeddings, which interleaves the
batch as ``[inst, class, inst, class, ...]``. The dataloader and the
later ``torch.chunk(..., 2, dim=0)`` logic assume *grouped* ordering
``[inst1..instB, class1..classB]``. The batch-size mismatch surfaces
when ``weighting`` is *not* chunked alongside ``model_pred`` and
``target``: ``weighting`` keeps the doubled batch while ``model_pred``
is split, producing a shape error on ``model_pred * weighting``.

This repro mirrors the buggy interleaving + missing chunk: the
``weighting`` keeps the full doubled batch B*2 while ``model_pred``
is chunked to B; their elementwise multiply mismatches.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden=64, train_batch=4):
        super().__init__()
        self.train_batch = train_batch
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, latents):
        # latents: (2 * train_batch, hidden)  -- prior-preservation doubles batch
        full = self.proj(latents)
        # Chunk model_pred into instance vs class halves (correct for split).
        model_pred, _ = torch.chunk(full, 2, dim=0)        # (train_batch, hidden)
        target = torch.zeros_like(model_pred)              # (train_batch, hidden)
        # BUG (pre-#13441): weighting is not chunked alongside model_pred.
        weighting = latents.new_ones(2 * self.train_batch, 1)   # (2*B, 1)
        loss = (model_pred - target) * weighting           # mismatch on dim 0
        return loss


INPUT_SHAPES = {"latents": (2 * 4, 64)}
