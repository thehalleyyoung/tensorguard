"""
Unfiltered post-freeze repro #14 (out-of-fragment, autograd checkpoint):
HF transformers - SmolLM3 gradient-checkpointing param sharing.

GitHub PR  : https://github.com/huggingface/transformers/pull/45650  (merged 2026-04-25)
Repository : huggingface/transformers
Buggy file : src/transformers/models/smollm3/modeling_smollm3.py
            (pre-#45650, SmolLM3ForCausalLM with gradient_checkpointing)

Root cause: the input/output embedding tied weights through the
checkpoint wrapper produced double-counted gradient and triggered
a leaf-tensor in-place modification error during backward.  Pure
forward shape is correct.

Out-of-fragment for TG (the first-order grad-flag lattice
{has_grad, no_grad, top} cannot reason about parameter sharing
or `torch.utils.checkpoint`, as flagged in the paper's
Limitations).  Expected verdict: silent verified (no forward
shape error) in TG's user-visible regime.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, vocab=32000, hidden=512):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        # BUG (pre-#45650): tied weights + gradient checkpoint
        # double-counts grads.  We tie weights to mimic the upstream.
        self.lm_head.weight = self.embed.weight

    def forward(self, ids):
        # ids: (B, S)
        h = self.embed(ids)
        # Real upstream wraps this in torch.utils.checkpoint.checkpoint;
        # we omit the wrapper because TG would inline the body anyway.
        return self.lm_head(h)


INPUT_SHAPES = {"ids": (1, 16)}
