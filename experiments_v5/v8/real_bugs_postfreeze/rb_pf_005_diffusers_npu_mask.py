"""
Post-freeze upstream-faithful real-bug repro: HF diffusers NPU attention mask shape
GitHub PR  : https://github.com/huggingface/diffusers/pull/13490  (merged 2026-04-16)
Repository : huggingface/diffusers
Buggy file : src/diffusers/models/attention_processor.py (NPU path, pre-#13490)

Root cause (per PR body): NPU's fusion attention operator does not
support broadcasting attention masks. A mask of shape ``[B, 1, 1, S]``
was passed to an op that expects ``[B, N, S, S]`` (or similar
non-broadcasting shapes); the upstream patch explicitly expands the
mask to ``[B, 1, S, S]`` before dispatch.

This repro mirrors the buggy mask reshape: mask is constructed at
``[B, 1, 1, S]`` and added to attention scores of shape ``[B, N, S, S]``;
on a non-broadcasting backend this is a shape error.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, num_heads=8, head_dim=64, seq_len=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.q_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)

    def forward(self, hidden, mask):
        # hidden: (B, S, num_heads * head_dim); mask: (B, 1, 1, S)
        B = hidden.shape[0]
        q = self.q_proj(hidden).view(B, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2))                # (B, N, S, S)
        # BUG (pre-#13490): mask comes in at (B,1,1,S); on the NPU
        # path this triggers an unsupported-broadcast error.  The
        # static-checkable analogue: the buggy code path then
        # reshapes the mask to (B, 1, S, S+1) by mistake.
        bad_mask = mask.expand(B, 1, self.seq_len, self.seq_len + 1)
        return scores + bad_mask


INPUT_SHAPES = {"hidden": (1, 128, 8 * 64), "mask": (1, 1, 1, 128)}
