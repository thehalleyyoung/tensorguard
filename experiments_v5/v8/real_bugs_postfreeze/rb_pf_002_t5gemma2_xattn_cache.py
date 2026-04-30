"""
Post-freeze upstream-faithful real-bug repro: HF transformers T5Gemma2 cross-attention cache
GitHub PR  : https://github.com/huggingface/transformers/pull/45540  (merged 2026-04-21)
GitHub Issue: https://github.com/huggingface/transformers/issues/45521
Repository : huggingface/transformers
Buggy file : src/transformers/models/t5gemma2/modeling_t5gemma2.py
            (pre-#45540, _prepare_cache_for_generation strips
            sliding_window from cross-attn config)

Root cause (per PR body): generation crashes with
``RuntimeError: The size of tensor a (4097) must match the size of
tensor b (5018) at non-singleton dimension 3`` for encoder lengths
>= sliding_window (4096): cross-attention is meant to attend to all
encoder tokens, but the cross-attn config inherits the decoder's
sliding-window mask, capping the K/V at sliding_window+1 while the
encoder K/V remained at full encoder length.

This repro mirrors the buggy attention head: scaled-dot-product-style
matmul of Q (decoder, sliding-window-truncated) against K^T (encoder,
full length). The dimension mismatch is at the ``Q @ K^T`` step.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, num_heads=8, head_dim=64, sliding_window=4096,
                 encoder_len=5018):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # BUG (pre-#45540): cross-attn cache is truncated to
        # sliding_window+1 along the K dim, but encoder Q has the
        # full encoder_len, so the matmul broadcast fails.
        self.q_len_truncated = sliding_window + 1   # 4097
        self.k_len_full = encoder_len               # 5018
        self.q_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)

    def forward(self, hidden):
        # hidden: (B, q_len_truncated, num_heads * head_dim)
        B = hidden.shape[0]
        q = self.q_proj(hidden).view(B, self.q_len_truncated, self.num_heads, self.head_dim).transpose(1, 2)
        # k is materialised at full encoder length (cross-attn full K)
        k_in = hidden.new_zeros(B, self.k_len_full, self.num_heads * self.head_dim)
        k = self.k_proj(k_in).view(B, self.k_len_full, self.num_heads, self.head_dim).transpose(1, 2)
        # Compare against truncated mask shape: q.size(-2) vs k.size(-2)
        # The buggy attention then does (q * scale) @ k.transpose(-1,-2)
        # then adds a mask whose last dim is q_len_truncated (4097)
        # against scores whose last dim is k_len_full (5018) -- mismatch.
        scores = torch.matmul(q, k.transpose(-1, -2))   # (B, H, 4097, 5018)
        # Buggy mask add: a tensor of shape (B, 1, 4097, 4097)
        bad_mask = hidden.new_zeros(B, 1, self.q_len_truncated, self.q_len_truncated)
        return scores + bad_mask    # broadcast mismatch on last dim


INPUT_SHAPES = {"hidden": (1, 4097, 8 * 64)}
