"""
Upstream-faithful real-bug repro: Longformer global-attention dim swap.
GitHub Issue: https://github.com/huggingface/transformers/issues/5646
Buggy file  : transformers/models/longformer/modeling_longformer.py
              (LongformerSelfAttention.forward pre-fix)
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, num_heads=12, max_num_global_attn_indices=5):
        super().__init__()
        self.num_heads = num_heads
        self.max_num_global_attn_indices = max_num_global_attn_indices

    def forward(self, attn_probs):
        batch_size, num_heads, seq_len, max_num_global = attn_probs.shape
        # BUG (pre-fix): swapped seq_len and max_num_global slots.
        return attn_probs.view(
            batch_size,
            self.num_heads,
            self.max_num_global_attn_indices,
            seq_len,
        )


INPUT_SHAPES = {"attn_probs": (1, 12, 512, 518)}
