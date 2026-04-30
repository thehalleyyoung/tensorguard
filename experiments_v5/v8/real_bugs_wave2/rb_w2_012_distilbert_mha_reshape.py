"""
Upstream-faithful real-bug repro: DistilBERT multi-head attention wrong reshape.

GitHub Issue: https://github.com/huggingface/transformers/issues/11023
Buggy file  : transformers/models/distilbert/modeling_distilbert.py
              (MultiHeadSelfAttention shape_list reshape uses context_len not dim)

DistilBERT multi-head attention reshapes the projected K/V tensors.
Bug: the reshape uses the sequence length as the head dimension instead
of the actual per-head size, causing a mismatch when seq_len != dim_per_head.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, n_heads=12, dim=768, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.dim_per_head = dim // n_heads
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)

    def forward(self, query, key, value):
        bsz, q_len, _ = query.shape
        _, k_len, _ = key.shape

        def shape(x, seq_len):
            # BUG: uses seq_len instead of self.dim_per_head for the last dim
            # x has (bsz, seq, dim) -> should be (bsz, n_heads, seq, dim_per_head)
            # but uses seq as the last dim: (bsz, n_heads, seq, seq_len)
            return x.view(bsz, seq_len, self.n_heads, seq_len).transpose(1, 2)

        q = shape(self.q_lin(query), q_len)  # last dim = q_len, not dim_per_head
        k = shape(self.k_lin(key), k_len)    # fails when dim != n_heads * seq_len
        return q, k


INPUT_SHAPES = {
    "query": (2, 15, 768),
    "key": (2, 15, 768),
    "value": (2, 15, 768),
}
