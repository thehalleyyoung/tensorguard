"""
Upstream-faithful real-bug repro: T5 attention position bias wrong shape.

GitHub Issue: https://github.com/huggingface/transformers/issues/9984
Buggy file  : transformers/models/t5/modeling_t5.py
              (T5Attention relative_attention_bias embedding lookup shape)

T5 attention uses relative position biases. The bug: the embedding is
indexed incorrectly, producing shape (batch, heads, seq, seq) when the
bias is (1, num_heads, seq, seq) — the batch dimension is wrong because
expand is applied on the num_heads axis not the batch axis.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, d_model=512, num_heads=8, relative_attention_num_buckets=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_value_proj_dim = d_model // num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q(hidden_states).view(bsz, seq_len, self.num_heads, self.key_value_proj_dim)
        k = self.k(hidden_states).view(bsz, seq_len, self.num_heads, self.key_value_proj_dim)

        # bias lookup: (seq, seq) -> relative_attention_bias -> (seq, seq, num_heads)
        # BUG: reshape to (bsz, num_heads, seq, seq) with wrong bsz dimension
        # The bias is (1, num_heads, seq, seq) but we try to reshape (seq*seq, num_heads)
        # as (bsz, num_heads, seq, seq) when bsz != 1
        bias = self.relative_attention_bias.weight  # (num_buckets, num_heads)
        # Simulate indexed lookup result: (1, num_heads, seq, seq)
        bias_indexed = bias[:1].view(1, self.num_heads, 1, 1).expand(1, self.num_heads, seq_len, seq_len)
        # BUG: try to reshape as (bsz, num_heads, seq, seq) — fails when bsz > 1
        bias_wrong = bias_indexed.view(bsz, self.num_heads, seq_len, seq_len)
        return q.transpose(1, 2) + bias_wrong


INPUT_SHAPES = {"hidden_states": (3, 6, 512)}
