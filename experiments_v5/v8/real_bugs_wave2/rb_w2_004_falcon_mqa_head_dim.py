"""
Upstream-faithful real-bug repro: Falcon multi-query attention head mismatch.

GitHub Issue: https://github.com/huggingface/transformers/issues/25111
Buggy file  : transformers/models/falcon/modeling_falcon.py
              (FalconAttention new_context_layer_shape uses wrong dim)

Falcon uses multi-query attention (1 KV head per N query heads).
The bug: when computing the output, uses num_heads (query heads) where
the split was done with num_kv_heads, creating a shape mismatch.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden_size=4096, num_attention_heads=71, num_kv_heads=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_attention_heads
        # Falcon: separate Q, KV projections
        self.query = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.key_value = nn.Linear(hidden_size, 2 * num_kv_heads * self.head_dim, bias=False)

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.shape
        q = self.query(hidden_states)  # (bsz, q_len, num_attn_heads * head_dim)
        kv = self.key_value(hidden_states)  # (bsz, q_len, 2 * num_kv_heads * head_dim)

        # BUG: reshape q with wrong head_dim because num_attention_heads doesn't divide hidden_size
        # 4096 // 71 = 57 (truncated), but 71*57 = 4047 != 4096
        q = q.view(bsz, q_len, self.num_attention_heads, self.head_dim)
        return q


INPUT_SHAPES = {"hidden_states": (1, 4, 4096)}
