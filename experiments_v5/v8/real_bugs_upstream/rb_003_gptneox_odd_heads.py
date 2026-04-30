"""
Upstream-faithful real-bug repro: GPT-NeoX QKV view when hidden_size %
num_attention_heads != 0.

GitHub Issue: https://github.com/huggingface/transformers/issues/23081
Buggy file  : transformers/models/gpt_neox/modeling_gpt_neox.py
              (GPTNeoXAttention pre-fix)

Real `nn.Linear(hidden_size, 3*hidden_size)` QKV projection followed by
the per-head split via `qkv.view(batch, seq, num_heads, 3*head_size)`,
where `head_size` is `hidden_size // num_heads` (silent integer trunc).
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden_size=1024, num_attention_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, hidden_states):
        qkv = self.query_key_value(hidden_states)
        # BUG: 3 * (hidden_size // num_heads) != 3 * hidden_size when not divisible
        new_qkv_shape = qkv.size()[:-1] + (
            self.num_attention_heads,
            3 * self.head_size,
        )
        return qkv.view(*new_qkv_shape)


INPUT_SHAPES = {"hidden_states": (1, 5, 1024)}
