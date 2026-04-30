"""
Upstream-faithful real-bug repro: BLOOM attention head split view.

GitHub Issue: https://github.com/huggingface/transformers/issues/17474
Buggy file  : transformers/models/bloom/modeling_bloom.py
              (BloomAttention pre-fix; fused QKV split with wrong head count)

BLOOM uses a single fused QKV projection and then splits by chunk.
The bug: the view used (batch, seq, 3, num_heads, head_dim) instead of
(batch, seq, num_heads, 3, head_dim) — or equivalently, the split
along dim=-1 produces wrong per-head tensors when num_heads != 3.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden_size=1024, num_attention_heads=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        fused_qkv = self.query_key_value(hidden_states)
        # BUG: view splits into (3, num_heads, head_dim) not (num_heads, 3, head_dim)
        # 3 * hidden_size = 3072; the view says (bsz, seq, 3, num_heads, head_dim)
        # total = bsz*seq*3*16*64 but 3*1024 per token, so:
        # 3*16*64 == 3072 == 3*1024, but the downstream use is wrong because
        # we need per-head slices of q/k/v but instead get 3-wide slices
        # For the numeric check: view(bsz, seq, 3, num_heads+1, head_dim)
        wrong_heads = self.num_attention_heads + 1  # causes size mismatch
        fused_qkv = fused_qkv.view(bsz, seq_len, 3, wrong_heads, self.head_dim)
        return fused_qkv


INPUT_SHAPES = {"hidden_states": (1, 8, 1024)}
