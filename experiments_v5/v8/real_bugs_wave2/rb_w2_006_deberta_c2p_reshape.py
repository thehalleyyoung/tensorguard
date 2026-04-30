"""
Upstream-faithful real-bug repro: DeBERTa disentangled attention wrong reshape.

GitHub Issue: https://github.com/huggingface/transformers/issues/13934
Buggy file  : transformers/models/deberta/modeling_deberta.py
              (DisentangledSelfAttention c2p attention content-to-position reshape)

DeBERTa uses disentangled attention with content-to-position scores.
The c2p score matrix should be (bsz, num_heads, seq_len, span_len)
but is reshaped from (bsz*num_heads, seq_len, -1) incorrectly when
bsz != 1 and the wrong split axis is used.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, position_buckets=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.pos_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        # Content projection
        q = self.query_proj(hidden_states)  # (bsz, seq, hidden)
        # Position projection — operates on relative positions
        # For simplicity simulate (seq, seq, hidden) attention scores:
        # BUG: reshape (bsz, seq, num_heads, head_size) -> (bsz*num_heads, seq, head_size)
        # then try to reshape back as (bsz, num_heads, seq, seq) when seq != head_size
        q_heads = q.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        q_heads = q_heads.permute(0, 2, 1, 3)  # (bsz, num_heads, seq, head_size)
        # BUG: flatten bsz and num_heads then reshape to (bsz, num_heads, seq, seq)
        # 64 * seq * head_size != bsz * num_heads * seq * seq unless head_size==seq
        q_flat = q_heads.reshape(bsz * self.num_attention_heads, seq_len, self.attention_head_size)
        # This view fails when seq_len != attention_head_size (seq=20, head_size=64)
        q_wrong = q_flat.view(bsz, self.num_attention_heads, seq_len, seq_len)
        return q_wrong


INPUT_SHAPES = {"hidden_states": (2, 20, 768)}
