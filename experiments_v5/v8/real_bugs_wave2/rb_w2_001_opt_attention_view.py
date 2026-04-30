"""
Upstream-faithful real-bug repro: OPT self-attention QKV view mismatch.

GitHub Issue: https://github.com/huggingface/transformers/issues/19927
Buggy file  : transformers/models/opt/modeling_opt.py
              (OPTAttention pre-fix; bsz-based view uses seq_len not bsz)

Real OPT attention that projects Q/K/V then splits heads via view.
The bug: query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
where the inner reshape factors are swapped vs what was passed.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, hidden_states):
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        # BUG: uses embed_dim where it should use head_dim = embed_dim//num_heads
        # view(bsz, tgt_len, num_heads, embed_dim) has wrong last dim
        wrong_dim = self.embed_dim  # should be self.head_dim
        query_states = query_states.view(bsz, tgt_len, self.num_heads, wrong_dim)
        return query_states.transpose(1, 2)


INPUT_SHAPES = {"hidden_states": (2, 10, 512)}
