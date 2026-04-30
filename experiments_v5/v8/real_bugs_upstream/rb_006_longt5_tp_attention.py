"""
Upstream-faithful real-bug repro: LongT5/MT5 attention shape under TP sharding.
GitHub PR : https://github.com/huggingface/transformers/pull/45109
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, d_model=512, d_kv=64, num_heads=8, tp_world_size=4):
        super().__init__()
        self.d_model = d_model
        self.d_kv = d_kv
        self.num_heads = num_heads
        sharded_inner = (num_heads * d_kv) // tp_world_size
        self.q = nn.Linear(d_model, sharded_inner, bias=False)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        q = self.q(hidden_states)
        # BUG: per-head reshape uses GLOBAL num_heads * d_kv layout.
        return q.view(batch_size, -1, self.num_heads, self.d_kv)


INPUT_SHAPES = {"hidden_states": (2, 10, 512)}
