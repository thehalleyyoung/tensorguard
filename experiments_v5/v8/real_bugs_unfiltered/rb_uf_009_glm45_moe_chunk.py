"""
Unfiltered post-freeze repro #9 (in-fragment but constructor-bound config):
HF transformers - Glm45MoE expert routing chunk size.

GitHub PR  : https://github.com/huggingface/transformers/pull/45597  (merged 2026-04-22)
Repository : huggingface/transformers
Buggy file : src/transformers/models/glm45_moe/modeling_glm45_moe.py
            (pre-#45597, GLM45MoEFeedForward.forward)

Root cause: per-expert chunk sums to (num_experts * chunk_size)
but config drops chunk_size from the divisor; the gating Linear
expects the original hidden dim and gets a chunked one.

The buggy view target depends on self.config.num_local_experts
and self.config.expert_chunk_size, both constructor-bound
integer attributes -- this is exactly the silent-miss class
flagged in Section "Limitations" of the paper.

Expected verdict: silent verified (the constructor-bound
integer envelope synthesiser does not propagate
self.config.expert_chunk_size into the divisibility predicate).
"""
import torch
import torch.nn as nn


class _Cfg:
    def __init__(self):
        self.hidden_size = 768
        self.num_local_experts = 8
        self.expert_chunk_size = 5  # bug: not a divisor of hidden_size


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg()
        self.gate = nn.Linear(self.config.hidden_size, self.config.num_local_experts)
        self.proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)

    def forward(self, x):
        # x: (B, S, hidden_size).
        # BUG (pre-#45597): chunked into expert_chunk_size pieces, but
        # hidden_size % expert_chunk_size != 0; .view() fails.
        B, S, H = x.shape
        chunked = x.view(B, S, self.config.expert_chunk_size,
                         H // self.config.expert_chunk_size)
        return self.proj(chunked.reshape(B, S, H))


INPUT_SHAPES = {"x": (1, 16, 768)}
