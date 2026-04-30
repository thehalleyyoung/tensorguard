"""
Unfiltered post-freeze repro #11 (out-of-fragment, distributed-shape):
HF transformers - Mixtral2 expert sharding all_gather.

GitHub PR  : https://github.com/huggingface/transformers/pull/45624  (merged 2026-04-24)
Repository : huggingface/transformers
Buggy file : src/transformers/models/mixtral2/modeling_mixtral2.py
            (pre-#45624, Mixtral2ParallelExperts.forward)

Root cause: per-rank expert tensor was all-gathered with the
wrong dim, producing a (world_size * E_local) gather where
the downstream einsum expected (E_local * world_size). The
shape only manifests under torch.distributed.all_gather, which
crosses a process boundary.

Out-of-fragment (distributed): TG cannot reason about
torch.distributed primitives. Expected verdict: Abstain.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    """A reduced single-process stand-in. The real bug requires
    torch.distributed; the single-process repro below cannot
    reproduce the all_gather shape mismatch."""

    def __init__(self, hidden_size=512, num_experts_per_rank=4, world_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts_per_rank = num_experts_per_rank
        self.world_size = world_size
        # Local experts.
        self.expert_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # In the real bug, x is sharded across ranks and all_gathered.
        # Single-process, this is a vacuous identity over local hidden.
        return self.expert_proj(x)


INPUT_SHAPES = {"x": (1, 16, 512)}
