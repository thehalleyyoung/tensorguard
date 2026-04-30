"""
Post-freeze upstream-faithful real-bug repro: HF transformers RouterParallel score/index pairing
GitHub PR  : https://github.com/huggingface/transformers/pull/45473  (merged 2026-04-16)
Repository : huggingface/transformers
Buggy file : src/transformers/integrations/tensor_parallel.py
            (pre-#45473, RouterParallel._prepare_output_fn)

Root cause (per PR body): the router produces three element-wise
paired tensors -- ``router_logits (seq, num_experts)``,
``router_scores (seq, top_k)``, ``router_indices (seq, top_k)`` --
and ``_prepare_output_fn`` is supposed to remap the last two to
local-expert indices. The pre-#45473 code reshaped them to the
*global* num_experts dimension instead of top_k, producing a shape
mismatch when the next op (``grouped_mm``) tries to combine them.

This repro mirrors the buggy reshape: scores are passed to a final
matmul that expects the (seq, top_k) shape, but the buggy path
expanded them to (seq, num_experts).
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden_size=512, num_experts=8, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts)
        # The downstream projection expects top_k experts per token.
        self.expert_out = nn.Linear(top_k, hidden_size)

    def forward(self, x):
        # x : (seq, hidden_size)
        logits = self.router(x)                                # (seq, num_experts)
        # BUG (pre-#45473): scores reshaped to num_experts instead of
        # top_k.  Real upstream uses topk; the buggy path takes the
        # softmax over the full expert axis and feeds it forward.
        scores = torch.softmax(logits, dim=-1)                 # (seq, num_experts)
        # The expert_out layer expects (seq, top_k) but receives
        # (seq, num_experts); Linear weight mismatch.
        return self.expert_out(scores)


INPUT_SHAPES = {"x": (16, 512)}
