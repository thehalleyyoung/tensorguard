"""
Post-freeze upstream-faithful real-bug repro: HF peft 3-dim MoE LoRA in/out_features swap
GitHub PR  : https://github.com/huggingface/peft/pull/3165  (merged 2026-04-15)
Repository : huggingface/peft
Buggy file : src/peft/tuners/lora/layer.py (pre-#3165, is_transposed reversal)

Root cause (per PR body): for 3-dim MoE parameters, the LoRA layer
checked ``is_transposed`` the wrong way round when picking which
dimension is ``in_features`` vs ``out_features``. Initialisation used
the swapped shape and the forward step used the swapped shape too, so
the two errors cancelled in greenfield training. They do *not* cancel
when loading a checkpoint trained with the correct shape: the
``base_layer.weight`` then has the correct (swapped-relative-to-LoRA)
shape, and ``lora_B @ lora_A @ x`` mismatches against the base
projection.

This repro mirrors a single MoE expert: a base 3-D weight stored as
``[num_experts, out_features, in_features]`` (correct upstream shape),
plus LoRA A/B initialised with the *swapped* in/out (buggy) and a
forward path that adds them.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, num_experts=8, in_features=2048, out_features=512, r=16):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        # Base weight is in correct upstream shape.
        self.base_weight = nn.Parameter(torch.zeros(num_experts, out_features, in_features))
        # BUG (pre-#3165): LoRA A and B were initialised with swapped
        # in/out because is_transposed was negated. With the correct
        # base, the LoRA delta has shape [num_experts, in_features,
        # out_features] which cannot be added to the base.
        self.lora_A = nn.Parameter(torch.zeros(num_experts, r, out_features))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, in_features, r))

    def forward(self, x):
        # x : (B, num_experts, in_features)
        base_out  = torch.einsum("bei,eoi->beo", x, self.base_weight)             # (B, E, out)
        # Buggy LoRA update: dimensions are swapped, so this is (B, E, in)
        lora_delta = torch.einsum("eir,erj->eij", self.lora_B, self.lora_A)        # (E, in, out) intended; actually (E, in, out)?
        # The buggy code path adds lora_delta as if it had base shape:
        # ``base_weight + lora_delta`` -> shape mismatch (out,in) vs (in,out)
        return base_out + (self.base_weight + lora_delta).sum(dim=-1).unsqueeze(0).expand(x.shape[0], -1, -1)


INPUT_SHAPES = {"x": (2, 8, 2048)}
