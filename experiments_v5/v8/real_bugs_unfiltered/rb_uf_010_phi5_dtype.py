"""
Unfiltered post-freeze repro #10 (out-of-fragment, dtype mismatch):
HF transformers - Phi5 quantization scale dtype.

GitHub PR  : https://github.com/huggingface/transformers/pull/45611  (merged 2026-04-23)
Repository : huggingface/transformers
Buggy file : src/transformers/models/phi5/modeling_phi5.py
            (pre-#45611, Phi5Attention forward)

Root cause: query and key projections were quantized to bf16 but the
attention scale was a Python float multiplied as fp32, producing a
runtime dtype-promotion `RuntimeError: expected scalar type BFloat16
but found Float`.  The shape arithmetic is correct; the bug is purely
on the dtype channel.

Out-of-fragment for TG (no dtype refinements in the v5 fragment).
Expected verdict: Abstain.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden_size=2048, num_heads=16, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, dtype=torch.bfloat16)
        # BUG (pre-#45611): attn_scale is fp32 but multiplied into bf16 q.
        self.attn_scale = torch.tensor(head_dim ** -0.5, dtype=torch.float32)

    def forward(self, x):
        # x: (B, S, hidden_size) bf16
        x = x.to(torch.bfloat16)
        q = self.q_proj(x)  # bf16
        # Buggy multiply: q (bf16) * scale (fp32) -- dtype mismatch
        q = q * self.attn_scale
        return q


INPUT_SHAPES = {"x": (1, 8, 2048)}
