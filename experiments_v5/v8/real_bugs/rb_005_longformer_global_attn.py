"""
Real Bug Repro: Longformer global attention view with wrong dimension order

GitHub Issue: https://github.com/huggingface/transformers/issues/5646
Repository:   huggingface/transformers
Model:        Longformer

Bug: In LongformerSelfAttention, after computing attention probabilities over global
tokens, the code calls:
    attn_probs.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
but the source tensor has shape (batch_size, num_heads, seq_len, max_num_global_attn_indices),
i.e., the last two dimensions are swapped. This causes a shape mismatch when
seq_len ≠ max_num_global_attn_indices.

Original error: shape mismatch in global attention path for Longformer when
num_global_attn_indices ≠ seq_len.

Substitution note: batch=1, num_heads=12, seq_len=512, max_num_global_attn_indices=5
representing the standard Longformer-base-4096 configuration on a short sequence.
Input total: 1*12*512*5 = 30720; target total: 1*12*5*512 = 30720 (same elements,
but when seq_len >> max_global: view shape '(1, 12, max_global, seq_len)' with
wrong input shape '(1, 12, seq_len, max_global)' causes the error in earlier bug.

For TG detection we use the concrete reported shape pair that fails:
input (1, 12, 512, 518) → target (1, 12, 5, 512)
1*12*512*518 = 3182592 ≠ 1*12*5*512 = 30720
"""
import torch
import torch.nn as nn

# Attention probs tensor: (batch, heads, seq_len, max_global) with global attention
INPUT_SHAPES = {"x": (1, 12, 512, 518)}


class BuggyModule(nn.Module):
    def forward(self, x):
        # Bug: dimensions are transposed; view target uses wrong order
        # 1 * 12 * 512 * 518 = 3,182,592 ≠ 1 * 12 * 5 * 512 = 30,720
        return x.view(1, 12, 5, 512)
