"""
Real Bug Repro: PEFT PrefixTuning flan-t5-small view mismatch

GitHub Issue: https://github.com/huggingface/peft/issues/385
Repository:   huggingface/peft (Hugging Face PEFT library)
Model:        google/flan-t5-small with PrefixTuning

Bug: In PEFT PrefixTuning's `get_prompt`, `past_key_values` is reshaped as:
    past_key_values.view(batch_size, num_virtual_tokens, num_layers * 2,
                         num_attention_heads, attention_head_size)
For flan-t5-small (d_model=512, num_heads=6):
    attention_head_size = d_model // num_heads = 512 // 6 = 85 (integer division)
    num_virtual_tokens=8, num_layers=8, batch_size=8.

The total elements in the view target:
    8 * 8 * 16 * 6 * 85 = 522,240
But the source tensor has:
    8 * 8 * 8192 = 524,288 elements (with default prefix model hidden size).

Original error:
  RuntimeError: shape '[8, 8, 16, 6, 85]' is invalid for input of size 524288
  8*8*16*6*85 = 522,240 ≠ 524,288

Root cause: d_model (512) is not divisible by num_heads (6), so integer division
silently loses remainder, making the reshape incompatible.

Substitution note: uses exact integers from the original issue error trace.
batch_size=8, num_virtual_tokens=8, num_layers*2=16, num_heads=6, head_dim=85.
Source tensor shape derived from 524288 / (8 * 8) = 8192 per (batch, token).
"""
import torch
import torch.nn as nn

# past_key_values before reshape: (batch=8, num_virtual_tokens=8, total_kv_dim=8192)
INPUT_SHAPES = {"x": (8, 8, 8192)}


class BuggyModule(nn.Module):
    def forward(self, x):
        # Bug: head_dim = 512 // 6 = 85, but 8*8*16*6*85 = 522,240 ≠ 524,288
        # The integer division truncates the fractional part of d_model/num_heads
        return x.view(8, 8, 16, 6, 85)
