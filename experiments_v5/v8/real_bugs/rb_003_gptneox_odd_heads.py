"""
Real Bug Repro: GPT-NeoX odd attention heads view mismatch

GitHub Issue: https://github.com/huggingface/transformers/issues/23081
Repository:   huggingface/transformers
Model:        GPT-NeoX (EleutherAI)

Bug: When `hidden_size` is not divisible by `num_attention_heads`, the QKV linear
layer outputs `3 * hidden_size = 3072` features, but the subsequent view to split
into per-head queries uses `head_size = hidden_size // num_heads = 1024 // 12 = 85`
(integer division), so `num_heads * head_size = 12 * 85 = 1020 ≠ 1024`. The
per-head view target `view(batch, seq, num_heads, head_size * 3)` = `view(1, 5, 12, 255)`
has total `12 * 255 = 3060 ≠ 3072`.

Original error: users observed shape mismatch when configuring models with hidden_size
not divisible by num_heads (e.g., hidden_size=1024, num_heads=12).

Substitution note: uses hardcoded integers matching hidden_size=1024, num_heads=12,
batch=1, seq_len=5.
"""
import torch
import torch.nn as nn

# Input to the module: hidden_states (batch=1, seq=5, hidden_size=1024)
INPUT_SHAPES = {"x": (1, 5, 1024)}


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # QKV projection: outputs 3 * hidden_size = 3072 features
        self.qkv = nn.Linear(1024, 3072)

    def forward(self, x):
        # Bug: head_size = 1024 // 12 = 85 (integer division, loses remainder)
        # view expects num_heads * 3 * head_size = 12 * 255 = 3060 per token,
        # but qkv outputs 3072 per token.
        # 1 * 5 * 3072 = 15360 ≠ 1 * 5 * 12 * 255 = 15300
        qkv = self.qkv(x)
        return qkv.view(1, 5, 12, 255)
