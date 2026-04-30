"""
Real Bug Repro: EleutherAI GPT-NeoX GQA reshape with non-integer KV head dim

GitHub Issue: https://github.com/EleutherAI/gpt-neox/issues/1314
Fixed in PR:  https://github.com/EleutherAI/gpt-neox/pull/1315 (commit 96c242eb)
Repository:   EleutherAI/gpt-neox

Bug: In `gqa_project`, the old code computed a "fake head dim" for the combined QKV
tensor and reshaped it as:
    new_qkv_shape = (sq, b, np, int(hn * (1 + 2 * kvp/np)))
where:
    - np = num_attention_heads_per_partition
    - kvp = num_kv_heads_per_partition
    - hn = hidden_size_per_attention_head

When kvp/np is not a simple fraction (e.g., np=5, kvp=1 under TP=8 with
40 total heads and 8 KV heads), `hn * (1 + 2 * 1/5) = 128 * 1.4 = 179.2`,
which truncates to 179 via `int()`. The reshape target has 5 * 179 = 895 features
per (sq, b) position, but the actual tensor has (5 + 2*1) * 128 = 896 features.

Original error:
  RuntimeError: shape '[4096, 1, 5, 179]' is invalid for input of size 3670016
  4096 * 1 * 5 * 179 = 3,665,920 ≠ 3,670,016

Config: hidden_size=5120, num_attention_heads=40, num_kv_heads=8, TP=8.
Per-partition: np=5, kvp=1, hn=128.

Substitution note: uses exact integers from the issue error message.
"""
import torch
import torch.nn as nn

# Combined QKV tensor after projection: [sq, b, (np + 2*kvp) * hn]
# = (4096, 1, (5 + 2*1) * 128) = (4096, 1, 896)
INPUT_SHAPES = {"x": (4096, 1, 896)}


class BuggyModule(nn.Module):
    def forward(self, x):
        # Bug: int(128 * (1 + 2 * (1/5))) = int(128 * 1.4) = int(179.2) = 179
        # 4096 * 1 * 5 * 179 = 3,665,920 ≠ 4096 * 1 * 896 = 3,670,016
        return x.view(4096, 1, 5, 179)
