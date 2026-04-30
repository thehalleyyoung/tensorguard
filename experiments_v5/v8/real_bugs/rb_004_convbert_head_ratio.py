"""
Real Bug Repro: ConvBERT wrong head_ratio causing view mismatch

GitHub Issue: https://github.com/huggingface/transformers/issues/21523
Repository:   huggingface/transformers
Model:        ConvBERT

Bug: ConvBERT uses a "mixed attention" head that combines full-attention heads and
conv-attention heads. With `head_ratio=4` and `hidden_size=768, num_heads=12`, the
mixed-attention portion should produce `768 / head_ratio = 192` features per token.
But a bug caused the view to use `768` instead of `192`, doubling the expected output.

The mixed attention output has `num_heads / head_ratio = 3` regular heads each of
`64` dimensions = 192 total features, but the code tried to view as if it were 384.

Substitution note: integers correspond to the default ConvBERT-base config:
hidden_size=768, num_attention_heads=12, head_ratio=4; batch=3, seq_len=10.
The buggy reshape target is (3, 10, 384) but the tensor only has 192 features/token.
3 * 10 * 384 = 11520 ≠ 3 * 10 * 192 = 5760 (the actual tensor size).
"""
import torch
import torch.nn as nn

# Input: mixed attention output with actual 192 features per position
INPUT_SHAPES = {"x": (3, 10, 192)}


class BuggyModule(nn.Module):
    def forward(self, x):
        # Bug: view uses 384 (= 768/2) but should use 192 (= 768/head_ratio = 768/4)
        # 3 * 10 * 192 = 5760 ≠ 3 * 10 * 384 = 11520
        return x.view(3, 10, 384)
