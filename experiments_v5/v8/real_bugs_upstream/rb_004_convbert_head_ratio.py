"""
Upstream-faithful real-bug repro: ConvBERT mixed-attention head_ratio view.
GitHub Issue: https://github.com/huggingface/transformers/issues/21523
Buggy file  : transformers/models/convbert/modeling_convbert.py
              (ConvBertSelfAttention pre-fix)
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, head_ratio=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_ratio = head_ratio

    def forward(self, mixed_query_layer):
        batch, seq, _ = mixed_query_layer.shape
        # BUG (pre-fix): hard-coded denominator 2 instead of self.head_ratio.
        wrong_features = self.hidden_size // 2
        return mixed_query_layer.view(batch, seq, wrong_features)


INPUT_SHAPES = {"mixed_query_layer": (3, 10, 768 // 4)}
