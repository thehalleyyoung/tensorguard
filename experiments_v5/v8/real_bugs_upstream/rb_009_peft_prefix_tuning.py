"""
Upstream-faithful real-bug repro: PEFT PrefixTuning view for flan-t5-small.
GitHub Issue: https://github.com/huggingface/peft/issues/385
Buggy file  : peft/peft_model.py + peft/tuners/prefix_tuning/model.py
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, d_model=512, num_attention_heads=6,
                 num_layers=8, num_virtual_tokens=8):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.num_layers_x2 = num_layers * 2
        self.num_virtual_tokens = num_virtual_tokens
        self.head_dim = d_model // num_attention_heads  # silent trunc

    def forward(self, past_key_values):
        batch_size = past_key_values.shape[0]
        # BUG (pre-fix): trailing prod = num_layers*2 * num_heads * head_dim
        return past_key_values.view(
            batch_size,
            self.num_virtual_tokens,
            self.num_layers_x2,
            self.num_attention_heads,
            self.head_dim,
        )


INPUT_SHAPES = {"past_key_values": (8, 8, 8 * 2 * 512)}
