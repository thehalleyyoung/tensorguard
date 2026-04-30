"""
Upstream-faithful real-bug repro: RoBERTa dense projection after attention.

GitHub Issue: https://github.com/huggingface/transformers/issues/10203
Buggy file  : transformers/models/roberta/modeling_roberta.py
              (RobertaSelfOutput dense uses wrong size when hidden_size not divisible)

RoBERTa self-attention output uses a dense projection to map from
all_head_size back to hidden_size. Bug: all_head_size = num_heads*head_size
where head_size = hidden_size // num_heads, which is smaller than expected
when hidden_size is not divisible by num_heads.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=11):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads  # truncated
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # BUG: dense expects all_head_size (= 759 when 768//11=69) but hidden_size is 768
        # The projection dense is defined as (all_head_size, hidden_size) but the
        # downstream LayerNorm expects the residual (hidden_size) to match dense output
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        # BUG: dense takes all_head_size input, but residual connection adds to
        # hidden_states which has last dim hidden_size != all_head_size
        self.dense = nn.Linear(self.all_head_size, hidden_size)

    def transpose_for_scores(self, x, bsz, seq_len):
        new_shape = (bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        q = self.transpose_for_scores(self.query(hidden_states), bsz, seq_len)
        v = self.value(hidden_states).view(bsz, seq_len,
                                            self.num_attention_heads,
                                            self.attention_head_size)
        v = v.permute(0, 2, 1, 3)
        context = v.permute(0, 2, 1, 3).reshape(bsz, seq_len, self.all_head_size)
        projected = self.dense(context)  # (bsz, seq_len, hidden_size)
        # BUG: residual addition: projected (768) + hidden_states (768) is OK here,
        # but the view in query (all_head_size=759 != hidden_size=768) is the shape bug
        # The bug manifests when: hidden_states.view(bsz, seq_len, 11, 69) fails
        # because 768 != 759
        wrong_view = hidden_states.view(bsz, seq_len, self.num_attention_heads,
                                         self.attention_head_size)
        return wrong_view


INPUT_SHAPES = {"hidden_states": (2, 12, 768)}
