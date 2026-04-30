"""
Upstream-faithful real-bug repro: GPT-2 attention head split.

GitHub Issue: https://github.com/huggingface/transformers/issues/12048
Buggy file  : transformers/models/gpt2/modeling_gpt2.py
              (GPT2Attention split_heads when n_head does not divide evenly)

GPT-2 attention splits a (batch, seq, 3*n_embd) QKV via view then split.
Bug: uses n_head count that causes uneven division of hidden features.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, n_embd=768, n_head=13):  # 13 does not divide 768
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        # BUG: 768 // 13 = 59, remainder 1; tensor total doesn't divide evenly
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_size = self.n_embd // self.n_head  # = 59 (truncated!)
        # view(bsz, seq, 13, 59) needs bsz*seq*767 but we have bsz*seq*768
        q = self._split_heads(q, self.n_head, head_size)
        return q


INPUT_SHAPES = {"hidden_states": (1, 6, 768)}
