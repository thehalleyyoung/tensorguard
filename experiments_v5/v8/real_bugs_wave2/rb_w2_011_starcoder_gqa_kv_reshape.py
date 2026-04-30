"""
Upstream-faithful real-bug repro: StarCoder GQA key/value head expand.

GitHub Issue: https://github.com/huggingface/transformers/issues/28174
Buggy file  : transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
              (GPTBigCodeAttention multi-query kv_cache view mismatch)

StarCoder/GPT-BigCode uses multi-query attention (1 KV head, N query heads).
Bug: when reshaping kv states, the expand factor uses query heads count
instead of 1 (kv heads), causing a size mismatch in the combined attention.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, hidden_size=6144, num_attention_heads=48, num_kv_heads=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_attention_heads  # 128
        # Multi-query: Q has num_heads heads, K/V have num_kv_heads=1
        self.c_attn = nn.Linear(hidden_size,
                                 (num_attention_heads + 2 * num_kv_heads) * self.head_dim)

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        combined = self.c_attn(hidden_states)
        # Split into Q (num_heads*head_dim), K (num_kv_heads*head_dim), V (num_kv_heads*head_dim)
        qkv_split = (self.num_attention_heads * self.head_dim,
                      self.num_kv_heads * self.head_dim,
                      self.num_kv_heads * self.head_dim)
        q, k, v = combined.split(qkv_split, dim=-1)
        # Reshape q: (bsz, seq, num_heads, head_dim) -> OK
        q = q.view(bsz, seq_len, self.num_attention_heads, self.head_dim)
        # BUG: reshape k using num_attention_heads instead of num_kv_heads
        # k has (num_kv_heads=1) * head_dim features but view uses num_attn_heads=48
        wrong_kv_heads = self.num_attention_heads  # 48 instead of 1
        k = k.view(bsz, seq_len, wrong_kv_heads, self.head_dim)
        return q, k


INPUT_SHAPES = {"hidden_states": (1, 8, 6144)}
