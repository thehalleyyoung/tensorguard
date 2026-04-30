"""Tests for the config-attribute / QKV-split upgrade in the static shape
verifier."""

import pytest

from src.api import verify_architecture


def test_nanogpt_attention_config():
    src = '''
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1))
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
'''
    res = verify_architecture(src, input_shapes={"x": ("B", "T", "n_embd_sym")})
    # No error-level shape bugs (symbolic propagation should keep it sound).
    assert all(
        "shape" not in (b.message or "").lower() or b.severity != "error"
        for b in res.bugs
    ), [(b.severity, b.message) for b in res.bugs]


def test_vit_mha_block():
    src = '''
import torch.nn as nn
class MHABlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
'''
    res = verify_architecture(src, input_shapes={"x": (2, 197, 768)})
    assert not any(b.severity == "error" for b in res.bugs), \
        [(b.severity, b.message) for b in res.bugs]


def test_bert_self_attention_style():
    src = '''
import torch.nn as nn
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = q @ k.transpose(-2, -1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return out
'''
    res = verify_architecture(src, input_shapes={"x": ("B", "T", "C")})
    assert not any(b.severity == "error" for b in res.bugs), \
        [(b.severity, b.message) for b in res.bugs]


def test_nanogpt_split_axis_bug_detected():
    """Wrong split axis (dim=1 instead of dim=2) must not be silently verified.
    Either an error is reported or analysis abstains (we expect detection).
    """
    src = '''
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1))
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
'''
    res = verify_architecture(src, input_shapes={"x": (2, 16, 768)})
    has_error = any(b.severity == "error" for b in res.bugs)
    assert has_error, "Bug should be detected (or at least flagged)"
