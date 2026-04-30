"""Tests for v5 QKV unpacking and einops rearrange."""
from __future__ import annotations

import pytest

from src.tensor_shapes import ShapeDim, TensorShape
from src.v5.symbolic_config import SymInt
from src.v5.qkv_unpacking import (
    unpack_split, unpack_chunk, unpack_unbind,
    parse_einops_rearrange, split_qkv,
)


def _ts(*dims):
    return TensorShape(tuple(ShapeDim(d) for d in dims))


# ── split ────────────────────────────────────────────────────────────────

def test_split_int_concrete_even():
    s = _ts(2, 6)
    r = unpack_split(s, 2, dim=-1)
    assert r.ok and len(r.shapes) == 3
    assert all(t.dims[-1].value == 2 for t in r.shapes)


def test_split_int_concrete_uneven():
    s = _ts(7)
    r = unpack_split(s, 3, dim=0)
    assert r.ok
    assert [t.dims[0].value for t in r.shapes] == [3, 3, 1]


def test_split_list_sizes():
    s = _ts(10)
    r = unpack_split(s, [2, 3, 5], dim=0)
    assert r.ok and [t.dims[0].value for t in r.shapes] == [2, 3, 5]


def test_split_list_sizes_mismatch():
    s = _ts(10)
    r = unpack_split(s, [2, 3, 4], dim=0)
    assert not r.ok


def test_split_symbolic_dim_abstains():
    s = _ts("B", "S")
    r = unpack_split(s, 2, dim=0)
    assert not r.ok


# ── chunk ────────────────────────────────────────────────────────────────

def test_chunk_concrete_divisible():
    s = _ts(2, 12)
    r = unpack_chunk(s, 3, dim=-1)
    assert r.ok and len(r.shapes) == 3
    assert [t.dims[-1].value for t in r.shapes] == [4, 4, 4]


def test_chunk_concrete_uneven():
    s = _ts(2, 11)
    r = unpack_chunk(s, 3, dim=-1)
    # PyTorch: chunk_size=ceil(11/3)=4 → [4,4,3]
    assert [t.dims[-1].value for t in r.shapes] == [4, 4, 3]


def test_chunk_symbolic():
    s = _ts("B", "D")
    r = unpack_chunk(s, 3, dim=-1)
    assert r.ok and len(r.shapes) == 3
    assert "//3" in r.shapes[0].dims[-1].value


# ── unbind ───────────────────────────────────────────────────────────────

def test_unbind_concrete():
    s = _ts(2, 4, 3, 16, 64)  # B, T, 3, H, D
    r = unpack_unbind(s, dim=2)
    assert r.ok and len(r.shapes) == 3
    assert r.shapes[0].dims == (ShapeDim(2), ShapeDim(4), ShapeDim(16), ShapeDim(64))


def test_unbind_symbolic_abstains():
    s = _ts(2, "K", 4)
    r = unpack_unbind(s, dim=1)
    assert not r.ok


# ── einops.rearrange ────────────────────────────────────────────────────

def test_rearrange_split_three_b_h_t_d():
    # The classic QKV pattern.
    inp = _ts(2, 4, 3 * 16 * 64)  # b, t, three*h*d
    r = parse_einops_rearrange(
        "b t (three h d) -> three b h t d",
        inp, axes_lengths={"three": 3, "h": 16},
    )
    assert r.ok
    assert len(r.shapes) == 3
    out = r.shapes[0]
    assert out.dims == (ShapeDim(2), ShapeDim(16), ShapeDim(4), ShapeDim(64))


def test_rearrange_with_symbolic_batch():
    inp = _ts("B", 4, 3 * 16 * 64)
    r = parse_einops_rearrange(
        "b t (three h d) -> three b h t d",
        inp, axes_lengths={"three": 3, "h": 16},
    )
    assert r.ok and len(r.shapes) == 3
    assert r.shapes[0].dims[0].value == "B"


def test_rearrange_mismatch_caught():
    inp = _ts(2, 4, 100)  # 100 not divisible by 3
    r = parse_einops_rearrange(
        "b t (three h d) -> three b h t d",
        inp, axes_lengths={"three": 3, "h": 5},
    )
    assert not r.ok


# ── split_qkv shortcut ─────────────────────────────────────────────────

def test_split_qkv_concrete():
    s = _ts(2, 4, 3 * 12 * 64)
    r = split_qkv(s, num_heads=12, head_dim=64)
    assert r.ok
    q, k, v = r.shapes
    assert q.dims == (ShapeDim(2), ShapeDim(4), ShapeDim(12), ShapeDim(64))


def test_split_qkv_size_mismatch():
    s = _ts(2, 4, 100)
    r = split_qkv(s, num_heads=12, head_dim=64)
    assert not r.ok


def test_split_qkv_with_symints():
    h = SymInt("h"); d = SymInt("d")
    s = _ts("B", "T", 3 * 12 * 64)  # concrete fused dim ok
    r = split_qkv(s, num_heads=h, head_dim=d)
    assert r.ok
    assert r.shapes[0].dims[-2].value == "h"
    assert r.shapes[0].dims[-1].value == "d"


# ── HF integration if available ────────────────────────────────────────

def test_hf_qkv_pattern_if_importable():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.BertConfig()
    H = cfg.num_attention_heads
    D = cfg.hidden_size // H
    # Mimic BertSelfAttention: x @ Wqkv → (B, T, 3*H*D), then split
    fused = _ts(2, 8, 3 * H * D)
    r = split_qkv(fused, num_heads=H, head_dim=D)
    assert r.ok
    assert r.shapes[0].dims[-2].value == H
    assert r.shapes[0].dims[-1].value == D
