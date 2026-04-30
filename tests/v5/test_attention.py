"""Tests for v5 attention/normalisation transfer rules."""
from __future__ import annotations

import pytest

from src.tensor_shapes import ShapeDim, TensorShape, TORCH_SHAPE_OPS
from src.v5.attention_norms import (
    sdpa_shape, mha_shape, layer_norm_shape, rms_norm_shape,
)


def _ts(*dims):
    return TensorShape(tuple(ShapeDim(d) for d in dims))


# ── SDPA ──────────────────────────────────────────────────────────────

def test_sdpa_basic():
    q = _ts(2, 8, 16, 64)   # (B, H, L, E)
    k = _ts(2, 8, 16, 64)
    v = _ts(2, 8, 16, 64)
    out = sdpa_shape(q, k, v)
    assert out == _ts(2, 8, 16, 64)


def test_sdpa_different_kv_embedding_v():
    q = _ts(2, 8, 16, 64)
    k = _ts(2, 8, 16, 64)
    v = _ts(2, 8, 16, 32)
    out = sdpa_shape(q, k, v)
    assert out == _ts(2, 8, 16, 32)


def test_sdpa_eq_mismatch_returns_none():
    q = _ts(2, 8, 16, 64)
    k = _ts(2, 8, 16, 32)  # E_k != E_q
    v = _ts(2, 8, 16, 32)
    assert sdpa_shape(q, k, v) is None


def test_sdpa_kv_seq_mismatch():
    q = _ts(2, 8, 16, 64)
    k = _ts(2, 8, 12, 64)
    v = _ts(2, 8, 16, 64)  # S mismatch
    assert sdpa_shape(q, k, v) is None


def test_sdpa_symbolic_dims():
    q = _ts("B", "H", "L", "D")
    k = _ts("B", "H", "S", "D")
    v = _ts("B", "H", "S", "D")
    out = sdpa_shape(q, k, v)
    assert out is not None
    assert out.dims[-1].value == "D"


# ── MultiheadAttention ────────────────────────────────────────────────

def test_mha_seq_first():
    q = _ts(10, 4, 64); k = _ts(10, 4, 64); v = _ts(10, 4, 64)
    out = mha_shape(q, k, v, embed_dim=64, num_heads=8)
    assert out == q


def test_mha_batch_first():
    q = _ts(4, 10, 64); k = _ts(4, 10, 64); v = _ts(4, 10, 64)
    out = mha_shape(q, k, v, embed_dim=64, num_heads=8, batch_first=True)
    assert out == q


def test_mha_embed_dim_indivisible():
    q = _ts(10, 4, 65); k = q; v = q
    assert mha_shape(q, k, v, embed_dim=65, num_heads=8) is None


# ── LayerNorm / F.layer_norm ─────────────────────────────────────────

def test_layer_norm_trailing_match():
    inp = _ts(2, 4, 64)
    out = layer_norm_shape(inp, [64])
    assert out == inp


def test_layer_norm_two_trailing_match():
    inp = _ts(2, 4, 32, 64)
    out = layer_norm_shape(inp, [32, 64])
    assert out == inp


def test_layer_norm_mismatch():
    inp = _ts(2, 4, 64)
    assert layer_norm_shape(inp, [128]) is None


def test_layer_norm_too_many_dims_in_norm_shape():
    inp = _ts(64,)
    assert layer_norm_shape(inp, [16, 64]) is None


def test_layer_norm_symbolic_passes():
    inp = _ts(2, 4, "D")
    out = layer_norm_shape(inp, ["D"])
    assert out is not None


# ── RMSNorm ──────────────────────────────────────────────────────────

def test_rms_norm_same_as_layer_norm():
    inp = _ts(2, 4, 64)
    assert rms_norm_shape(inp, [64]) == layer_norm_shape(inp, [64])


# ── Dispatch-table side-effect ───────────────────────────────────────

def test_dispatch_registration_present():
    for name in ("scaled_dot_product_attention", "layer_norm",
                 "rms_norm", "LayerNorm", "RMSNorm"):
        assert name in TORCH_SHAPE_OPS


# ── Real-world integration ───────────────────────────────────────────

def test_real_torch_layernorm_if_importable():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    ln = nn.LayerNorm([64])
    x = torch.randn(2, 4, 64)
    y = ln(x)
    inp = _ts(*x.shape)
    out = layer_norm_shape(inp, [64])
    assert tuple(d.value for d in out.dims) == tuple(y.shape)


def test_real_sdpa_if_available():
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    if not hasattr(F, "scaled_dot_product_attention"):
        pytest.skip("SDPA not available")
    q = torch.randn(2, 8, 16, 64)
    k = torch.randn(2, 8, 16, 64)
    v = torch.randn(2, 8, 16, 64)
    y = F.scaled_dot_product_attention(q, k, v)
    out = sdpa_shape(_ts(*q.shape), _ts(*k.shape), _ts(*v.shape))
    assert tuple(d.value for d in out.dims) == tuple(y.shape)
