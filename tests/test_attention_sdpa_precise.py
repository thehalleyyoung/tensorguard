"""Step 27 -- attention / transformer building blocks (SDPA, MHA).

Before Step 27 the modern functional core of attention,
``F.scaled_dot_product_attention``, was unmodelled: it fell through to the no-op
ACTIVATION fallback, so its output shape (query shape with the last dim replaced
by the value's last dim) was wrong and Q/K/V shape mismatches went undetected.
(``nn.MultiheadAttention`` the *layer* was already handled, with an
``embed_dim % num_heads`` check.)

Step 27 adds ``OpKind.SDPA`` and ``compute_sdpa_shape`` (differential-tested vs
``torch.scaled_dot_product_attention``), wires it into both engine paths, and
captures the q/k/v operands. Soundness posture: a violation is emitted only for
a *provable* concrete mismatch — query/key embed dim (the last dim) or key/value
sequence length (the second-to-last dim). Leading batch/head dims are not
checked (SDPA broadcasts them and grouped-query attention deliberately allows
unequal head counts), so GQA stays free of false positives.

These tests prove the behaviour end-to-end, including a realistic from-scratch
multi-head attention block (Linear + reshape + transpose + SDPA + reshape) and a
real timm Vision Transformer whose trace contains a dozen SDPA nodes.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tensor_shapes import ShapeDim, TensorShape, compute_sdpa_shape
from src.fx_extractor import verify_module

HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


def S(*ds):
    return TensorShape(tuple(ShapeDim(d) for d in ds))


def _dims(shape):
    return tuple(d.value for d in shape.dims)


def _violation_kinds(result):
    if result.counterexample is None:
        return []
    return [v.kind for v in result.counterexample.violations]


def _is_unsafe(result):
    return (not result.safe) and "shape_incompatible" in _violation_kinds(result)


def _verify(module, **shapes):
    return verify_module(module, input_shapes=shapes)


class _SDPA(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


# ---------------------------------------------------------------------------
# Unit: compute_sdpa_shape
# ---------------------------------------------------------------------------

def test_sdpa_output_shape_4d():
    out, err = compute_sdpa_shape(S(2, 4, 5, 8), S(2, 4, 7, 8), S(2, 4, 7, 9))
    assert err is None
    assert _dims(out) == (2, 4, 5, 9)


def test_sdpa_output_shape_3d():
    out, err = compute_sdpa_shape(S(2, 5, 8), S(2, 7, 8), S(2, 7, 9))
    assert err is None
    assert _dims(out) == (2, 5, 9)


def test_sdpa_embed_dim_mismatch_error():
    _out, err = compute_sdpa_shape(S(2, 4, 5, 8), S(2, 4, 7, 6), S(2, 4, 7, 9))
    assert err is not None


def test_sdpa_seq_len_mismatch_error():
    _out, err = compute_sdpa_shape(S(2, 4, 5, 8), S(2, 4, 7, 8), S(2, 4, 6, 9))
    assert err is not None


def test_sdpa_symbolic_abstains():
    # symbolic embed dim on key -> no error, output takes value's last dim.
    out, err = compute_sdpa_shape(S("B", 4, "L", 8), S("B", 4, "Sq", "E"), S("B", 4, "Sq", 9))
    assert err is None
    assert _dims(out) == ("B", 4, "L", 9)


def test_sdpa_gqa_unequal_heads_no_error():
    # Grouped-query attention: query has more heads than key/value. Leading
    # dims are not checked, so this must not be flagged.
    out, err = compute_sdpa_shape(S(2, 8, 5, 16), S(2, 2, 7, 16), S(2, 2, 7, 16))
    assert err is None
    assert _dims(out) == (2, 8, 5, 16)


def test_sdpa_differential_vs_torch():
    if not HAS_SDPA:
        return
    random.seed(7)
    for _ in range(1000):
        B = random.randint(1, 3)
        H = random.randint(1, 4)
        L = random.randint(1, 6)
        Sq = random.randint(1, 6)
        E = random.randint(1, 8)
        Ev = random.randint(1, 8)
        if random.random() < 0.5:
            q, k, v = S(B, H, L, E), S(B, H, Sq, E), S(B, H, Sq, Ev)
            tq, tk, tv = torch.randn(B, H, L, E), torch.randn(B, H, Sq, E), torch.randn(B, H, Sq, Ev)
        else:
            q, k, v = S(B, L, E), S(B, Sq, E), S(B, Sq, Ev)
            tq, tk, tv = torch.randn(B, L, E), torch.randn(B, Sq, E), torch.randn(B, Sq, Ev)
        out, err = compute_sdpa_shape(q, k, v)
        real = tuple(F.scaled_dot_product_attention(tq, tk, tv).shape)
        assert err is None
        assert _dims(out) == real


# ---------------------------------------------------------------------------
# End-to-end: verify_module
# ---------------------------------------------------------------------------

def test_e2e_sdpa_good_safe():
    if not HAS_SDPA:
        return
    assert _verify(_SDPA(), q=(2, 4, 5, 8), k=(2, 4, 7, 8), v=(2, 4, 7, 9)).safe


def test_e2e_sdpa_embed_mismatch_unsafe():
    if not HAS_SDPA:
        return
    assert _is_unsafe(_verify(_SDPA(), q=(2, 4, 5, 8), k=(2, 4, 7, 6), v=(2, 4, 7, 9)))


def test_e2e_sdpa_seq_mismatch_unsafe():
    if not HAS_SDPA:
        return
    assert _is_unsafe(_verify(_SDPA(), q=(2, 4, 5, 8), k=(2, 4, 7, 8), v=(2, 4, 6, 9)))


def test_e2e_sdpa_output_flows_downstream():
    if not HAS_SDPA:
        return

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(9, 16)

        def forward(self, q, k, v):
            return self.w(F.scaled_dot_product_attention(q, k, v))  # last dim 9

    assert _verify(M(), q=(2, 4, 5, 8), k=(2, 4, 7, 8), v=(2, 4, 7, 9)).safe

    class MBad(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(8, 16)

        def forward(self, q, k, v):
            return self.w(F.scaled_dot_product_attention(q, k, v))  # out last is 9

    assert not _verify(MBad(), q=(2, 4, 5, 8), k=(2, 4, 7, 8), v=(2, 4, 7, 9)).safe


def test_e2e_from_scratch_mha_block_safe():
    if not HAS_SDPA:
        return

    class MHABlock(nn.Module):
        def __init__(self, d=32, h=4):
            super().__init__()
            self.h, self.d = h, d
            self.qp = nn.Linear(d, d)
            self.kp = nn.Linear(d, d)
            self.vp = nn.Linear(d, d)
            self.op = nn.Linear(d, d)

        def forward(self, x):
            B, T, _ = x.shape
            q = self.qp(x).reshape(B, T, self.h, self.d // self.h).transpose(1, 2)
            k = self.kp(x).reshape(B, T, self.h, self.d // self.h).transpose(1, 2)
            v = self.vp(x).reshape(B, T, self.h, self.d // self.h).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(1, 2).reshape(B, T, self.d)
            return self.op(a)

    r = _verify(MHABlock(), x=(2, 7, 32))
    assert r.safe, _violation_kinds(r)


# ---------------------------------------------------------------------------
# nn.MultiheadAttention layer (already supported) — guard against regressions
# ---------------------------------------------------------------------------

def test_mha_layer_valid_safe():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = nn.MultiheadAttention(32, 4, batch_first=True)

        def forward(self, x):
            out, _ = self.mha(x, x, x)
            return out

    assert _verify(M(), x=(2, 5, 32)).safe


# ---------------------------------------------------------------------------
# Real model: timm ViT (uses F.scaled_dot_product_attention internally)
# ---------------------------------------------------------------------------

def test_real_timm_vit_uses_sdpa_and_is_safe():
    try:
        import timm
    except ImportError:
        return
    from src.model_checker import OpKind
    from src.fx_extractor import fx_trace_to_graph

    m = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
    try:
        traced = torch.fx.symbolic_trace(m)
    except Exception:
        return
    g = fx_trace_to_graph(traced)
    sdpa_steps = [s for s in g.steps if s.op == OpKind.SDPA]
    # The model genuinely exercises the SDPA path.
    assert len(sdpa_steps) > 0
    r = verify_module(m, input_shapes={"x": (1, 3, 224, 224)})
    assert r.safe, _violation_kinds(r)
