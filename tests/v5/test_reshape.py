"""Tests for v5 SMT-checked reshape divisibility."""
from __future__ import annotations

import pytest

from src.tensor_shapes import ShapeDim, TensorShape
from src.v5.symbolic_config import SymInt
from src.v5.reshape_neg1 import (
    check_reshape_divisibility, check_concrete,
)

z3 = pytest.importorskip("z3")


def _ts(*dims):
    return TensorShape(tuple(ShapeDim(d) for d in dims))


# ── Concrete fast path ──────────────────────────────────────────────────

def test_concrete_safe():
    r = check_concrete([2, 3, 4], [6, -1])
    assert r.verdict == "safe" and r.inferred_neg1 == 4


def test_concrete_unsafe_indivisible():
    r = check_concrete([2, 3, 5], [6, -1])
    # 30 / 6 = 5 → safe
    assert r.verdict == "safe"
    r = check_concrete([2, 3, 5], [4, -1])
    assert r.verdict == "unsafe"


def test_concrete_no_neg1_match():
    assert check_concrete([2, 3], [6]).verdict == "safe"
    assert check_concrete([2, 3], [7]).verdict == "unsafe"


def test_concrete_multiple_neg1():
    assert check_concrete([6], [-1, -1]).verdict == "unsafe"


# ── Symbolic / SMT path ────────────────────────────────────────────────

def test_symbolic_safe_qkv_pattern():
    # Input: (B, T, 3*H*D)  reshape to (B, T, 3, H, D)
    B = SymInt("B"); T = SymInt("T"); H = SymInt("H"); D = SymInt("D")
    inp = _ts(B, T, 3 * H * D)   # SymExpr in last position
    r = check_reshape_divisibility(inp, [B, T, 3, H, D])
    # numel = B*T*(3*H*D); Q = B*T*3*H*D — equal → divisible (and quotient 1).
    assert r.verdict == "safe", r.detail


def test_symbolic_unsafe_when_factor_missing():
    # Input (B, T, H*D) reshaped to (B, T, 3, H, D) — needs H*D divisible by 3*H*D
    # which is impossible (3*H*D > H*D); Z3 should find a counterexample.
    B = SymInt("B"); T = SymInt("T"); H = SymInt("H"); D = SymInt("D")
    inp = _ts(B, T, H * D)
    r = check_reshape_divisibility(inp, [B, T, 3, H, D])
    assert r.verdict == "unsafe", r.detail


def test_symbolic_safe_with_neg1():
    # Input (B, 768) reshape to (B, -1, 64) → -1 = 12 (always)
    B = SymInt("B")
    inp = _ts(B, 768)
    r = check_reshape_divisibility(inp, [B, -1, 64])
    assert r.verdict == "safe"


def test_symbolic_unsafe_with_neg1_indivisible():
    B = SymInt("B")
    inp = _ts(B, 768)
    r = check_reshape_divisibility(inp, [B, -1, 7])  # 768 % 7 != 0
    assert r.verdict == "unsafe"


def test_symbolic_h_divides_n_pattern():
    # Reshape (B, T, h) to (B, T, n, h//n).  Without the divisibility
    # invariant on h, n we should be able to find a counterexample.
    B = SymInt("B"); T = SymInt("T"); h = SymInt("h"); n = SymInt("n")
    inp = _ts(B, T, h)
    r = check_reshape_divisibility(inp, [B, T, n, -1])
    # P = B*T*h, Q = B*T*n; need h % n == 0; not in general → unsafe.
    assert r.verdict == "unsafe"


# ── Real-world integration ─────────────────────────────────────────────

def test_real_torchvision_resnet_view_if_importable():
    tv = pytest.importorskip("torchvision")
    # ResNet uses x.view(x.size(0), -1) before the FC layer.
    # Concrete: input (N, C, 1, 1) → view (N, -1) is always safe.
    r = check_concrete([1, 512, 1, 1], [1, -1])
    assert r.verdict == "safe" and r.inferred_neg1 == 512
