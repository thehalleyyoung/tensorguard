"""
K-Z-K Idempotence Tests.

Verifies that the Knuth-Bendix → Z3 → Knuth-Bendix pipeline is idempotent:

    normalize_z3_expr(normalize_z3_expr(expr)) == normalize_z3_expr(expr)

This guards against the double-simplification hazard where Z3's internal
simplifier and the KB rewrite system interact non-idempotently.
"""

from __future__ import annotations

import pytest

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

pytestmark = pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")

if HAS_Z3:
    from src.knuth_bendix import normalize_z3_expr, term_to_z3, z3_to_term


# ======================================================================
# Helpers
# ======================================================================

def assert_kzk_idempotent(expr: "z3.ExprRef", label: str = "") -> None:
    """Assert normalize_z3_expr is idempotent on *expr*."""
    r1 = normalize_z3_expr(expr)
    r2 = normalize_z3_expr(r1)
    assert z3.eq(r1, r2), (
        f"K-Z-K not idempotent{' for ' + label if label else ''}: "
        f"normalize({expr}) = {r1}, normalize(normalize({expr})) = {r2}"
    )


def bc(*args):
    """Uninterpreted bc function."""
    sorts = [z3.IntSort()] * (len(args) + 1)
    f = z3.Function("bc", *sorts)
    return f(*args)


def numel(s):
    f = z3.Function("numel", z3.IntSort(), z3.IntSort())
    return f(s)


def reshape(s, t):
    f = z3.Function("reshape", z3.IntSort(), z3.IntSort(), z3.IntSort())
    return f(s, t)


def transpose(s, d0, d1):
    f = z3.Function("transpose", z3.IntSort(), z3.IntSort(),
                     z3.IntSort(), z3.IntSort())
    return f(s, d0, d1)


def conv_out(h, k, stride, pad):
    f = z3.Function("conv_out", z3.IntSort(), z3.IntSort(),
                     z3.IntSort(), z3.IntSort(), z3.IntSort())
    return f(h, k, stride, pad)


def pool_out(h, k, stride, pad):
    f = z3.Function("pool_out", z3.IntSort(), z3.IntSort(),
                     z3.IntSort(), z3.IntSort(), z3.IntSort())
    return f(h, k, stride, pad)


# ======================================================================
# Simple Arithmetic
# ======================================================================


class TestKZKArithmetic:
    """Idempotence on simple arithmetic identities."""

    def test_add_zero(self):
        x = z3.Int("dim_h")
        assert_kzk_idempotent(x + 0, "dim + 0")

    def test_mul_one(self):
        x = z3.Int("dim_w")
        assert_kzk_idempotent(x * 1, "dim * 1")

    def test_sub_zero(self):
        x = z3.Int("dim_c")
        assert_kzk_idempotent(x - 0, "dim - 0")

    def test_add_commutative(self):
        x, y = z3.Int("a"), z3.Int("b")
        assert_kzk_idempotent(x + y, "a + b")

    def test_mul_commutative(self):
        x, y = z3.Int("a"), z3.Int("b")
        assert_kzk_idempotent(x * y, "a * b")

    def test_nested_arithmetic(self):
        x, y = z3.Int("a"), z3.Int("b")
        assert_kzk_idempotent((x + y) * 1 + 0, "(a+b)*1+0")

    def test_constant_folding(self):
        assert_kzk_idempotent(z3.IntVal(3) + z3.IntVal(4), "3+4")


# ======================================================================
# Broadcast Expressions
# ======================================================================


class TestKZKBroadcast:
    """Idempotence on broadcast (bc) expressions."""

    def test_bc_identity_right(self):
        x = z3.Int("dim")
        assert_kzk_idempotent(bc(x, z3.IntVal(1)), "bc(dim, 1)")

    def test_bc_identity_left(self):
        x = z3.Int("dim")
        assert_kzk_idempotent(bc(z3.IntVal(1), x), "bc(1, dim)")

    def test_bc_idempotent(self):
        x = z3.Int("dim")
        assert_kzk_idempotent(bc(x, x), "bc(dim, dim)")

    def test_bc_nested_identity(self):
        x, y = z3.Int("a"), z3.Int("b")
        assert_kzk_idempotent(bc(bc(x, z3.IntVal(1)), y),
                              "bc(bc(a,1), b)")

    def test_bc_double_idempotent(self):
        x = z3.Int("dim")
        assert_kzk_idempotent(bc(bc(x, x), x), "bc(bc(dim,dim), dim)")


# ======================================================================
# Reshape / Transpose
# ======================================================================


class TestKZKReshapeTranspose:
    """Idempotence on reshape and transpose expressions."""

    def test_numel_reshape(self):
        s, t = z3.Int("shape"), z3.Int("target")
        assert_kzk_idempotent(numel(reshape(s, t)), "numel(reshape(s, t))")

    def test_double_transpose(self):
        s = z3.Int("shape")
        d0, d1 = z3.IntVal(0), z3.IntVal(1)
        assert_kzk_idempotent(transpose(transpose(s, d0, d1), d0, d1),
                              "transp(transp(s,0,1),0,1)")

    def test_nested_reshape_numel(self):
        s, t1, t2 = z3.Int("s"), z3.Int("t1"), z3.Int("t2")
        assert_kzk_idempotent(numel(reshape(reshape(s, t1), t2)),
                              "numel(reshape(reshape(s,t1),t2))")

    def test_reshape_standalone(self):
        s, t = z3.Int("s"), z3.Int("t")
        assert_kzk_idempotent(reshape(s, t), "reshape(s, t)")

    def test_transpose_standalone(self):
        s, d0, d1 = z3.Int("s"), z3.IntVal(0), z3.IntVal(2)
        assert_kzk_idempotent(transpose(s, d0, d1), "transpose(s, 0, 2)")


# ======================================================================
# Conv Output Formulas
# ======================================================================


class TestKZKConv:
    """Idempotence on convolution output formulas."""

    def test_conv_stride1_pad0(self):
        h, k = z3.Int("h"), z3.Int("k")
        assert_kzk_idempotent(conv_out(h, k, z3.IntVal(1), z3.IntVal(0)),
                              "conv(h,k,1,0)")

    def test_conv_with_padding(self):
        h, k, p = z3.Int("h"), z3.Int("k"), z3.Int("p")
        assert_kzk_idempotent(conv_out(h, k, z3.IntVal(1), p),
                              "conv(h,k,1,p)")

    def test_conv_with_stride(self):
        h, k, s = z3.Int("h"), z3.Int("k"), z3.Int("s")
        assert_kzk_idempotent(conv_out(h, k, s, z3.IntVal(0)),
                              "conv(h,k,s,0)")


# ======================================================================
# Pool Output Formulas
# ======================================================================


class TestKZKPool:
    """Idempotence on pooling output formulas."""

    def test_pool_stride_eq_kernel(self):
        h, k = z3.Int("h"), z3.Int("k")
        assert_kzk_idempotent(pool_out(h, k, k, z3.IntVal(0)),
                              "pool(h,k,k,0)")

    def test_pool_with_padding(self):
        h, k, p = z3.Int("h"), z3.Int("k"), z3.Int("p")
        assert_kzk_idempotent(pool_out(h, k, k, p),
                              "pool(h,k,k,p)")


# ======================================================================
# Mixed / Composed Expressions
# ======================================================================


class TestKZKMixed:
    """Idempotence on mixed expressions combining multiple operations."""

    def test_bc_of_conv(self):
        h, k = z3.Int("h"), z3.Int("k")
        expr = bc(conv_out(h, k, z3.IntVal(1), z3.IntVal(0)), z3.IntVal(1))
        assert_kzk_idempotent(expr, "bc(conv(h,k,1,0), 1)")

    def test_numel_reshape_plus_zero(self):
        s, t = z3.Int("s"), z3.Int("t")
        expr = numel(reshape(s, t)) + 0
        assert_kzk_idempotent(expr, "numel(reshape(s,t)) + 0")

    def test_bc_transpose_chain(self):
        s = z3.Int("s")
        d0, d1 = z3.IntVal(0), z3.IntVal(1)
        expr = bc(transpose(transpose(s, d0, d1), d0, d1), z3.IntVal(1))
        assert_kzk_idempotent(expr, "bc(transp(transp(s,0,1),0,1), 1)")

    def test_conv_then_pool(self):
        h, k1, k2 = z3.Int("h"), z3.Int("k1"), z3.Int("k2")
        conv_result = conv_out(h, k1, z3.IntVal(1), z3.IntVal(0))
        expr = pool_out(conv_result, k2, k2, z3.IntVal(0))
        assert_kzk_idempotent(expr, "pool(conv(h,k1,1,0), k2,k2,0)")

    def test_complex_nested(self):
        a, b, c = z3.Int("a"), z3.Int("b"), z3.Int("c")
        expr = bc(bc(a, z3.IntVal(1)), bc(b, b))
        assert_kzk_idempotent(expr, "bc(bc(a,1), bc(b,b))")

    def test_mul_one_inside_bc(self):
        x = z3.Int("x")
        expr = bc(x * 1, z3.IntVal(1))
        assert_kzk_idempotent(expr, "bc(x*1, 1)")

    def test_triple_normalize(self):
        """Triple application should also equal single application."""
        x = z3.Int("x")
        expr = bc(bc(x, z3.IntVal(1)), x)
        r1 = normalize_z3_expr(expr)
        r3 = normalize_z3_expr(normalize_z3_expr(normalize_z3_expr(expr)))
        assert z3.eq(r1, r3), (
            f"Triple normalize differs: {r1} vs {r3}"
        )
