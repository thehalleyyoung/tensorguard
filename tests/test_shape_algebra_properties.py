"""Step 19 -- property-based tests over the shape-algebra transfer functions.

The existing `test_denotational_semantics.py` checks the concrete/abstract
transfer functions on hand-picked examples. This module adds **property-based**
coverage with Hypothesis: it generates thousands of random shapes and asserts
the *algebraic laws* of the shape algebra and, crucially, the **per-node
soundness theorem** of the abstract interpretation:

    α(⟦op⟧(σ))  ⊑  ⟦op⟧♯(α(σ))

i.e. the abstract transfer function over-approximates the concrete one. This is
exactly the lemma the verifier's soundness argument is built on, so testing it
over a large random input space is high value.

Algebraic laws covered: transpose involution, reshape element-count
preservation + round-trip, flatten element-count preservation, squeeze∘unsqueeze
inverse, unsqueeze rank growth, global squeeze removes all size-1 dims, cat
dimension summation, matmul output shape + inner-dim precondition, add broadcast
commutativity, and identity.
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# Cold-start import/JIT cost can make the first test's input generation look
# slow; suppress that health check globally. Per-test @settings inherit it.
settings.register_profile(
    "shape_algebra", suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("shape_algebra")

from src.denotational_semantics import (
    AbstractShape,
    abstract_add,
    abstract_cat,
    abstract_flatten,
    abstract_identity,
    abstract_matmul,
    abstract_reshape,
    abstract_squeeze,
    abstract_transpose,
    abstract_unsqueeze,
    concrete_add,
    concrete_cat,
    concrete_flatten,
    concrete_identity,
    concrete_matmul,
    concrete_reshape,
    concrete_squeeze,
    concrete_transpose,
    concrete_unsqueeze,
)

# ---- strategies -----------------------------------------------------------
DIM = st.integers(min_value=1, max_value=6)


def shapes(min_rank: int = 1, max_rank: int = 5):
    return st.lists(DIM, min_size=min_rank, max_size=max_rank).map(tuple)


def _prod(t):
    return math.prod(t) if t else 1


def _abs(t):
    return AbstractShape(dims=tuple(t))


# Helper: the per-node soundness theorem for an op on concrete inputs.
# On all-concrete inputs the abstract result must over-approximate (here:
# equal, since dims are concrete) the abstraction of the concrete result.
def _assert_sound(concrete_result, abstract_result):
    assert _abs(concrete_result).leq(abstract_result), (
        "soundness violated: alpha(%r) not <= %r"
        % (concrete_result, abstract_result.dims))


# ---- transpose ------------------------------------------------------------
@settings(max_examples=300)
@given(shapes(min_rank=2), st.integers(-5, 5), st.integers(-5, 5))
def test_transpose_is_an_involution(a, i, j):
    rank = len(a)
    i %= rank
    j %= rank
    once = concrete_transpose([a], dim0=i, dim1=j)
    twice = concrete_transpose([once], dim0=i, dim1=j)
    assert twice == a


@settings(max_examples=300)
@given(shapes(min_rank=2), st.integers(-5, 5), st.integers(-5, 5))
def test_transpose_preserves_multiset_and_is_sound(a, i, j):
    rank = len(a)
    i %= rank
    j %= rank
    cr = concrete_transpose([a], dim0=i, dim1=j)
    assert sorted(cr) == sorted(a)
    ar = abstract_transpose([_abs(a)], dim0=i, dim1=j)
    _assert_sound(cr, ar)


# ---- reshape --------------------------------------------------------------
@settings(max_examples=300)
@given(shapes())
def test_reshape_roundtrip_and_count(a):
    target = tuple(reversed(a))  # same element count by construction
    r = concrete_reshape([a], target=target)
    assert _prod(r) == _prod(a)
    back = concrete_reshape([r], target=a)
    assert back == a
    # abstract_reshape returns the target verbatim -> sound w.r.t. concrete.
    _assert_sound(r, abstract_reshape([_abs(a)], target=target))


@settings(max_examples=200)
@given(shapes())
def test_reshape_to_flat_vector(a):
    n = _prod(a)
    r = concrete_reshape([a], target=(n,))
    assert r == (n,)


# ---- flatten --------------------------------------------------------------
@settings(max_examples=300)
@given(shapes(), st.integers(-5, 5), st.integers(-5, 5))
def test_flatten_preserves_element_count_and_is_sound(a, s, e):
    rank = len(a)
    s %= rank
    e %= rank
    assume(s <= e)
    cr = concrete_flatten([a], start_dim=s, end_dim=e)
    assert _prod(cr) == _prod(a)
    assert len(cr) == rank - (e - s)
    ar = abstract_flatten([_abs(a)], start_dim=s, end_dim=e)
    _assert_sound(cr, ar)


# ---- squeeze / unsqueeze --------------------------------------------------
@settings(max_examples=300)
@given(shapes(), st.integers(0, 6))
def test_unsqueeze_then_squeeze_is_identity(a, d):
    d %= (len(a) + 1)
    u = concrete_unsqueeze([a], dim=d)
    assert len(u) == len(a) + 1
    assert u[d] == 1
    back = concrete_squeeze([u], dim=d)
    assert back == a
    # soundness of both transfers on concrete inputs
    _assert_sound(u, abstract_unsqueeze([_abs(a)], dim=d))
    _assert_sound(back, abstract_squeeze([_abs(u)], dim=d))


@settings(max_examples=300)
@given(shapes())
def test_global_squeeze_removes_all_ones(a):
    r = concrete_squeeze([a], dim=None)
    assert all(d != 1 for d in r)
    # every non-1 dim of a survives, in order
    assert r == tuple(d for d in a if d != 1)
    _assert_sound(r, abstract_squeeze([_abs(a)], dim=None))


# ---- cat ------------------------------------------------------------------
@settings(max_examples=300)
@given(shapes(), st.integers(0, 5), st.lists(DIM, min_size=1, max_size=4))
def test_cat_sums_along_dim_preserves_others(base, d, extra_sizes):
    rank = len(base)
    d %= rank
    # Build sibling shapes identical to base except along dim d.
    sib_shapes = [base]
    for sz in extra_sizes:
        s = list(base)
        s[d] = sz
        sib_shapes.append(tuple(s))
    cr = concrete_cat(sib_shapes, dim=d)
    assert cr[d] == sum(s[d] for s in sib_shapes)
    for k in range(rank):
        if k != d:
            assert cr[k] == base[k]
    ar = abstract_cat([_abs(s) for s in sib_shapes], dim=d)
    _assert_sound(cr, ar)


# ---- matmul ---------------------------------------------------------------
@settings(max_examples=300)
@given(st.lists(DIM, min_size=0, max_size=3), DIM, DIM, DIM)
def test_matmul_output_shape_and_soundness(batch, m, k, n):
    a = tuple(batch) + (m, k)
    b = tuple(batch) + (k, n)
    cr = concrete_matmul([a, b])
    assert cr == tuple(batch) + (m, n)
    ar = abstract_matmul([_abs(a), _abs(b)])
    _assert_sound(cr, ar)


@settings(max_examples=200)
@given(DIM, DIM, DIM, DIM)
def test_matmul_inner_dim_mismatch_raises(m, k1, k2, n):
    assume(k1 != k2)
    a = (m, k1)
    b = (k2, n)
    raised = False
    try:
        concrete_matmul([a, b])
    except ValueError:
        raised = True
    assert raised


# ---- add ------------------------------------------------------------------
@settings(max_examples=300)
@given(shapes(), shapes())
def test_add_broadcast_is_shape_commutative(a, b):
    # Only consider broadcastable pairs.
    try:
        ab = concrete_add([a, b])
    except ValueError:
        assume(False)
        return
    ba = concrete_add([b, a])
    assert ab == ba
    _assert_sound(ab, abstract_add([_abs(a), _abs(b)]))


@settings(max_examples=200)
@given(shapes())
def test_add_with_self_is_identity_shape(a):
    assert concrete_add([a, a]) == a
    _assert_sound(a, abstract_add([_abs(a), _abs(a)]))


# ---- identity -------------------------------------------------------------
@settings(max_examples=100)
@given(shapes())
def test_identity_is_identity(a):
    assert concrete_identity([a]) == a
    assert abstract_identity([_abs(a)]).dims == tuple(a)
    _assert_sound(a, abstract_identity([_abs(a)]))
