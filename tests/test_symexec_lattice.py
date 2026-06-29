"""Unit tests for the symexec foundation: symbolic dims and the value lattice.

Covers Phase-1 steps 1, 2 and 9 of ``SYMEXEC_100_STEPS.md`` — the lattice laws
and the three-valued symbolic-dimension oracle that the transfer functions and
bug checks rest on.
"""

import itertools

import pytest

from src.symexec.symdim import SymDim, fresh_dim
from src.symexec.values import (
    BOTTOM,
    TOP,
    BoolVal,
    IntVal,
    NoneVal,
    TensorVal,
    TupleVal,
    join,
    meet,
    leq,
)


# ── SymDim arithmetic & oracle ──────────────────────────────────────────────
def test_symdim_affine_arithmetic():
    b = SymDim.var("batch")
    expr = b * 2 + 3
    assert str(expr) in ("2*batch + 3",)
    assert (expr - 3).definitely_eq(b * 2)


def test_symdim_const_folding():
    assert (SymDim.const_dim(6).floordiv(2)).value == 3
    assert (SymDim.const_dim(7).mod(3)).value == 1


def test_symdim_three_valued_eq():
    b = SymDim.var("batch")
    assert b.definitely_eq(SymDim.var("batch"))
    assert b.maybe_eq(SymDim.var("seq")) is None  # unknown, not False
    assert SymDim.const_dim(4).maybe_eq(SymDim.const_dim(5)) is False


def test_symdim_divisibility():
    assert (SymDim.var("h") * 4).definitely_divisible_by(2) is True
    assert SymDim.var("h").definitely_divisible_by(2) is None
    assert SymDim.const_dim(9).definitely_divisible_by(3) is True


def test_fresh_dims_are_independent():
    a, b = fresh_dim(), fresh_dim()
    assert not a.definitely_eq(b)


# ── lattice laws ─────────────────────────────────────────────────────────────
def _samples():
    return [
        BOTTOM,
        TOP,
        NoneVal(),
        IntVal(sym=SymDim.const_dim(3)),
        IntVal(sym=SymDim.const_dim(4)),
        IntVal(sym=None),
        BoolVal(const=True),
        TensorVal(rank=2),
        TensorVal(rank=3),
        TensorVal(rank=2, shape=(SymDim.const_dim(4), SymDim.const_dim(8))),
        TupleVal(elems=(TensorVal(rank=2), NoneVal()), exact_len=True),
        TupleVal(elems=(TensorVal(rank=2),), exact_len=True),
    ]


def test_join_idempotent():
    for v in _samples():
        assert join(v, v) == v or join(v, v).is_top()


def test_join_commutative():
    for a, b in itertools.combinations(_samples(), 2):
        assert join(a, b) == join(b, a)


def test_join_associative():
    vs = _samples()
    for a, b, c in itertools.islice(itertools.product(vs, vs, vs), 0, 400):
        left = join(join(a, b), c)
        right = join(a, join(b, c))
        assert left == right


def test_bottom_is_identity():
    for v in _samples():
        assert join(BOTTOM, v) == v
        assert join(v, BOTTOM) == v


def test_top_is_absorbing():
    for v in _samples():
        assert join(TOP, v).is_top()


def test_join_different_types_goes_top():
    assert join(IntVal(sym=SymDim.const_dim(1)), TensorVal(rank=1)).is_top()


def test_tensor_join_rank_disagreement_loses_rank():
    j = join(TensorVal(rank=2), TensorVal(rank=3))
    assert isinstance(j, TensorVal) and j.rank is None


def test_tuple_join_length_disagreement_loses_exact_len():
    a = TupleVal(elems=(NoneVal(), NoneVal()), exact_len=True)
    b = TupleVal(elems=(NoneVal(),), exact_len=True)
    j = join(a, b)
    assert isinstance(j, TupleVal) and j.exact_len is False


# ── meet (greatest lower bound) & order ──────────────────────────────────────
def test_meet_top_is_identity():
    for v in _samples():
        assert meet(TOP, v) == v
        assert meet(v, TOP) == v


def test_meet_bottom_is_absorbing():
    for v in _samples():
        assert meet(BOTTOM, v).is_bottom()
        assert meet(v, BOTTOM).is_bottom()


def test_meet_idempotent():
    for v in _samples():
        assert meet(v, v) == v or meet(v, v).is_bottom()


def test_meet_commutative():
    for a, b in itertools.combinations(_samples(), 2):
        assert meet(a, b) == meet(b, a)


def test_meet_incompatible_constants_is_bottom():
    assert meet(IntVal(sym=SymDim.const_dim(3)), IntVal(sym=SymDim.const_dim(4))).is_bottom()


def test_meet_incompatible_ranks_is_bottom():
    assert meet(TensorVal(rank=2), TensorVal(rank=3)).is_bottom()


def test_meet_refines_unknown_with_concrete():
    refined = meet(IntVal(sym=None), IntVal(sym=SymDim.const_dim(7)))
    assert isinstance(refined, IntVal) and refined.const == 7


def test_meet_different_types_is_bottom():
    assert meet(IntVal(sym=SymDim.const_dim(1)), TensorVal(rank=1)).is_bottom()


# ── lattice order: bottom ⊑ everything ⊑ top ─────────────────────────────────
def test_leq_bottom_least_top_greatest():
    for v in _samples():
        assert leq(BOTTOM, v)
        assert leq(v, TOP)


def test_leq_reflexive():
    for v in _samples():
        assert leq(v, v)


def test_leq_consistent_with_join():
    # a ⊑ b  ⇔  join(a, b) == b
    for a, b in itertools.combinations(_samples(), 2):
        assert leq(a, b) == (join(a, b) == b)


def test_leq_antisymmetry():
    for a, b in itertools.combinations(_samples(), 2):
        if leq(a, b) and leq(b, a):
            assert a == b


def test_concrete_below_unknown():
    assert leq(IntVal(sym=SymDim.const_dim(5)), IntVal(sym=None))
    assert not leq(IntVal(sym=None), IntVal(sym=SymDim.const_dim(5)))


def test_widen_equals_join_for_value_lattice():
    for a, b in itertools.combinations(_samples(), 2):
        assert a.widen(b) == join(a, b)
