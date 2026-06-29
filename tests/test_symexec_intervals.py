"""Tests for Step 4: the integer interval + symbolic facet of ``IntVal`` and the
numeric reasoning it unlocks (division-by-zero, ranges).

These exercise the data/flow numeric reasoning that lets the engine move beyond
purely structural bugs toward value-dependent ones.
"""

import pytest

from src.symexec import analyze_source, SymBugKind
from src.symexec.values import IntVal, int_const, int_range, join, meet


def _kinds(r):
    return [b.kind for b in r.bugs]


# ── IntVal facets ────────────────────────────────────────────────────────────
def test_int_const_has_both_facets():
    v = int_const(7)
    assert v.const == 7
    assert v.lo() == 7 and v.hi() == 7
    assert v.may_be_zero() is False


def test_int_range_bounds():
    v = int_range(2, 9)
    assert v.lo() == 2 and v.hi() == 9
    assert v.may_be_zero() is False
    assert v.const is None  # not a singleton


def test_int_range_including_zero_may_be_zero():
    assert int_range(-3, 3).may_be_zero() is True
    assert int_range(0, 0).contains_only_zero() is True


def test_intval_join_widens_interval():
    j = join(int_const(3), int_const(8))
    assert isinstance(j, IntVal)
    assert j.lo() == 3 and j.hi() == 8  # convex hull


def test_intval_meet_narrows_interval():
    m = meet(int_range(0, 10), int_range(5, 20))
    assert isinstance(m, IntVal)
    assert m.lo() == 5 and m.hi() == 10


def test_intval_meet_disjoint_ranges_is_bottom():
    m = meet(int_range(0, 3), int_range(7, 9))
    assert m.is_bottom()


# ── division / modulo by zero ────────────────────────────────────────────────
def test_division_by_literal_zero():
    r = analyze_source("def f(x):\n    return x / 0\n", "d.py")
    assert SymBugKind.DIVISION_BY_ZERO in _kinds(r)


def test_modulo_by_literal_zero():
    r = analyze_source("def f(x):\n    return x % 0\n", "m.py")
    assert SymBugKind.DIVISION_BY_ZERO in _kinds(r)


def test_floordiv_by_zero_via_computed_constant():
    # divisor computed as 2 - 2 == 0, caught via symbolic/interval folding
    r = analyze_source("def f(x):\n    d = 2 - 2\n    return x // d\n", "fd.py")
    assert SymBugKind.DIVISION_BY_ZERO in _kinds(r)


def test_no_false_positive_unknown_divisor():
    r = analyze_source("def g(x, n):\n    return x / n\n", "g.py")
    assert r.bugs == []


def test_no_false_positive_nonzero_divisor():
    r = analyze_source("def h(x):\n    return x / 2\n", "h.py")
    assert r.bugs == []


def test_arithmetic_propagates_interval():
    r = analyze_source("def f():\n    a = 3 + 4\n    b = a * 2\n    return b\n", "a.py")
    # no bug, but must not crash and must keep analysis sound
    assert r.bugs == []
