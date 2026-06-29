"""Tests for Step 5: container abstract values (list/tuple/dict/set), small-map
precision, and the index/key safety checks they enable.
"""

import pytest

from src.symexec import analyze_source, SymBugKind
from src.symexec.values import (
    DictVal,
    ListVal,
    SetVal,
    TupleVal,
    IntVal,
    NONE,
    TOP,
    int_const,
    join,
    summarize_container,
)


def _kinds(r):
    return [b.kind for b in r.bugs]


# ── list / tuple index out of bounds ─────────────────────────────────────────
def test_list_index_out_of_bounds():
    r = analyze_source("def f():\n    xs = [1, 2, 3]\n    return xs[5]\n", "l.py")
    assert SymBugKind.RANK_INDEX_ERROR in _kinds(r)


def test_tuple_index_out_of_bounds():
    r = analyze_source("def f():\n    t = (1, 2)\n    return t[3]\n", "t.py")
    assert SymBugKind.RANK_INDEX_ERROR in _kinds(r)


def test_negative_in_range_index_is_safe():
    r = analyze_source("def f():\n    xs = [1, 2, 3]\n    return xs[-1]\n", "ok.py")
    assert r.bugs == []


def test_negative_out_of_range_index():
    r = analyze_source("def f():\n    xs = [1, 2]\n    return xs[-5]\n", "neg.py")
    assert SymBugKind.RANK_INDEX_ERROR in _kinds(r)


def test_unknown_length_list_no_false_positive():
    r = analyze_source("def f(xs):\n    return xs[100]\n", "u.py")
    assert r.bugs == []


# ── dict small-map precision ─────────────────────────────────────────────────
def test_dict_missing_key():
    r = analyze_source("def f():\n    d = {'a': 1, 'b': 2}\n    return d['c']\n", "d.py")
    assert SymBugKind.NONE_PROPAGATION in _kinds(r)


def test_dict_present_key_is_safe():
    r = analyze_source("def f():\n    d = {'a': 1}\n    return d['a']\n", "d2.py")
    assert r.bugs == []


def test_dict_with_spread_does_not_false_report():
    # ``**other`` makes the key set inexact, so a missing literal key must not
    # be reported.
    r = analyze_source(
        "def f(other):\n    d = {'a': 1, **other}\n    return d['z']\n", "d3.py"
    )
    assert r.bugs == []


# ── value-level container lattice ────────────────────────────────────────────
def test_dict_join_merges_known_keys():
    a = DictVal(known=(("a", int_const(1)), ("b", int_const(2))), exact_keys=True)
    b = DictVal(known=(("a", int_const(1)),), exact_keys=True)
    j = join(a, b)
    assert isinstance(j, DictVal)
    # only 'a' is common; exact_keys must drop because key sets differ
    assert dict(j.known).keys() == {"a"}
    assert j.exact_keys is False


def test_set_join():
    j = join(SetVal(elem=int_const(1), length=3), SetVal(elem=int_const(2), length=3))
    assert isinstance(j, SetVal) and j.length == 3


def test_list_join_keeps_exact_elems_when_aligned():
    a = ListVal(elem=int_const(1), length=2, exact_elems=(int_const(1), int_const(2)))
    b = ListVal(elem=int_const(3), length=2, exact_elems=(int_const(3), int_const(4)))
    j = join(a, b)
    assert isinstance(j, ListVal) and j.length == 2 and j.exact_elems is not None


# ── widening collapses oversized precise containers ──────────────────────────
def test_summarize_collapses_large_list():
    big = ListVal(
        elem=int_const(0), length=100, exact_elems=tuple(int_const(i) for i in range(100))
    )
    s = summarize_container(big)
    assert isinstance(s, ListVal) and s.exact_elems is None and s.length is None


def test_summarize_keeps_small_list():
    small = ListVal(elem=int_const(0), length=2, exact_elems=(int_const(1), int_const(2)))
    assert summarize_container(small) is small
