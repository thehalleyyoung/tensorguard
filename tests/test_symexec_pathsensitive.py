"""Tests for path-sensitive guard refinement (data/flow reasoning).

These exercise the interpreter's ``_refine`` machinery: facts established by an
``if`` guard must flow into the corresponding branch so that

* genuine forced failures on a feasible path are still reported, and
* failures that the guard *excludes* are not reported (no false positives).
"""

import pytest

from src.symexec import analyze_source, SymBugKind


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


# ── division-by-zero refinement ──────────────────────────────────────────────
def test_unguarded_div_by_zero_is_reported():
    src = "def f():\n    n = 0\n    return 5 // n\n"
    assert SymBugKind.DIVISION_BY_ZERO in _kinds(src)


def test_eq_zero_guard_excludes_zero_in_else():
    src = (
        "def f():\n"
        "    n = 0\n"
        "    if n == 0:\n"
        "        return 0\n"
        "    return 5 // n\n"
    )
    assert _kinds(src) == []


def test_ne_zero_guard_allows_division():
    src = (
        "def f():\n"
        "    n = 0\n"
        "    if n != 0:\n"
        "        return 5 // n\n"
        "    return 0\n"
    )
    assert _kinds(src) == []


def test_gt_zero_guard_allows_division():
    src = (
        "def f():\n"
        "    n = 0\n"
        "    if n > 0:\n"
        "        return 5 // n\n"
        "    return 0\n"
    )
    assert _kinds(src) == []


def test_not_eq_zero_via_not_operator():
    src = (
        "def f():\n"
        "    n = 0\n"
        "    if not n == 0:\n"
        "        return 5 // n\n"
        "    return 0\n"
    )
    assert _kinds(src) == []


def test_truthiness_guard_excludes_zero():
    src = (
        "def f():\n"
        "    n = 0\n"
        "    if n:\n"
        "        return 5 // n\n"
        "    return 0\n"
    )
    assert _kinds(src) == []


def test_swapped_operand_zero_guard():
    # ``0 != n`` is the same as ``n != 0``
    src = (
        "def f():\n"
        "    n = 0\n"
        "    if 0 != n:\n"
        "        return 5 // n\n"
        "    return 0\n"
    )
    assert _kinds(src) == []


# ── None refinement ──────────────────────────────────────────────────────────
def test_none_deref_reported():
    src = "def f():\n    x = None\n    return x.shape\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


def test_none_subscript_reported():
    src = "def f():\n    x = None\n    return x[0]\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


def test_is_none_reassign_clears_deref():
    src = (
        "def f():\n"
        "    x = None\n"
        "    if x is None:\n"
        "        x = make()\n"
        "    return x.shape\n"
    )
    assert _kinds(src) == []


def test_is_not_none_guard_clears_deref():
    src = (
        "def f(x):\n"
        "    if x is not None:\n"
        "        return x.shape\n"
        "    return None\n"
    )
    assert _kinds(src) == []


def test_and_guard_refines_both_operands():
    src = (
        "def f():\n"
        "    n = 0\n"
        "    if n != 0 and n < 100:\n"
        "        return 5 // n\n"
        "    return 0\n"
    )
    assert _kinds(src) == []


# ── assert-based refinement ──────────────────────────────────────────────────
def test_assert_ne_zero_clears_div_by_zero():
    src = "def f():\n    n = 0\n    assert n != 0\n    return 5 // n\n"
    assert _kinds(src) == []


def test_assert_is_not_none_clears_deref():
    src = "def f():\n    x = None\n    assert x is not None\n    return x.shape\n"
    assert _kinds(src) == []


def test_assert_truthiness_clears_div_by_zero():
    src = "def f():\n    n = 0\n    assert n\n    return 5 // n\n"
    assert _kinds(src) == []
