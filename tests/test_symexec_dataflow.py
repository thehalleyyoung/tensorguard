"""Tests for Step 14 (unreachable-path pruning on contradictory guards) and
Step 25 (reduction transfer functions with dim/keepdim rank effects).

These exercise data/flow precision: a contradictory guard's body must not
produce reports, and a reduction's rank effect must flow into the subsequent
subscript rank-safety check.
"""

import pytest

from src.symexec import analyze_source, SymBugKind


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


# ── Step 14: contradictory guard ⇒ dead branch ───────────────────────────────
def test_contradictory_numeric_guard_prunes_body():
    # n = 5 then ``if n < 0`` is infeasible; the divide in the body is dead.
    src = "def f():\n    n = 5\n    if n < 0:\n        return 1 // n\n    return 0\n"
    assert _kinds(src) == []


def test_feasible_eq_zero_branch_still_reports():
    # ``if n == 0`` with n = 0 is feasible, so the in-branch divide must fire.
    src = (
        "def f(x):\n"
        "    n = 0\n"
        "    if n == 0:\n"
        "        return x // n\n"
        "    return 0\n"
    )
    assert SymBugKind.DIVISION_BY_ZERO in _kinds(src)


def test_tensor_is_none_branch_is_dead():
    # x is a tensor, so ``if x is None`` body is unreachable (no rank error).
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(3, 4)\n"
        "    if x is None:\n"
        "        return x[0, 0, 0, 0]\n"
        "    return 0\n"
    )
    assert _kinds(src) == []


def test_impossible_range_intersection_prunes():
    # ``if n > 10`` after n forced to 0: infeasible.
    src = (
        "def f(x):\n"
        "    n = 0\n"
        "    if n > 10:\n"
        "        return x // n\n"
        "    return 0\n"
    )
    assert _kinds(src) == []


# ── Step 25: reduction rank effects ──────────────────────────────────────────
def test_reduction_drops_rank_then_over_index():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    y = x.sum(dim=1)\n"
        "    return y[0, 0, 0]\n"
    )
    assert SymBugKind.RANK_INDEX_ERROR in _kinds(src)


def test_reduction_multi_axis_drops_two_ranks():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    y = x.sum(dim=(1, 2))\n"
        "    return y[0, 0]\n"
    )
    assert SymBugKind.RANK_INDEX_ERROR in _kinds(src)


def test_reduction_keepdim_kwarg_preserves_rank():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    y = x.sum(dim=1, keepdim=True)\n"
        "    return y[0, 0, 0]\n"
    )
    assert _kinds(src) == []


def test_reduction_positional_keepdim_preserves_rank():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    y = x.sum(1, True)\n"
        "    return y[0, 0, 0]\n"
    )
    assert _kinds(src) == []


def test_full_reduction_is_scalar():
    # ``x.sum()`` is 0-d; indexing it at all is a rank error.
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3)\n"
        "    y = x.sum()\n"
        "    return y[0]\n"
    )
    assert SymBugKind.RANK_INDEX_ERROR in _kinds(src)
