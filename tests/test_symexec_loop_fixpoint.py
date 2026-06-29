"""Steps 17 & 38 — widening-driven loop fixpoint.

Loops are analysed by computing a *loop-head invariant* that over-approximates
the state before every iteration (not just one pass).  Numeric accumulation is
kept terminating by interval widening (unstable bounds jump to ±∞), the first
``_LOOP_UNROLL`` iterations are unrolled precisely for early-bug precision, and
the converged invariant gives a sound post-loop state.
"""

from __future__ import annotations

import ast

from src.symexec import analyze_source, SymBugKind
from src.symexec.interpreter import Interpreter
from src.symexec.state import State


def _run_body(src: str):
    """Execute a single top-level function's statements (except the final
    ``return``) on a fresh state and hand back the resulting state."""
    mod = ast.parse(src)
    interp = Interpreter(mod)
    fn = mod.body[0]
    st = State()
    body = fn.body
    if body and isinstance(body[-1], ast.Return):
        body = body[:-1]
    for s in body:
        st = interp.exec_stmt(s, st)
    return st, interp


def _kinds(src: str, name: str = "m"):
    return [b.kind for b in analyze_source(src, name).bugs]


# ── widening makes accumulation terminate at a sound invariant ──────────────
def test_for_counter_widens_to_unbounded_above():
    st, _ = _run_body(
        "def f():\n    x = 0\n    for i in range(10):\n        x = x + 1\n    return x\n"
    )
    x = st.get("x")
    # Sound for *all* iteration counts: lower bound stays 0, upper bound widens
    # to +inf (None) rather than the 2 seen during precise unrolling.
    assert x.lo() == 0
    assert x.hi() is None


def test_for_decrement_widens_to_unbounded_below():
    st, _ = _run_body(
        "def f():\n    x = 0\n    for i in range(10):\n        x = x - 1\n    return x\n"
    )
    x = st.get("x")
    assert x.hi() == 0
    assert x.lo() is None


def test_while_exit_refines_guard_false():
    # After the loop the guard ``n > 0`` is false, so the exit state has n <= 0.
    st, _ = _run_body(
        "def f():\n    n = 5\n    while n > 0:\n        n = n - 1\n    return n\n"
    )
    n = st.get("n")
    assert n.hi() is not None and n.hi() <= 0


# ── invariant over-approximates the zero-iteration (entry) state ────────────
def test_loop_head_invariant_includes_entry_value():
    # ``y`` is never touched by the loop, so it must survive unchanged.
    st, _ = _run_body(
        "def f():\n    y = 7\n    for i in range(3):\n        z = i\n    return y\n"
    )
    assert st.get("y").const == 7


# ── bugs inside the loop body are still reported ────────────────────────────
def test_none_deref_inside_for_body_reported():
    src = "def f():\n    for i in range(3):\n        x = None\n        y = x.attr\n    return 0\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


def test_none_deref_inside_while_body_reported():
    src = "def f():\n    n = 3\n    while n > 0:\n        x = None\n        y = x.attr\n        n = n - 1\n    return 0\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


# ── no false positives from benign loops ────────────────────────────────────
def test_benign_for_loop_no_bugs():
    src = "def f():\n    total = 0\n    for i in range(10):\n        total = total + i\n    return total\n"
    assert _kinds(src) == []


def test_benign_while_loop_no_bugs():
    src = "def f():\n    n = 100\n    s = 0\n    while n > 0:\n        s = s + n\n        n = n - 1\n    return s\n"
    assert _kinds(src) == []


# ── termination: nested accumulation must not hang or blow up ────────────────
def test_nested_loops_terminate():
    st, _ = _run_body(
        "def f():\n"
        "    x = 0\n"
        "    for i in range(10):\n"
        "        for j in range(10):\n"
        "            x = x + 1\n"
        "    return x\n"
    )
    x = st.get("x")
    assert x.lo() == 0 and x.hi() is None


def test_while_true_loop_terminates_analysis():
    # An unbounded ``while True`` must still terminate the *analysis* (fixpoint),
    # producing a sound over-approximation rather than looping forever.
    src = "def f():\n    x = 0\n    while True:\n        x = x + 1\n    return x\n"
    # Should not raise / hang; result list is well-defined.
    assert isinstance(_kinds(src), list)


# ── single reported instance despite multiple analysis passes ────────────────
def test_loop_body_bug_not_duplicated():
    src = "def f():\n    for i in range(3):\n        x = None\n        y = x.attr\n    return 0\n"
    bugs = analyze_source(src, "m").bugs
    none_bugs = [b for b in bugs if b.kind == SymBugKind.NONE_PROPAGATION]
    # The unroll + final reporting passes hit the same line; engine dedup keeps 1.
    assert len(none_bugs) == 1


# ── narrowing recovers precision lost to widening (Step 18) ──────────────────
def test_narrowing_recovers_constant_upper_bound():
    # Widening blows ``i`` up to [0, +inf); narrowing the guard ``i < 10`` brings
    # it back, and the false-guard exit pins it to exactly 10.
    st, _ = _run_body(
        "def f():\n    i = 0\n    while i < 10:\n        i = i + 1\n    return i\n"
    )
    i = st.get("i")
    assert i.lo() == 10 and i.hi() == 10


def test_narrowing_keeps_unconstrained_accumulator_unbounded():
    # ``range(5)`` bounds the loop var, not ``i``; nothing constrains ``i`` so it
    # must stay soundly unbounded above (narrowing only recovers guard-implied
    # bounds, never invents them).
    st, _ = _run_body(
        "def f():\n    i = 0\n    for k in range(5):\n        i = i + 1\n    return i\n"
    )
    i = st.get("i")
    assert i.lo() == 0 and i.hi() is None


def test_narrowing_is_sound_lower_bound_preserved():
    # ``while i < 100`` from i=0: post-loop i is exactly 100 (guard false), a
    # finite recovery — must never under-approximate (lo stays >= 0).
    st, _ = _run_body(
        "def f():\n    i = 0\n    while i < 100:\n        i = i + 1\n    return i\n"
    )
    i = st.get("i")
    assert i.lo() == 100 and i.hi() == 100


def test_narrowing_terminates_and_no_false_positive():
    src = "def f():\n    i = 0\n    while i < 50:\n        i = i + 1\n    return i\n"
    assert _kinds(src) == []
