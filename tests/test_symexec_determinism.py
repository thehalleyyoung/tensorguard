"""Step 20 — determinism & fixpoint caching.

Two guarantees:

* **Determinism / canonical ordering** — analysing the same source twice yields
  byte-identical results, and reports are emitted in a canonical order (by source
  position, then kind, then message) independent of internal traversal or
  fixpoint-pass order.
* **Fixpoint caching** — re-entering the *same* loop with a lattice-identical
  entry state reuses the converged invariant and re-emits its bugs, so nested
  loops do not lose reports and repeated work is avoided.
"""

from __future__ import annotations

import ast

from src.symexec import analyze_source, SymBugKind
from src.symexec.interpreter import Interpreter
from src.symexec.state import State


def _keys(src: str, name: str = "m"):
    return [(b.line, b.col, b.kind.name, b.message) for b in analyze_source(src, name).bugs]


MULTI_BUG_SRC = (
    "def f():\n"
    "    a = None\n"
    "    z = a.foo\n"          # earlier-line bug
    "    for i in range(3):\n"
    "        x = None\n"
    "        y = x.attr\n"      # later-line bug, inside a loop
    "    return 0\n"
)


# ── determinism ─────────────────────────────────────────────────────────────
def test_same_input_is_deterministic():
    assert _keys(MULTI_BUG_SRC) == _keys(MULTI_BUG_SRC)


def test_report_order_is_canonical():
    keys = _keys(MULTI_BUG_SRC)
    assert keys == sorted(keys)
    # specifically: the line-2 deref precedes the in-loop deref
    assert keys[0][0] < keys[-1][0]


def test_determinism_across_fresh_interpreters():
    # Distinct Interpreter instances (fresh caches) still agree.
    r1 = [(b.kind, b.line, b.message) for b in analyze_source(MULTI_BUG_SRC, "m").bugs]
    r2 = [(b.kind, b.line, b.message) for b in analyze_source(MULTI_BUG_SRC, "m").bugs]
    assert r1 == r2


# ── fixpoint caching ─────────────────────────────────────────────────────────
def test_loop_cache_populated_after_analysis():
    mod = ast.parse("def f():\n    for i in range(3):\n        x = i + 1\n    return x\n")
    interp = Interpreter(mod)
    interp.run_function(mod.body[0], args={})
    assert len(interp._loop_cache) >= 1


def test_loop_cache_hit_reuses_invariant_and_rebinds_bugs():
    # Drive the same loop twice from an identical entry state: the second call
    # must be a cache hit yet still contribute the same bug(s).
    mod = ast.parse(
        "def f():\n    for i in range(3):\n        x = None\n        y = x.attr\n    return 0\n"
    )
    interp = Interpreter(mod)
    forstmt = mod.body[0].body[0]
    elem_state = State()

    def enter(s):
        return s.copy()

    inv1 = interp._run_loop(forstmt, forstmt.body, elem_state, enter)
    n_after_first = len(interp.bugs)
    cache_size = len(interp._loop_cache)
    inv2 = interp._run_loop(forstmt, forstmt.body, elem_state, enter)
    # cache hit: no new cache entry, invariant identical, bugs re-emitted
    assert len(interp._loop_cache) == cache_size
    assert inv1.equals(inv2)
    assert len(interp.bugs) == 2 * n_after_first  # same bugs re-appended


def test_nested_loop_inner_bug_reported_exactly_once():
    src = (
        "def f():\n"
        "    for i in range(4):\n"
        "        for j in range(4):\n"
        "            x = None\n"
        "            y = x.attr\n"
        "    return 0\n"
    )
    bugs = [b for b in analyze_source(src, "m").bugs if b.kind == SymBugKind.NONE_PROPAGATION]
    assert len(bugs) == 1


def test_cache_keyed_per_loop_not_shared():
    # Two different loops must get independent cache entries.
    mod = ast.parse(
        "def f():\n"
        "    for i in range(3):\n"
        "        a = i + 1\n"
        "    for k in range(3):\n"
        "        b = k + 2\n"
        "    return a\n"
    )
    interp = Interpreter(mod)
    interp.run_function(mod.body[0], args={})
    assert len(interp._loop_cache) >= 2
