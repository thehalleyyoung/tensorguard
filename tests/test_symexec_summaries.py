"""Step 44 — function summaries: cache (callee, frame-depth, arg/self
abstraction) -> (return, transitive bugs, single-return flag), replaying the
return and re-emitting bugs on reuse instead of re-executing the body.  The
cache is a behavior-preserving memoization: reports and inferred shapes must be
identical to the un-cached analysis, and deterministic across call order."""

import ast

from src.symexec.engine import analyze_source
from src.symexec.interpreter import Interpreter
from src.symexec.bugs import SymBugKind


def _kinds(src):
    return [b.kind for b in analyze_source(src).bugs]


def _run(src, entry="f"):
    m = ast.parse(src)
    it = Interpreter(m)
    fn = next(n for n in m.body if isinstance(n, ast.FunctionDef) and n.name == entry)
    it.run_function(fn, {})
    return it


# --------------------------------------------------------------------------
# Cache population & reuse
# --------------------------------------------------------------------------
def test_repeated_identical_call_populates_cache():
    src = """
import torch
def helper(x):
    return x.reshape(2, 7)
def f():
    a = torch.zeros(3, 4)
    return helper(a), helper(a), helper(a)
"""
    it = _run(src)
    # helper + f both summarized
    assert len(it._summary_cache) == 2


def test_identical_calls_dedup_to_single_report():
    src = """
import torch
def helper(x):
    return x.reshape(2, 7)
def f():
    a = torch.zeros(3, 4)
    return helper(a), helper(a), helper(a)
"""
    assert _kinds(src) == [SymBugKind.RESHAPE_SIZE_MISMATCH]


def test_cache_hit_still_reports_bug():
    # The 2nd+ calls hit the cache; the bug must still surface (re-emitted).
    src = """
import torch
def helper(x):
    return x.reshape(2, 7)
def f():
    a = torch.zeros(3, 4)
    _ = helper(a)
    return helper(a)
"""
    assert SymBugKind.RESHAPE_SIZE_MISMATCH in _kinds(src)


# --------------------------------------------------------------------------
# Context sensitivity preserved (distinct abstractions -> distinct entries)
# --------------------------------------------------------------------------
def test_distinct_args_handled_independently():
    src = """
import torch
def helper(x):
    return x.reshape(2, 7)
def f():
    a = torch.zeros(3, 4)   # 12 elems -> bug
    b = torch.zeros(2, 7)   # 14 elems -> ok
    return helper(a), helper(b)
"""
    # exactly one report: from the 12-elem context, not suppressed by the ok one
    assert _kinds(src) == [SymBugKind.RESHAPE_SIZE_MISMATCH]


def test_clean_call_no_false_positive():
    src = """
import torch
def helper(x):
    return x.reshape(2, 6)   # 12 -> ok
def f():
    a = torch.zeros(3, 4)
    return helper(a), helper(a)
"""
    assert _kinds(src) == []


# --------------------------------------------------------------------------
# Behavior preservation: cached == uncached
# --------------------------------------------------------------------------
def test_cached_matches_uncached_reports():
    one = """
import torch
def helper(x):
    return x.reshape(2, 7)
def f():
    a = torch.zeros(3, 4)
    return helper(a)
"""
    many = """
import torch
def helper(x):
    return x.reshape(2, 7)
def f():
    a = torch.zeros(3, 4)
    return helper(a), helper(a), helper(a), helper(a)
"""
    assert _kinds(one) == _kinds(many)


def test_return_value_reused_for_downstream_shape():
    # helper's inferred return shape must flow through a cache hit so a
    # downstream reshape on the reused result is still checked.
    src = """
import torch
def make():
    return torch.zeros(3, 4)   # 12 elems
def f():
    a = make()
    b = make()                 # cache hit, same inferred (3,4)
    return b.reshape(5, 5)     # 25 != 12 -> bug
"""
    assert SymBugKind.RESHAPE_SIZE_MISMATCH in _kinds(src)


# --------------------------------------------------------------------------
# Methods (self abstraction in the key)
# --------------------------------------------------------------------------
def test_method_summary_distinguishes_self():
    src = """
import torch
class M:
    def __init__(self, t):
        self.t = t
    def go(self):
        return self.t.reshape(2, 7)
def f():
    a = torch.zeros(3, 4)   # 12 -> bug in go
    m = M(a)
    return m.go(), m.go()
"""
    assert SymBugKind.RESHAPE_SIZE_MISMATCH in _kinds(src)


# --------------------------------------------------------------------------
# Determinism: report set independent of call order
# --------------------------------------------------------------------------
def test_order_independent_reports():
    a_first = """
import torch
def helper(x):
    return x.reshape(2, 7)
def f():
    a = torch.zeros(3, 4)
    b = torch.zeros(2, 7)
    return helper(a), helper(b)
"""
    b_first = """
import torch
def helper(x):
    return x.reshape(2, 7)
def f():
    a = torch.zeros(3, 4)
    b = torch.zeros(2, 7)
    return helper(b), helper(a)
"""
    assert _kinds(a_first) == _kinds(b_first)


# --------------------------------------------------------------------------
# Recursion still terminates with the cache present
# --------------------------------------------------------------------------
def test_recursion_terminates():
    src = """
def rec(n):
    if n <= 0:
        return 0
    return rec(n - 1)
def f():
    return rec(3)
"""
    # must not hang or blow the stack; analysis completes
    assert isinstance(_kinds(src), list)


# --------------------------------------------------------------------------
# Key helper: depth is part of the key
# --------------------------------------------------------------------------
def test_key_includes_depth_and_args():
    src = """
def g(x):
    return x
def f():
    return g(1)
"""
    m = ast.parse(src)
    it = Interpreter(m)
    g = next(n for n in m.body if isinstance(n, ast.FunctionDef) and n.name == "g")
    from src.symexec.values import int_const

    k0 = it._summary_key(g, {"x": int_const(1)}, None)
    k1 = it._summary_key(g, {"x": int_const(2)}, None)
    assert k0 != k1  # different args -> different key
    # depth is encoded as the 2nd field
    assert k0.split("\x01")[1] == "0"
