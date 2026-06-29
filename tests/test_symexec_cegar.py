"""Step 55 — interpolant-based refinement (CEGAR).

When an enclosing guard already established a symbolic dimension fact, a nested
guard that contradicts it produces a *spurious* path: infeasible for every
concrete shape.  The executor now (a) asks the Z3 bridge whether the accumulated
path facts are jointly unsatisfiable, (b) extracts the minimal responsible
subset (a Craig-style interpolant via :func:`smt_bridge.unsat_core`), and (c)
prunes the dead path while recording the interpolant.

Soundness: a path is refined away only on a Z3-*proved* contradiction; missing
z3 / ``unknown`` keeps the path reachable, so refinement never hides a real bug.
"""

import ast

import pytest

from src.symexec.engine import analyze_source
from src.symexec.bugs import SymBugKind
from src.symexec import smt_bridge as B
from src.symexec import cegar
from src.symexec.interpreter import Interpreter

z3only = pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")


def _eq(a, b):
    return B.DimConstraint(B.SymDim.var(a), "==", B.SymDim.var(b))


def _ne(a, b):
    return B.DimConstraint(B.SymDim.var(a), "!=", B.SymDim.var(b))


# --------------------------------------------------------------------------
# cegar.refine — the core classifier
# --------------------------------------------------------------------------

@z3only
def test_refine_flags_contradiction_as_spurious():
    r = cegar.refine([_ne("a", "b"), _eq("a", "b")])
    assert r.spurious is True
    assert r.interpolant  # non-empty interpolant
    # The interpolant is exactly the contradictory pair (minimal core).
    assert set(r.interpolant) == {_ne("a", "b"), _eq("a", "b")}


@z3only
def test_refine_keeps_satisfiable_path():
    r = cegar.refine([_ne("a", "b")])
    assert r.spurious is False
    assert r.interpolant == ()


@z3only
def test_refine_minimal_core_ignores_irrelevant_fact():
    # ``c == d`` is irrelevant to the a/b contradiction and must not appear.
    facts = [_eq("c", "d"), _ne("a", "b"), _eq("a", "b")]
    r = cegar.refine(facts)
    assert r.spurious is True
    assert _eq("c", "d") not in set(r.interpolant)
    assert set(r.interpolant) == {_ne("a", "b"), _eq("a", "b")}


def test_refine_empty_is_feasible():
    r = cegar.refine([])
    assert r.spurious is False
    assert r.interpolant == ()


def test_refine_without_z3_keeps_path(monkeypatch):
    monkeypatch.setattr(B, "Z3_AVAILABLE", False)
    r = cegar.refine([_ne("a", "b"), _eq("a", "b")])
    assert r.spurious is False  # cannot prove unsat ⇒ never prune


@z3only
def test_interpolant_helper_returns_core():
    core = cegar.interpolant([_ne("a", "b"), _eq("a", "b")])
    assert set(core) == {_ne("a", "b"), _eq("a", "b")}
    assert cegar.interpolant([_eq("a", "b")]) == []


# --------------------------------------------------------------------------
# smt_bridge.unsat_core
# --------------------------------------------------------------------------

@z3only
def test_unsat_core_none_when_satisfiable():
    assert B.unsat_core([_eq("a", "b")]) is None


@z3only
def test_unsat_core_uses_wellformed_floor():
    # ``a < 1`` is unsat under the floor dims >= 1, with no second user fact.
    c = B.DimConstraint(B.SymDim.var("a"), "<", B.SymDim.const_dim(1))
    core = B.unsat_core([c])
    assert core == [c]


# --------------------------------------------------------------------------
# interpreter integration — dead-path pruning
# --------------------------------------------------------------------------

def _run(src):
    return analyze_source(src)


@z3only
def test_contradictory_nested_branch_is_pruned():
    # Inside ``a != b`` the nested ``a == b`` guard is dead: the reshape fault
    # it guards must NOT be reported, and a refinement must be recorded.
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"]):
    if x.size(0) != x.size(1):
        if x.size(0) == x.size(1):
            return x.reshape(x.size(0), x.size(1), 2)
    return x
"""
    res = _run(src)
    kinds = [b.kind for b in res.bugs]
    assert SymBugKind.RESHAPE_SIZE_MISMATCH not in kinds


@z3only
def test_feasible_nested_branch_still_reports():
    # Without the contradicting outer guard the symbolic reshape fault on the
    # reachable path is still reported (pruning must not over-suppress).
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"]):
    if x.size(0) == x.size(1):
        return x.reshape(x.size(0), x.size(1), 2)
    return x
"""
    res = _run(src)
    kinds = [b.kind for b in res.bugs]
    assert SymBugKind.RESHAPE_SIZE_MISMATCH in kinds


@z3only
def test_interpreter_records_refinement_on_prune():
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"]):
    if x.size(0) != x.size(1):
        if x.size(0) == x.size(1):
            y = x.reshape(x.size(0), x.size(1), 2)
    return x
"""
    mod = ast.parse(src)
    interp = Interpreter(mod)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef):
            interp.run_function(node, args={}, self_val=None)
    assert any(r.spurious and r.interpolant for r in interp._refinements)


def test_no_prune_without_z3(monkeypatch):
    monkeypatch.setattr(B, "Z3_AVAILABLE", False)
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"]):
    if x.size(0) != x.size(1):
        if x.size(0) == x.size(1):
            y = x.reshape(x.size(0), x.size(1), 2)
    return x
"""
    mod = ast.parse(src)
    interp = Interpreter(mod)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef):
            interp.run_function(node, args={}, self_val=None)
    assert interp._refinements == []  # nothing proved ⇒ no pruning
