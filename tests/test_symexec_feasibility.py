"""Step 52 — feasibility-gated reporting.

A candidate fault is reported only when the conjunction of the accumulated
symbolic-dimension *path constraints* and the failing condition is satisfiable
(Z3-checked via the Step-51 bridge).  Reports on a provably-infeasible (dead)
path — e.g. a branch guarded by mutually contradictory dimension comparisons —
are suppressed.

Soundness: suppression happens *only* on a Z3-proved ``unsat``.  Without a
solver (or on ``unknown``) every report is kept, so the gate can never trade the
zero-false-positive guarantee for an unprovable false negative.
"""

import ast

import pytest

from src.symexec.engine import analyze_source
from src.symexec.bugs import SymBugKind, SymBug
from src.symexec.interpreter import Interpreter
from src.symexec.state import State
from src.symexec.symdim import SymDim
from src.symexec import smt_bridge as B


RANK = SymBugKind.RANK_INDEX_ERROR


def _kinds(src):
    return [b.kind for b in analyze_source(src).bugs]


# --------------------------------------------------------------------------
# integration: dead-branch suppression vs live control
# --------------------------------------------------------------------------

_DEAD = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"], y: Float[Tensor, "c d"]):
    if x.size(0) == y.size(0):
        if x.size(0) != y.size(0):
            return x[0, 0, 0]
    return x
"""

_LIVE = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"], y: Float[Tensor, "c d"]):
    if x.size(0) == y.size(0):
        return x[0, 0, 0]
    return x
"""

# a == b, b == c, a != c is unsat by transitivity (linear arithmetic).
_DEAD_TRANSITIVE = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a"], y: Float[Tensor, "b"], z: Float[Tensor, "c"]):
    if x.size(0) == y.size(0):
        if y.size(0) == z.size(0):
            if x.size(0) != z.size(0):
                return x[0, 0, 0]
    return x
"""


@pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")
def test_dead_branch_report_suppressed():
    assert _kinds(_DEAD) == []


@pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")
def test_live_branch_report_kept():
    assert RANK in _kinds(_LIVE)


@pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")
def test_dead_branch_transitive_contradiction_suppressed():
    assert _kinds(_DEAD_TRANSITIVE) == []


# --------------------------------------------------------------------------
# soundness: without a solver, nothing is suppressed
# --------------------------------------------------------------------------

def test_no_suppression_without_z3(monkeypatch):
    monkeypatch.setattr(B, "Z3_AVAILABLE", False)
    # The same dead branch must now keep its report (feasibility is "unknown").
    assert RANK in _kinds(_DEAD)


# --------------------------------------------------------------------------
# unit: the _emit choke point honours explicit failing conditions
# --------------------------------------------------------------------------

def _mk_interp():
    return Interpreter(ast.parse(""))


def _dummy_bug():
    return SymBug(
        kind=RANK,
        message="dummy",
        line=1,
        col=0,
        function="f",
        confidence=0.9,
        fix_suggestion="",
    )


def test_emit_no_facts_always_reports():
    it = _mk_interp()
    it._emit(_dummy_bug())
    assert len(it.bugs) == 1


@pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")
def test_emit_suppressed_when_condition_contradicts_path():
    it = _mk_interp()
    a, b = SymDim.var("a"), SymDim.var("b")
    it._cur_dim_facts = (B.eq(a, b),)
    # failing condition a != b is unsatisfiable under the path fact a == b
    it._emit(_dummy_bug(), conditions=(B.ne(a, b),))
    assert it.bugs == []


@pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")
def test_emit_kept_when_condition_consistent_with_path():
    it = _mk_interp()
    a, b = SymDim.var("a"), SymDim.var("b")
    it._cur_dim_facts = (B.eq(a, b),)
    it._emit(_dummy_bug(), conditions=(B.gt(a, 0),))
    assert len(it.bugs) == 1


@pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")
def test_emit_suppressed_on_infeasible_path_alone():
    it = _mk_interp()
    a, b = SymDim.var("a"), SymDim.var("b")
    it._cur_dim_facts = (B.eq(a, b), B.ne(a, b))
    it._emit(_dummy_bug())
    assert it.bugs == []


# --------------------------------------------------------------------------
# unit: dim-fact recording
# --------------------------------------------------------------------------

def test_record_dim_fact_skips_constant_only_comparison():
    it = _mk_interp()
    state = State()
    cmp = ast.parse("1 == 2", mode="eval").body
    it._refine_compare(cmp, True, state)
    assert state.dim_facts == ()


# --------------------------------------------------------------------------
# unit: State carries and intersects dim_facts soundly
# --------------------------------------------------------------------------

def test_state_copy_preserves_dim_facts():
    a, b = SymDim.var("a"), SymDim.var("b")
    s = State()
    s.dim_facts = (B.eq(a, b),)
    assert s.copy().dim_facts == (B.eq(a, b),)


def test_state_join_intersects_dim_facts():
    # Step 56: the merge keeps every fact *entailed by both* incoming paths.
    # The fact common to both survives; a fact stated on only one path and not
    # implied by the other is dropped.  (This holds with or without z3: without
    # it, ``entails`` degrades to membership and the join is the old syntactic
    # intersection.)
    a, b, c = SymDim.var("a"), SymDim.var("b"), SymDim.var("c")
    s1 = State()
    s1.dim_facts = (B.eq(a, b), B.eq(b, c))
    s2 = State()
    s2.dim_facts = (B.eq(a, b), B.gt(a, 0))
    merged = s1.join(s2)
    assert B.eq(a, b) in merged.dim_facts  # common ⇒ kept
    assert B.eq(b, c) not in merged.dim_facts  # one-sided, not entailed ⇒ dropped


def test_state_join_with_unreachable_keeps_other_side():
    a, b = SymDim.var("a"), SymDim.var("b")
    live = State()
    live.dim_facts = (B.eq(a, b),)
    dead = State(reachable=False)
    assert live.join(dead).dim_facts == (B.eq(a, b),)
