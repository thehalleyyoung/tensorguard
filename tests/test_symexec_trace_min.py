"""Step 58 — trace minimization (shrink a counterexample to its minimal slice).

:func:`trace_min.minimize` reduces the set of path conditions that witness a
forced failure to a **1-minimal** slice (delta-debugging): no single remaining
condition can be dropped without the failure ceasing to be forced.  The
broadcast detector attaches this slice to its report so an owner sees the exact
conditions that make the mismatch unavoidable.

Minimization only affects diagnostics — the report fires on the full path facts
regardless — and the slice is re-validated by the ``holds`` predicate, so it is
a sound explanation.
"""

import pytest

from src.symexec.engine import analyze_source
from src.symexec.bugs import SymBugKind
from src.symexec import smt_bridge as B
from src.symexec import trace_min
from src.symexec.interpreter import Interpreter
from src.symexec.symdim import SymDim

z3only = pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")

a, b, c, d = (SymDim.var(n) for n in "abcd")


# --------------------------------------------------------------------------
# generic delta-debug minimizer
# --------------------------------------------------------------------------

def test_minimize_drops_incidental_elements():
    # "holds" iff the subset contains 2 and 3; everything else is incidental.
    holds = lambda s: 2 in s and 3 in s
    assert trace_min.minimize([1, 2, 3, 4], holds) == [2, 3]


def test_minimize_is_order_preserving():
    holds = lambda s: 3 in s and 1 in s
    assert trace_min.minimize([1, 2, 3], holds) == [1, 3]


def test_minimize_empty_when_unconditional():
    assert trace_min.minimize([1, 2, 3], lambda s: True) == []


def test_minimize_returns_full_when_predicate_fails_on_full():
    assert trace_min.minimize([1, 2], lambda s: False) == [1, 2]


def test_minimize_keeps_all_when_each_is_necessary():
    holds = lambda s: set(s) == {1, 2, 3}
    assert trace_min.minimize([1, 2, 3], holds) == [1, 2, 3]


def test_minimize_one_minimal_property():
    # After minimization no single element may be removed while holds stays True.
    holds = lambda s: 2 in s and 4 in s
    out = trace_min.minimize([1, 2, 3, 4, 5], holds)
    assert holds(out)
    for x in out:
        assert not holds([y for y in out if y != x])


# --------------------------------------------------------------------------
# broadcast forcing predicate + minimization over real DimConstraints
# --------------------------------------------------------------------------

@z3only
def test_broadcast_forced_predicate():
    facts = [B.ne(a, b), B.ne(a, 1), B.ne(b, 1)]
    assert Interpreter._broadcast_forced(facts, a, b) is True
    # Drop "a != b": now a == b is feasible ⇒ broadcastable ⇒ not forced.
    assert Interpreter._broadcast_forced([B.ne(a, 1), B.ne(b, 1)], a, b) is False


@z3only
def test_minimize_drops_unrelated_broadcast_fact():
    facts = [B.ne(a, b), B.ne(a, 1), B.ne(b, 1), B.eq(c, d)]
    holds = lambda s: Interpreter._broadcast_forced(s, a, b)
    minimal = trace_min.minimize(facts, holds)
    assert B.eq(c, d) not in minimal  # incidental ⇒ removed
    assert set(minimal) == {B.ne(a, b), B.ne(a, 1), B.ne(b, 1)}


@z3only
def test_minimize_each_broadcast_fact_is_necessary():
    facts = [B.ne(a, b), B.ne(a, 1), B.ne(b, 1)]
    holds = lambda s: Interpreter._broadcast_forced(s, a, b)
    minimal = trace_min.minimize(facts, holds)
    assert set(minimal) == set(facts)  # none removable


# --------------------------------------------------------------------------
# integration: the broadcast report carries the minimal slice
# --------------------------------------------------------------------------

_SRC = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a"], y: Float[Tensor, "b"]):
    if x.size(0) != y.size(0):
        if x.size(0) != 1:
            if y.size(0) != 1:
                return x + y
    return x
"""


@z3only
def test_broadcast_report_includes_minimal_conditions():
    bugs = [b for b in analyze_source(_SRC).bugs if b.kind == SymBugKind.BROADCAST_MISMATCH]
    assert bugs
    ev = bugs[0].evidence
    assert ev is not None
    assert "minimal failing conditions" in ev
    # All three guard conditions are necessary and must appear.
    for needle in ("a != b", "a != 1", "b != 1"):
        assert needle in ev
