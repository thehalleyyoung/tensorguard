"""Step 57 — disjunctive states (bounded powerset for path precision).

:class:`DisjunctiveState` keeps branch states apart as a bounded set of
alternatives, so straight-line code following an ``if`` is analysed on each path
precisely instead of on a single lossy join.  ``collapse`` (the join of all
disjuncts) is always a sound single-state summary, and the width is bounded (on
overflow the set collapses), so the domain is finite and every operation
terminates.
"""

import ast

import pytest

from src.symexec.engine import analyze_source
from src.symexec.interpreter import Interpreter
from src.symexec.bugs import SymBugKind
from src.symexec.disjunctive import DisjunctiveState, DEFAULT_BOUND
from src.symexec.state import State
from src.symexec.values import int_const

RESHAPE = SymBugKind.RESHAPE_SIZE_MISMATCH


def _state(**env):
    s = State()
    for k, v in env.items():
        s.set(k, v)
    return s


# --------------------------------------------------------------------------
# DisjunctiveState domain
# --------------------------------------------------------------------------

def test_singleton_collapse_is_identity():
    s = _state(n=int_const(3))
    d = DisjunctiveState.singleton(s)
    assert d.width() == 1
    assert d.collapse() is s  # single disjunct returned unchanged


def test_dedup_merges_structurally_equal_states():
    a = _state(n=int_const(3))
    b = _state(n=int_const(3))
    d = DisjunctiveState.of([a, b])
    assert d.width() == 1


def test_distinct_states_kept_apart():
    d = DisjunctiveState.of([_state(n=int_const(2)), _state(n=int_const(3))])
    assert d.width() == 2


def test_collapse_joins_distinct_disjuncts():
    d = DisjunctiveState.of([_state(n=int_const(2)), _state(n=int_const(3))])
    merged = d.collapse()
    # The joined ``n`` is no longer a single constant (2 ⊔ 3 widens away from 2).
    n = merged.get("n")
    assert getattr(n, "const", None) is None


def test_bound_overflow_collapses_to_single():
    states = [_state(n=int_const(i)) for i in range(5)]
    d = DisjunctiveState.of(states, bound=3)
    assert d.width() == 1  # exceeded bound ⇒ collapsed to one joined summary


def test_live_drops_unreachable():
    live = _state(n=int_const(1))
    dead = State.unreachable()
    d = DisjunctiveState.of([live, dead])
    assert d.live().width() == 1
    assert not d.live().is_empty()


def test_is_empty_when_all_terminated():
    d = DisjunctiveState.of([State.unreachable()])
    assert d.is_empty()
    assert d.collapse().reachable is False


def test_map_applies_to_each_reachable_path():
    d = DisjunctiveState.of([_state(n=int_const(2)), _state(n=int_const(3))])

    def bump(s):
        s2 = s.copy()
        s2.set("m", int_const(0))
        return s2

    out = d.map(bump)
    assert out.width() == 2
    assert all(s.get("m") is not None for s in out.disjuncts)


def test_flat_map_expands_paths():
    d = DisjunctiveState.singleton(_state(n=int_const(1)))
    out = d.flat_map(lambda s: [s.copy(), s.copy()])
    # the two copies are structurally equal ⇒ deduped back to one
    assert out.width() == 1


def test_join_unions_disjuncts():
    a = DisjunctiveState.of([_state(n=int_const(1))])
    b = DisjunctiveState.of([_state(n=int_const(2))])
    assert a.join(b).width() == 2


def test_join_respects_bound():
    a = DisjunctiveState.of([_state(n=int_const(i)) for i in range(3)], bound=4)
    b = DisjunctiveState.of([_state(n=int_const(i)) for i in range(3, 7)], bound=4)
    assert a.join(b).width() <= 1 or a.join(b).width() <= 4


# --------------------------------------------------------------------------
# integration: path precision attributable to disjunctive states
# --------------------------------------------------------------------------

_SRC = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"], flag):
    if flag:
        n = 2
    else:
        n = 3
    y = x.reshape(x.size(0), x.size(1), n)
    return y
"""


def _kinds_with_bound(src, bound):
    mod = ast.parse(src)
    interp = Interpreter(mod)
    interp._disj_bound = bound
    for node in mod.body:
        if isinstance(node, ast.FunctionDef):
            interp.run_function(node, args={}, self_val=None)
    return [b.kind for b in interp.bugs]


def test_disjunctive_analysis_catches_per_path_reshape():
    # With the disjunction kept apart, ``n`` is concrete (2 then 3) on each path
    # and the reshape ``[a, b] -> [a, b, n]`` is a forced mismatch on both.
    assert RESHAPE in [b.kind for b in analyze_source(_SRC).bugs]


def test_eager_join_masks_the_fault():
    # Forcing the width bound to 1 reproduces the old eager-join behaviour: ``n``
    # becomes a non-constant int after the merge, so the reshape detector
    # abstains — demonstrating the precision is attributable to Step 57.
    assert RESHAPE not in _kinds_with_bound(_SRC, bound=1)


def test_disjunctive_bound_recovers_the_fault():
    assert RESHAPE in _kinds_with_bound(_SRC, bound=8)


def test_no_false_positive_when_both_paths_safe():
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"], flag):
    if flag:
        n = 0
    else:
        n = 1
    y = x.reshape(x.size(0), x.size(1))
    return y
"""
    # A shape-preserving reshape is fine on every path: no report.
    assert RESHAPE not in [b.kind for b in analyze_source(src).bugs]
