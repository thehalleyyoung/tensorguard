"""Step 56 — relational domains (optional octagon / polyhedra for coupled dims).

A :class:`RelationalDomain` is a conjunction of dimension constraints with a
*semantic* join: a constraint survives the merge of two branches iff **both**
branches entail it (proved via Z3), drawing candidates from the union of the two
branches' stated constraints.  This is sound (a kept fact holds on each path)
and strictly more precise than a syntactic intersection, which only retains
literally-shared constraints.

Soundness floor: with z3 unavailable ``entails`` degrades to membership, so the
join reduces to the old syntactic intersection — never to something stronger.
"""

import pytest

from src.symexec import smt_bridge as B
from src.symexec.relational import RelationalDomain, join_facts, meet_facts
from src.symexec.state import State
from src.symexec.symdim import SymDim

z3only = pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")

a, b, c, d = (SymDim.var(n) for n in "abcd")


def _dom(*cs):
    return RelationalDomain.of(cs)


# --------------------------------------------------------------------------
# smt_bridge.negate
# --------------------------------------------------------------------------

def test_negate_inverts_each_operator():
    assert B.negate(B.eq(a, b)) == B.ne(a, b)
    assert B.negate(B.lt(a, b)) == B.ge(a, b)
    assert B.negate(B.le(a, b)) == B.gt(a, b)
    assert B.negate(B.divisible(a, 4)).op == "%!=0"


# --------------------------------------------------------------------------
# entailment
# --------------------------------------------------------------------------

def test_entails_membership_without_solver():
    dom = _dom(B.eq(a, b))
    assert dom.entails(B.eq(a, b))  # syntactic membership, no z3 needed


@z3only
def test_entails_implied_constraint():
    # a == b  ⇒  a <= b   and   b <= a
    dom = _dom(B.eq(a, b))
    assert dom.entails(B.le(a, b))
    assert dom.entails(B.le(b, a))


@z3only
def test_entails_transitive():
    # a <= b, b <= c  ⇒  a <= c
    dom = _dom(B.le(a, b), B.le(b, c))
    assert dom.entails(B.le(a, c))


@z3only
def test_does_not_entail_unrelated():
    dom = _dom(B.eq(a, b))
    assert not dom.entails(B.eq(c, d))


def test_entails_without_z3_is_membership_only(monkeypatch):
    monkeypatch.setattr(B, "Z3_AVAILABLE", False)
    dom = _dom(B.eq(a, b))
    assert dom.entails(B.eq(a, b))
    assert not dom.entails(B.le(a, b))  # cannot prove without a solver


# --------------------------------------------------------------------------
# bottom detection
# --------------------------------------------------------------------------

@z3only
def test_is_bottom_on_contradiction():
    assert _dom(B.ne(a, b), B.eq(a, b)).is_bottom()


def test_top_is_not_bottom():
    assert RelationalDomain.top().is_top()
    assert not RelationalDomain.top().is_bottom()


# --------------------------------------------------------------------------
# join — the key precision win
# --------------------------------------------------------------------------

@z3only
def test_join_keeps_differently_expressed_equivalent_fact():
    # Branch 1 states a == b; branch 2 states a <= b AND a >= b (same meaning).
    # A syntactic intersection loses everything; the semantic join keeps the
    # equality (each branch entails all three candidate constraints).
    j = _dom(B.eq(a, b)).join(_dom(B.le(a, b), B.ge(a, b)))
    assert j.entails(B.eq(a, b))
    assert not j.is_top()


@z3only
def test_join_drops_one_sided_unentailed_fact():
    j = _dom(B.eq(a, b), B.eq(c, d)).join(_dom(B.eq(a, b)))
    assert B.eq(a, b) in j.constraints
    assert B.eq(c, d) not in j.constraints  # only on one branch, not entailed


def test_join_with_top_is_top():
    assert _dom(B.eq(a, b)).join(RelationalDomain.top()).is_top()


def test_join_identical_is_idempotent():
    dom = _dom(B.eq(a, b), B.le(b, c))
    assert dom.join(dom).constraints == dom.constraints


def test_join_without_z3_is_syntactic_intersection(monkeypatch):
    monkeypatch.setattr(B, "Z3_AVAILABLE", False)
    j = _dom(B.eq(a, b), B.eq(b, c)).join(_dom(B.eq(a, b), B.eq(c, d)))
    assert set(j.constraints) == {B.eq(a, b)}  # only literally-shared survives


# --------------------------------------------------------------------------
# meet & widen
# --------------------------------------------------------------------------

def test_meet_is_conjunction():
    m = _dom(B.eq(a, b)).meet(_dom(B.le(b, c)))
    assert set(m.constraints) == {B.eq(a, b), B.le(b, c)}


@z3only
def test_widen_keeps_only_stable_constraints():
    # self has {a == b, b == c}; other only entails a == b ⇒ b == c is dropped.
    w = _dom(B.eq(a, b), B.eq(b, c)).widen(_dom(B.eq(a, b)))
    assert B.eq(a, b) in w.constraints
    assert B.eq(b, c) not in w.constraints


def test_widen_terminates_by_shrinking():
    # Widening can only drop constraints, so iterating reaches a fixpoint.
    cur = _dom(B.eq(a, b), B.eq(b, c), B.le(c, d))
    prev = None
    for _ in range(10):
        nxt = cur.widen(RelationalDomain.top())
        if prev is not None and nxt.constraints == prev.constraints:
            break
        prev, cur = nxt, nxt
    assert cur.is_top()  # widening against Top removes everything


# --------------------------------------------------------------------------
# tuple helpers used by State._merge
# --------------------------------------------------------------------------

@z3only
def test_join_facts_helper_matches_domain():
    got = join_facts((B.eq(a, b), B.eq(b, c)), (B.le(a, b), B.ge(a, b)))
    assert B.eq(a, b) in got or RelationalDomain(got).entails(B.eq(a, b))


def test_meet_facts_helper_unions():
    got = meet_facts((B.eq(a, b),), (B.le(b, c),))
    assert set(got) == {B.eq(a, b), B.le(b, c)}


# --------------------------------------------------------------------------
# integration: State.join uses the relational join
# --------------------------------------------------------------------------

@z3only
def test_state_join_keeps_semantically_common_fact():
    s1 = State()
    s1.dim_facts = (B.eq(a, b),)
    s2 = State()
    s2.dim_facts = (B.le(a, b), B.ge(a, b))
    merged = s1.join(s2)
    assert RelationalDomain(merged.dim_facts).entails(B.eq(a, b))
