"""Differential bridge: the relational / SMT-feasibility layer
(``smt_bridge.py`` + ``relational.py``) obeys the soundness properties that are
machine-checked in ``lean/TensorGuard/Symexec/Relational.lean``.

The Lean module proves, over the concrete semantics "evaluate every affine form
under an integer assignment and check the relation":

  * ``negate_sound``        — ``negate(c)`` holds iff ``c`` fails, on every env;
  * ``meet_sound``          — ``meet`` is exactly conjunction of the two domains;
  * ``entails_of_unsat``    — if ``facts ∧ ¬c`` has no model then ``facts ⟹ c``;
  * ``join_sound_left/right`` — ``join`` over-approximates each branch;
  * ``widen_sound``         — ``widen`` only drops constraints (γ can only grow).

These tests pin the *Python* implementation to that proven model by
brute-forcing many concrete assignments, so the code cannot silently drift from
the property the Lean kernel certified.  Z3-backed checks are gated on
``Z3_AVAILABLE`` and use the engine's own ``positive_floor`` (dimension
variables ``>= 1``) so the brute-force universe matches the solver's.
"""

from __future__ import annotations

import itertools

import pytest

from src.symexec import smt_bridge as smt
from src.symexec.relational import RelationalDomain, join_facts, meet_facts
from src.symexec.smt_bridge import DimConstraint, negate
from src.symexec.symdim import SymDim

_VARS = ["b", "s"]
# Dimension variables are >= 1 (the engine's positive_floor), so the empirical
# universe matches what the Z3 feasibility gate quantifies over.
_ENVS = [
    dict(zip(_VARS, combo)) for combo in itertools.product(range(1, 7), repeat=2)
]


def _eval(d: SymDim, env: dict) -> int:
    return d.const + sum(coeff * env[name] for name, coeff in d.terms)


def _sat(c: DimConstraint, env: dict) -> bool:
    """Concrete satisfaction of a constraint — the Python mirror of Lean ``sat``."""
    lhs = _eval(c.lhs, env)
    if c.op in ("%==0", "%!=0"):
        ok = lhs % c.rhs == 0
        return ok if c.op == "%==0" else not ok
    rhs = _eval(c.rhs if isinstance(c.rhs, SymDim) else SymDim.const_dim(c.rhs), env)
    return {
        "==": lhs == rhs,
        "!=": lhs != rhs,
        "<": lhs < rhs,
        "<=": lhs <= rhs,
        ">": lhs > rhs,
        ">=": lhs >= rhs,
    }[c.op]


def _models(constraints, env: dict) -> bool:
    return all(_sat(c, env) for c in constraints)


def _constraints():
    b, s = SymDim.var("b"), SymDim.var("s")
    return [
        smt.eq(b, s),
        smt.ne(b, s),
        smt.lt(b, s),
        smt.le(b, s),
        smt.gt(b, s),
        smt.ge(b, s),
        smt.eq(b, 4),
        smt.le(b + s, 8),
        smt.divisible(b * 2, 2),
        smt.not_divisible(b * 2 + 1, 2),
        smt.divisible(b, 3),
    ]


def test_negate_is_exact_complement():
    """negate_sound: sat(negate(c), env) == (not sat(c, env)) on every env."""
    for c in _constraints():
        nc = negate(c)
        for env in _ENVS:
            assert _sat(nc, env) == (not _sat(c, env)), (
                f"negate({c.op}) is not the exact complement at {env}"
            )


def test_negate_negate_is_identity():
    """negate_negate: double negation returns the original semantics."""
    for c in _constraints():
        nnc = negate(negate(c))
        for env in _ENVS:
            assert _sat(nnc, env) == _sat(c, env)


def test_meet_is_exactly_conjunction():
    """meet_sound: models(meet(A,B)) iff models(A) and models(B)."""
    b, s = SymDim.var("b"), SymDim.var("s")
    A = (smt.le(b, s),)
    B = (smt.le(s, b),)
    merged = meet_facts(A, B)
    for env in _ENVS:
        assert _models(merged, env) == (_models(A, env) and _models(B, env))


@pytest.mark.skipif(not smt.Z3_AVAILABLE, reason="z3 not installed")
def test_entails_reduction_is_sound():
    """entails_of_unsat: whenever the domain entails c (via the facts ∧ ¬c unsat
    reduction), every concrete model of the domain satisfies c."""
    b, s = SymDim.var("b"), SymDim.var("s")
    dom = RelationalDomain.of((smt.eq(b, s), smt.le(s, 4)))
    # b == s ∧ s <= 4 entails b <= 4 and b == s; both must hold on every model.
    for c in (smt.le(b, 4), smt.eq(b, s), smt.le(b, s)):
        if dom.entails(c):
            for env in _ENVS:
                if _models(dom.constraints, env):
                    assert _sat(c, env), (
                        f"entailed {c.op} violated by model {env}"
                    )


@pytest.mark.skipif(not smt.Z3_AVAILABLE, reason="z3 not installed")
def test_join_over_approximates_both_branches():
    """join_sound: every kept constraint is entailed by both operands, so every
    model of either branch models the join."""
    b, s = SymDim.var("b"), SymDim.var("s")
    # Branch A states a==b directly; branch B states it as a<=b ∧ a>=b.
    A = RelationalDomain.of((smt.eq(b, s),))
    B = RelationalDomain.of((smt.le(b, s), smt.ge(b, s)))
    merged = join_facts(A.constraints, B.constraints)
    for env in _ENVS:
        if _models(A.constraints, env):
            assert _models(merged, env), f"join lost an A-model at {env}"
        if _models(B.constraints, env):
            assert _models(merged, env), f"join lost a B-model at {env}"


@pytest.mark.skipif(not smt.Z3_AVAILABLE, reason="z3 not installed")
def test_join_recovers_semantically_equal_facts():
    """Non-vacuity: the *semantic* join keeps a==b even though one branch only
    states it as the pair (a<=b, a>=b) — a syntactic intersection would lose it."""
    b, s = SymDim.var("b"), SymDim.var("s")
    A = RelationalDomain.of((smt.eq(b, s),))
    B = RelationalDomain.of((smt.le(b, s), smt.ge(b, s)))
    merged = join_facts(A.constraints, B.constraints)
    # On every model the merged domain forces b == s.
    for env in _ENVS:
        if _models(merged, env):
            assert env["b"] == env["s"]
    assert merged, "join collapsed to top, losing the shared equality"


def test_widen_only_drops_constraints():
    """widen_sound: widening keeps a subset of self's constraints, so every model
    of self still models the widened domain (γ can only grow)."""
    b, s = SymDim.var("b"), SymDim.var("s")
    self_dom = RelationalDomain.of((smt.le(b, s), smt.eq(b, 4)))
    other = RelationalDomain.of((smt.le(b, s),))
    widened = self_dom.widen(other)
    assert set(widened.constraints).issubset(set(self_dom.constraints))
    for env in _ENVS:
        if _models(self_dom.constraints, env):
            assert _models(widened.constraints, env)
