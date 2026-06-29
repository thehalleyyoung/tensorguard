"""Step 51 — Z3 bridge for ``SymDim``.

Lowering affine dimension expressions and dimension constraints into Z3 integer
arithmetic, and answering satisfiability soundly (only a proved ``unsat`` is
ever reported; everything else, including a missing solver, is ``unknown``).
"""

import importlib

import pytest

from src.symexec.symdim import SymDim
from src.symexec import smt_bridge as B


def _v(name):
    return SymDim.var(name)


# --------------------------------------------------------------------------
# constraint construction / validation
# --------------------------------------------------------------------------

def test_relational_constructors_round_trip():
    b = _v("b")
    assert B.eq(b, 4).op == "=="
    assert B.ne(b, 4).op == "!="
    assert B.lt(b, 4).op == "<"
    assert B.le(b, 4).op == "<="
    assert B.gt(b, 4).op == ">"
    assert B.ge(b, 4).op == ">="


def test_divisibility_constructors():
    b = _v("b")
    assert B.divisible(b, 2).op == "%==0"
    assert B.not_divisible(b, 2).op == "%!=0"


def test_zero_modulus_rejected():
    with pytest.raises(ValueError):
        B.divisible(_v("b"), 0)


def test_unknown_op_rejected():
    with pytest.raises(ValueError):
        B.DimConstraint(_v("b"), "~=", _v("c"))


# --------------------------------------------------------------------------
# satisfiability — requires z3
# --------------------------------------------------------------------------

z3only = pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")


@z3only
def test_eq_const_is_sat():
    assert B.check([B.eq(_v("b"), 4)]) == "sat"


@z3only
def test_eq_and_ne_same_pair_is_unsat():
    b, c = _v("b"), _v("c")
    assert B.check([B.eq(b, c), B.ne(b, c)]) == "unsat"


@z3only
def test_const_mismatch_is_unsat():
    # b == 4 and b == 5 cannot both hold.
    b = _v("b")
    assert B.check([B.eq(b, 4), B.eq(b, 5)]) == "unsat"


@z3only
def test_affine_coupling_is_exact():
    # 2*b and b+b are the same expression; (2*b != b+b) is UNSAT.
    b = _v("b")
    assert B.check([B.ne(b * 2, b + b)]) == "unsat"


@z3only
def test_shared_variable_links_constraints():
    # b == c, c == 4  ⇒  b is forced to 4, so b == 5 is then UNSAT.
    b, c = _v("b"), _v("c")
    assert B.check([B.eq(b, c), B.eq(c, 4), B.eq(b, 5)]) == "unsat"
    assert B.check([B.eq(b, c), B.eq(c, 4), B.eq(b, 4)]) == "sat"


@z3only
def test_divisibility_contradiction_is_unsat():
    b = _v("b")
    assert B.check([B.divisible(b, 2), B.not_divisible(b, 2)]) == "unsat"


@z3only
def test_concrete_value_violating_divisibility_is_unsat():
    b = _v("b")
    assert B.check([B.eq(b, 3), B.divisible(b, 2)]) == "unsat"
    assert B.check([B.eq(b, 4), B.divisible(b, 2)]) == "sat"


# --------------------------------------------------------------------------
# well-formedness floor
# --------------------------------------------------------------------------

@z3only
def test_default_floor_forbids_zero_dim():
    # default positive_floor == 1, so b == 0 is unsatisfiable.
    assert B.check([B.eq(_v("b"), 0)]) == "unsat"


@z3only
def test_floor_zero_allows_zero_dim():
    assert B.check([B.eq(_v("b"), 0)], positive_floor=0) == "sat"


@z3only
def test_floor_makes_below_floor_unsat():
    # b < 1 is UNSAT at floor 1 but SAT (b == 0) at floor 0.
    assert B.check([B.lt(_v("b"), 1)]) == "unsat"
    assert B.check([B.lt(_v("b"), 1)], positive_floor=0) == "sat"


# --------------------------------------------------------------------------
# model extraction (for counterexample lifting, Step 54)
# --------------------------------------------------------------------------

@z3only
def test_model_satisfies_constraints():
    b, c = _v("b"), _v("c")
    m = B.model([B.eq(b, 4), B.eq(c, b + 1)])
    assert m is not None
    assert m["b"] == 4
    assert m["c"] == 5


@z3only
def test_model_none_when_unsat():
    b = _v("b")
    assert B.model([B.eq(b, 1), B.eq(b, 2)]) is None


@z3only
def test_model_respects_floor():
    # with floor 1 the only constraint b <= 1 forces b == 1.
    m = B.model([B.le(_v("b"), 1)])
    assert m is not None and m["b"] == 1


# --------------------------------------------------------------------------
# feasibility primitive (sound suppression for Step 52)
# --------------------------------------------------------------------------

@z3only
def test_feasible_true_for_sat():
    assert B.feasible([B.eq(_v("b"), 4)]) is True


@z3only
def test_feasible_false_only_for_proved_unsat():
    b, c = _v("b"), _v("c")
    assert B.feasible([B.eq(b, c), B.ne(b, c)]) is False


# --------------------------------------------------------------------------
# soundness: no fabricated unsat without a solver
# --------------------------------------------------------------------------

def test_check_is_unknown_without_z3(monkeypatch):
    monkeypatch.setattr(B, "Z3_AVAILABLE", False)
    # Even an obvious contradiction must degrade to "unknown" (never "unsat")
    # so a feasibility gate cannot suppress a real bug it cannot disprove.
    b, c = _v("b"), _v("c")
    assert B.check([B.eq(b, c), B.ne(b, c)]) == "unknown"


def test_feasible_is_true_without_z3(monkeypatch):
    monkeypatch.setattr(B, "Z3_AVAILABLE", False)
    b, c = _v("b"), _v("c")
    # unknown ⇒ keep the report ⇒ feasible() is True.
    assert B.feasible([B.eq(b, c), B.ne(b, c)]) is True


def test_model_is_none_without_z3(monkeypatch):
    monkeypatch.setattr(B, "Z3_AVAILABLE", False)
    assert B.model([B.eq(_v("b"), 4)]) is None
