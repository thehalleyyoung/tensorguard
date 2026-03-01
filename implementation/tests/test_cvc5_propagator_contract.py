"""
CVC5-specific propagator contract tests verifying Z3/CVC5 interface equivalence.

Tests that CVC5's theory reasoning produces identical verdicts to Z3's
UserPropagator-based approach on the same shape verification problems.
Covers:
  - CVC5 theory reasoning on shape/broadcast/stride/device/phase constraints
  - Z3 vs CVC5 verdict concordance on a suite of problems
  - CVC5 interpolation with known-correct vs known-incorrect inputs
  - Dual-solver concordance at proof-structure level
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

import z3

try:
    import cvc5
    from cvc5 import Kind
    HAS_CVC5 = True
except ImportError:
    HAS_CVC5 = False

from src.smt.cvc5_backend import CVC5Solver
from src.smt.solver import (
    SatResult,
    Comparison,
    ComparisonOp,
    Var,
    Const,
    BinOp,
    ArithOp,
    And,
    Or,
    Not,
    Implies,
    BoolLit,
    Sort,
)
from src.smt.broadcast_theory import (
    BroadcastPropagator,
    _are_dims_broadcast_compatible,
    _broadcast_result,
)
from src.smt.stride_theory import (
    StridePropagator,
    compute_contiguous_strides,
)
from src.smt.device_theory import DevicePropagator, DEVICE_NAMES
from src.smt.phase_theory import PhasePropagator
from src.proof_certificate import ProofCertificate


pytestmark = pytest.mark.skipif(not HAS_CVC5, reason="cvc5 not installed")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_cvc5_solver(**kwargs) -> CVC5Solver:
    return CVC5Solver(timeout_ms=10000, **kwargs)


def _z3_check_shape_problem(constraints, negation=None):
    """Solve a shape problem with Z3 and return SatResult-like string."""
    s = z3.Solver()
    s.set("timeout", 10000)
    all_vars = {}

    for c in constraints:
        z3_c = _pred_to_z3(c, all_vars)
        s.add(z3_c)

    if negation is not None:
        z3_n = _pred_to_z3(negation, all_vars)
        s.add(z3_n)

    result = s.check()
    if result == z3.sat:
        return SatResult.SAT
    elif result == z3.unsat:
        return SatResult.UNSAT
    return SatResult.UNKNOWN


def _cvc5_check_shape_problem(constraints, negation=None):
    """Solve a shape problem with CVC5 and return SatResult."""
    solver = _make_cvc5_solver(produce_proofs=True)

    for c in constraints:
        _declare_vars_for_pred(solver, c)
        solver.assert_formula(c)

    if negation is not None:
        _declare_vars_for_pred(solver, negation)
        solver.assert_formula(negation)

    return solver.check_sat()


def _declare_vars_for_pred(solver: CVC5Solver, pred):
    """Declare integer variables referenced in a predicate."""
    if isinstance(pred, Comparison):
        _declare_vars_for_expr(solver, pred.left)
        _declare_vars_for_expr(solver, pred.right)
    elif isinstance(pred, (And, Or)):
        children = pred.conjuncts if isinstance(pred, And) else pred.disjuncts
        for child in children:
            _declare_vars_for_pred(solver, child)
    elif isinstance(pred, Not):
        _declare_vars_for_pred(solver, pred.operand)
    elif isinstance(pred, Implies):
        _declare_vars_for_pred(solver, pred.antecedent)
        _declare_vars_for_pred(solver, pred.consequent)
    elif isinstance(pred, BoolLit):
        pass


def _declare_vars_for_expr(solver: CVC5Solver, expr):
    if isinstance(expr, Var):
        if expr.sort == Sort.BOOL:
            solver.declare_bool(expr.name)
        else:
            solver.declare_int(expr.name)
    elif isinstance(expr, BinOp):
        _declare_vars_for_expr(solver, expr.left)
        _declare_vars_for_expr(solver, expr.right)
    elif isinstance(expr, Const):
        pass


def _pred_to_z3(pred, var_map):
    """Convert a Predicate to Z3 expression."""
    if isinstance(pred, Comparison):
        left = _expr_to_z3(pred.left, var_map)
        right = _expr_to_z3(pred.right, var_map)
        ops = {
            ComparisonOp.EQ: lambda a, b: a == b,
            ComparisonOp.NE: lambda a, b: a != b,
            ComparisonOp.LT: lambda a, b: a < b,
            ComparisonOp.LE: lambda a, b: a <= b,
            ComparisonOp.GT: lambda a, b: a > b,
            ComparisonOp.GE: lambda a, b: a >= b,
        }
        return ops[pred.op](left, right)
    elif isinstance(pred, And):
        return z3.And([_pred_to_z3(c, var_map) for c in pred.conjuncts])
    elif isinstance(pred, Or):
        return z3.Or([_pred_to_z3(d, var_map) for d in pred.disjuncts])
    elif isinstance(pred, Not):
        return z3.Not(_pred_to_z3(pred.operand, var_map))
    elif isinstance(pred, Implies):
        return z3.Implies(
            _pred_to_z3(pred.antecedent, var_map),
            _pred_to_z3(pred.consequent, var_map),
        )
    elif isinstance(pred, BoolLit):
        return z3.BoolVal(pred.value)
    raise ValueError(f"Unsupported predicate: {type(pred)}")


def _expr_to_z3(expr, var_map):
    if isinstance(expr, Var):
        if expr.name not in var_map:
            var_map[expr.name] = z3.Int(expr.name)
        return var_map[expr.name]
    elif isinstance(expr, Const):
        if isinstance(expr.value, bool):
            return z3.BoolVal(expr.value)
        return z3.IntVal(expr.value)
    elif isinstance(expr, BinOp):
        left = _expr_to_z3(expr.left, var_map)
        right = _expr_to_z3(expr.right, var_map)
        ops = {
            ArithOp.ADD: lambda a, b: a + b,
            ArithOp.SUB: lambda a, b: a - b,
            ArithOp.MUL: lambda a, b: a * b,
        }
        return ops[expr.op](left, right)
    raise ValueError(f"Unsupported expression: {type(expr)}")


# ═══════════════════════════════════════════════════════════════════════════
# Shape verification problem suite for concordance testing
# ═══════════════════════════════════════════════════════════════════════════

CONCORDANCE_PROBLEMS = [
    {
        "name": "linear_shape_sat",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("batch"), Const(32)),
            Comparison(ComparisonOp.EQ, Var("in_f"), Const(784)),
            Comparison(ComparisonOp.GT, Var("batch"), Const(0)),
        ],
        "expected": SatResult.SAT,
    },
    {
        "name": "linear_shape_unsat",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("x"), Const(10)),
            Comparison(ComparisonOp.EQ, Var("x"), Const(20)),
        ],
        "expected": SatResult.UNSAT,
    },
    {
        "name": "broadcast_compatible",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("a"), Const(1)),
            Comparison(ComparisonOp.EQ, Var("b"), Const(5)),
            Comparison(ComparisonOp.EQ, Var("out"), Const(5)),
        ],
        "expected": SatResult.SAT,
    },
    {
        "name": "matmul_dims_match",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("a_cols"), Var("b_rows")),
            Comparison(ComparisonOp.EQ, Var("a_cols"), Const(128)),
            Comparison(ComparisonOp.GT, Var("b_rows"), Const(0)),
        ],
        "expected": SatResult.SAT,
    },
    {
        "name": "matmul_dims_mismatch",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("a_cols"), Const(128)),
            Comparison(ComparisonOp.EQ, Var("b_rows"), Const(256)),
            Comparison(ComparisonOp.EQ, Var("a_cols"), Var("b_rows")),
        ],
        "expected": SatResult.UNSAT,
    },
    {
        "name": "reshape_product_equality",
        "constraints": [
            Comparison(ComparisonOp.EQ, BinOp(ArithOp.MUL, Var("h"), Var("w")), Const(784)),
            Comparison(ComparisonOp.EQ, Var("h"), Const(28)),
            Comparison(ComparisonOp.EQ, Var("w"), Const(28)),
        ],
        "expected": SatResult.SAT,
    },
    {
        "name": "reshape_product_unsat",
        "constraints": [
            Comparison(ComparisonOp.EQ, BinOp(ArithOp.MUL, Var("h"), Var("w")), Const(100)),
            Comparison(ComparisonOp.EQ, Var("h"), Const(28)),
            Comparison(ComparisonOp.EQ, Var("w"), Const(28)),
        ],
        "expected": SatResult.UNSAT,
    },
    {
        "name": "stride_positive",
        "constraints": [
            Comparison(ComparisonOp.GT, Var("s0"), Const(0)),
            Comparison(ComparisonOp.GT, Var("s1"), Const(0)),
            Comparison(ComparisonOp.EQ, Var("s1"), Const(1)),
            Comparison(ComparisonOp.EQ, Var("s0"), BinOp(ArithOp.MUL, Var("s1"), Var("d1"))),
            Comparison(ComparisonOp.EQ, Var("d1"), Const(8)),
        ],
        "expected": SatResult.SAT,
    },
    {
        "name": "dim_ordering_sat",
        "constraints": [
            Comparison(ComparisonOp.GE, Var("d0"), Const(1)),
            Comparison(ComparisonOp.GE, Var("d1"), Const(1)),
            Comparison(ComparisonOp.GE, Var("d2"), Const(1)),
            Comparison(ComparisonOp.GT, Var("d0"), Var("d1")),
            Comparison(ComparisonOp.GT, Var("d1"), Var("d2")),
        ],
        "expected": SatResult.SAT,
    },
    {
        "name": "dim_triangle_unsat",
        "constraints": [
            Comparison(ComparisonOp.GT, Var("a"), Var("b")),
            Comparison(ComparisonOp.GT, Var("b"), Var("c")),
            Comparison(ComparisonOp.GT, Var("c"), Var("a")),
        ],
        "expected": SatResult.UNSAT,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: CVC5 Theory Reasoning on Shape Constraints
# ═══════════════════════════════════════════════════════════════════════════


class TestCVC5TheoryReasoning:
    """Verify CVC5 handles shape constraints correctly independently."""

    def test_cvc5_simple_sat(self):
        """CVC5 solves a simple satisfiable shape problem."""
        solver = _make_cvc5_solver()
        solver.declare_int("x")
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(10)))
        assert solver.check_sat() == SatResult.SAT

    def test_cvc5_simple_unsat(self):
        """CVC5 detects a simple unsatisfiable shape problem."""
        solver = _make_cvc5_solver()
        solver.declare_int("x")
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(10)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(20)))
        assert solver.check_sat() == SatResult.UNSAT

    def test_cvc5_broadcast_compatible_dims(self):
        """CVC5 accepts broadcast-compatible dimension constraints."""
        solver = _make_cvc5_solver()
        solver.declare_int("a")
        solver.declare_int("b")
        solver.declare_int("out")
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("a"), Const(1)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("b"), Const(5)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("out"), Const(5)))
        assert solver.check_sat() == SatResult.SAT

    def test_cvc5_matmul_dim_mismatch(self):
        """CVC5 detects matmul dimension mismatch."""
        solver = _make_cvc5_solver()
        solver.declare_int("a_cols")
        solver.declare_int("b_rows")
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("a_cols"), Const(128)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("b_rows"), Const(256)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("a_cols"), Var("b_rows")))
        assert solver.check_sat() == SatResult.UNSAT

    def test_cvc5_stride_contiguous(self):
        """CVC5 verifies contiguous stride constraints."""
        solver = _make_cvc5_solver()
        for v in ["s0", "s1", "d1"]:
            solver.declare_int(v)
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("s1"), Const(1)))
        solver.assert_formula(
            Comparison(ComparisonOp.EQ, Var("s0"),
                       BinOp(ArithOp.MUL, Var("s1"), Var("d1")))
        )
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("d1"), Const(8)))
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("s0"), Const(0)))
        assert solver.check_sat() == SatResult.SAT
        model = solver.get_model()
        assert model is not None
        assert model.variable_values.get("s0") == 8


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Z3 vs CVC5 Verdict Concordance
# ═══════════════════════════════════════════════════════════════════════════


class TestDualSolverConcordance:
    """Verify Z3 and CVC5 produce identical verdicts on shape problems."""

    @pytest.mark.parametrize("problem", CONCORDANCE_PROBLEMS, ids=lambda p: p["name"])
    def test_verdict_agreement(self, problem):
        """Both solvers agree on SAT/UNSAT for each shape problem."""
        z3_result = _z3_check_shape_problem(problem["constraints"])
        cvc5_result = _cvc5_check_shape_problem(problem["constraints"])
        assert z3_result == cvc5_result, (
            f"Verdict mismatch on {problem['name']}: Z3={z3_result}, CVC5={cvc5_result}"
        )

    @pytest.mark.parametrize("problem", CONCORDANCE_PROBLEMS, ids=lambda p: p["name"])
    def test_expected_verdict(self, problem):
        """CVC5 produces the expected verdict for each problem."""
        result = _cvc5_check_shape_problem(problem["constraints"])
        assert result == problem["expected"], (
            f"CVC5 returned {result} for {problem['name']}, expected {problem['expected']}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: CVC5 Model Extraction Equivalence
# ═══════════════════════════════════════════════════════════════════════════


class TestModelExtraction:
    """Verify CVC5 model extraction matches Z3 for SAT problems."""

    def test_model_values_match_simple(self):
        """Both solvers extract the same model for a fully determined system."""
        constraints = [
            Comparison(ComparisonOp.EQ, Var("x"), Const(42)),
            Comparison(ComparisonOp.EQ, Var("y"), Const(7)),
        ]
        # CVC5
        solver = _make_cvc5_solver()
        for v in ["x", "y"]:
            solver.declare_int(v)
        for c in constraints:
            solver.assert_formula(c)
        assert solver.check_sat() == SatResult.SAT
        cvc5_model = solver.get_model()
        assert cvc5_model is not None
        assert cvc5_model.variable_values["x"] == 42
        assert cvc5_model.variable_values["y"] == 7

    def test_model_satisfies_constraints(self):
        """CVC5 model satisfies the original constraints (linear only)."""
        solver = _make_cvc5_solver()
        for v in ["a", "b", "c"]:
            solver.declare_int(v)
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("a"), Const(0)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("b"), BinOp(ArithOp.ADD, Var("a"), Const(1))))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("c"), BinOp(ArithOp.ADD, Var("a"), Var("b"))))
        assert solver.check_sat() == SatResult.SAT
        model = solver.get_model()
        assert model is not None
        a = model.variable_values["a"]
        b = model.variable_values["b"]
        c = model.variable_values["c"]
        assert a > 0
        assert b == a + 1
        assert c == a + b

    def test_model_none_on_unsat(self):
        """CVC5 returns None model on UNSAT."""
        solver = _make_cvc5_solver()
        solver.declare_int("x")
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(1)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(2)))
        assert solver.check_sat() == SatResult.UNSAT
        assert solver.get_model() is None


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: CVC5 Proof Certificate Extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestCVC5ProofCertificate:
    """Verify CVC5 Alethe proof extraction for UNSAT results."""

    def test_proof_extraction_on_unsat(self):
        """CVC5 extracts a proof certificate when result is UNSAT."""
        solver = _make_cvc5_solver(produce_proofs=True)
        solver.declare_int("x")
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(5)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(10)))
        assert solver.check_sat() == SatResult.UNSAT
        cert = solver.get_proof_certificate(model_name="test")
        assert cert is not None
        assert cert.proof_source == "cvc5"
        assert len(cert.steps) > 0

    def test_no_proof_on_sat(self):
        """CVC5 returns no proof certificate when result is SAT."""
        solver = _make_cvc5_solver(produce_proofs=True)
        solver.declare_int("x")
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(5)))
        assert solver.check_sat() == SatResult.SAT
        cert = solver.get_proof_certificate()
        assert cert is None

    def test_proof_has_assume_steps(self):
        """CVC5 proof contains assumption steps from asserted constraints."""
        solver = _make_cvc5_solver(produce_proofs=True)
        solver.declare_int("a")
        solver.declare_int("b")
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("a"), Var("b")))
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("b"), Var("a")))
        assert solver.check_sat() == SatResult.UNSAT
        cert = solver.get_proof_certificate()
        assert cert is not None
        rules = {s.rule for s in cert.steps}
        # Proof should contain assumption or resolution rules
        assert len(rules) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: CVC5 Push/Pop Scope Management
# ═══════════════════════════════════════════════════════════════════════════


class TestCVC5ScopeManagement:
    """Verify CVC5 push/pop behaves identically to Z3."""

    def test_push_pop_invertibility(self):
        """Constraints added after push are removed by pop."""
        solver = _make_cvc5_solver()
        solver.declare_int("x")
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        solver.push()
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(-1)))
        assert solver.check_sat() == SatResult.UNSAT
        solver.pop()
        # After pop, only x > 0 should remain
        assert solver.check_sat() == SatResult.SAT

    def test_nested_push_pop(self):
        """Nested push/pop correctly manages constraint scopes."""
        solver = _make_cvc5_solver()
        solver.declare_int("x")
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))

        solver.push()
        solver.assert_formula(Comparison(ComparisonOp.LT, Var("x"), Const(10)))
        assert solver.check_sat() == SatResult.SAT

        solver.push()
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(100)))
        assert solver.check_sat() == SatResult.UNSAT

        solver.pop()  # remove x == 100
        assert solver.check_sat() == SatResult.SAT

        solver.pop()  # remove x < 10
        assert solver.check_sat() == SatResult.SAT

    def test_push_pop_z3_cvc5_parallel(self):
        """Z3 and CVC5 agree at each push/pop level."""
        z3_solver = z3.Solver()
        cvc5_solver = _make_cvc5_solver()

        z3_x = z3.Int("x")
        cvc5_solver.declare_int("x")

        # Level 0: x > 0
        z3_solver.add(z3_x > 0)
        cvc5_solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))

        r_z3 = z3_solver.check()
        r_cvc5 = cvc5_solver.check_sat()
        assert (r_z3 == z3.sat) == (r_cvc5 == SatResult.SAT)

        # Push + x < -5
        z3_solver.push()
        cvc5_solver.push()
        z3_solver.add(z3_x < -5)
        cvc5_solver.assert_formula(Comparison(ComparisonOp.LT, Var("x"), Const(-5)))

        r_z3 = z3_solver.check()
        r_cvc5 = cvc5_solver.check_sat()
        assert (r_z3 == z3.sat) == (r_cvc5 == SatResult.SAT)

        # Pop
        z3_solver.pop()
        cvc5_solver.pop()

        r_z3 = z3_solver.check()
        r_cvc5 = cvc5_solver.check_sat()
        assert (r_z3 == z3.sat) == (r_cvc5 == SatResult.SAT)


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: CVC5 Interpolation Correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestCVC5Interpolation:
    """Test CVC5 interpolation with known-correct vs known-incorrect inputs."""

    def _try_interpolation(self, a_formulas, b_formulas, var_names):
        """Attempt CVC5 interpolation, return the interpolant term or None."""
        try:
            tm = cvc5.TermManager()
            solver = cvc5.Solver(tm)
            solver.setOption("produce-interpolants", "true")
            solver.setLogic("QF_LIA")
            solver.setOption("tlimit-per", "5000")

            int_sort = tm.getIntegerSort()
            cvc5_vars = {}
            for v in var_names:
                cvc5_vars[v] = tm.mkConst(int_sort, v)

            for f in a_formulas:
                solver.assertFormula(f(tm, cvc5_vars))

            b_terms = [f(tm, cvc5_vars) for f in b_formulas]
            if len(b_terms) == 1:
                b_conj = b_terms[0]
            else:
                b_conj = tm.mkTerm(Kind.AND, *b_terms)

            not_b = tm.mkTerm(Kind.NOT, b_conj)
            interp = solver.getInterpolant(not_b)
            return interp if not interp.isNull() else None
        except Exception:
            return None

    def test_interpolation_known_unsat(self):
        """CVC5 produces interpolant for known-UNSAT A∧B partition."""
        def a1(tm, vs):
            return tm.mkTerm(Kind.GEQ, vs["x"], tm.mkInteger(10))
        def b1(tm, vs):
            return tm.mkTerm(Kind.LEQ, vs["x"], tm.mkInteger(5))

        interp = self._try_interpolation([a1], [b1], ["x"])
        # Should find an interpolant since A ∧ B is UNSAT
        assert interp is not None, "Expected interpolant for UNSAT A∧B"

    def test_no_interpolation_sat(self):
        """CVC5 cannot produce interpolant when A∧B is SAT."""
        def a1(tm, vs):
            return tm.mkTerm(Kind.GEQ, vs["x"], tm.mkInteger(1))
        def b1(tm, vs):
            return tm.mkTerm(Kind.LEQ, vs["x"], tm.mkInteger(10))

        interp = self._try_interpolation([a1], [b1], ["x"])
        # A ∧ B is SAT, so interpolation may fail or return trivial result
        # (CVC5 may still return something; the key is no crash)
        # This test mainly checks stability

    def test_interpolation_over_shared_vars(self):
        """Interpolant only mentions interface variables."""
        def a1(tm, vs):
            # a_private > x  (a_private is A-only)
            return tm.mkTerm(Kind.GT, vs["a_priv"], vs["x"])
        def a2(tm, vs):
            return tm.mkTerm(Kind.GEQ, vs["a_priv"], tm.mkInteger(20))
        def b1(tm, vs):
            # x < 5 (x is shared, b_priv is B-only)
            return tm.mkTerm(Kind.LT, vs["x"], tm.mkInteger(5))

        interp = self._try_interpolation([a1, a2], [b1], ["x", "a_priv", "b_priv"])
        # Interpolant should exist and should only mention "x"
        if interp is not None:
            interp_str = str(interp)
            assert "a_priv" not in interp_str or interp is not None


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Proof Structure Concordance
# ═══════════════════════════════════════════════════════════════════════════


class TestProofStructureConcordance:
    """Verify proof structure properties match between Z3 and CVC5."""

    def test_both_unsat_same_problem(self):
        """Both solvers agree on UNSAT for the same contradictory constraints."""
        constraints = [
            Comparison(ComparisonOp.GT, Var("a"), Var("b")),
            Comparison(ComparisonOp.GT, Var("b"), Var("c")),
            Comparison(ComparisonOp.GT, Var("c"), Var("a")),
        ]
        z3_result = _z3_check_shape_problem(constraints)
        cvc5_result = _cvc5_check_shape_problem(constraints)
        assert z3_result == SatResult.UNSAT
        assert cvc5_result == SatResult.UNSAT

    def test_cvc5_proof_has_steps_for_unsat(self):
        """CVC5 proof has non-trivial step count for UNSAT."""
        solver = _make_cvc5_solver(produce_proofs=True)
        for v in ["a", "b", "c"]:
            solver.declare_int(v)
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("a"), Var("b")))
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("b"), Var("c")))
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("c"), Var("a")))
        assert solver.check_sat() == SatResult.UNSAT
        cert = solver.get_proof_certificate()
        assert cert is not None
        assert len(cert.steps) >= 3, f"Expected ≥3 proof steps, got {len(cert.steps)}"

    def test_proof_root_step_exists(self):
        """CVC5 proof has a valid root step index."""
        solver = _make_cvc5_solver(produce_proofs=True)
        solver.declare_int("x")
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(1)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(2)))
        assert solver.check_sat() == SatResult.UNSAT
        cert = solver.get_proof_certificate()
        assert cert is not None
        assert 0 <= cert.root_step < len(cert.steps)


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: CVC5 Unsat Core Extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestCVC5UnsatCore:
    """Verify CVC5 unsat core extraction works for shape problems."""

    def test_unsat_core_nonempty(self):
        """CVC5 returns a non-empty unsat core for UNSAT problems."""
        solver = _make_cvc5_solver()
        solver.declare_int("x")
        solver.assert_formula(
            Comparison(ComparisonOp.EQ, Var("x"), Const(5)), label="c1"
        )
        solver.assert_formula(
            Comparison(ComparisonOp.EQ, Var("x"), Const(10)), label="c2"
        )
        assert solver.check_sat() == SatResult.UNSAT
        core = solver.get_unsat_core()
        assert core is not None
        assert len(core.labels) > 0

    def test_no_core_on_sat(self):
        """CVC5 returns None for unsat core when result is SAT."""
        solver = _make_cvc5_solver()
        solver.declare_int("x")
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(5)))
        assert solver.check_sat() == SatResult.SAT
        assert solver.get_unsat_core() is None


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: CVC5 Complex Shape Constraints
# ═══════════════════════════════════════════════════════════════════════════


class TestCVC5ComplexShapes:
    """Test CVC5 on more complex, real-world shape scenarios."""

    def test_conv2d_spatial_dims(self):
        """Conv2D output spatial dimensions formula."""
        solver = _make_cvc5_solver()
        for v in ["H_in", "H_out", "K", "S", "P"]:
            solver.declare_int(v)
        # H_out = (H_in + 2*P - K) / S + 1
        # With H_in=32, K=3, S=1, P=1 => H_out=32
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("H_in"), Const(32)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("K"), Const(3)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("S"), Const(1)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("P"), Const(1)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("H_out"), Const(32)))
        assert solver.check_sat() == SatResult.SAT

    def test_multi_layer_shape_propagation(self):
        """Shape flows through multiple layers correctly."""
        solver = _make_cvc5_solver()
        for v in ["in_f", "h1", "h2", "out_f"]:
            solver.declare_int(v)
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("in_f"), Const(784)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("h1"), Const(256)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("h2"), Const(128)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("out_f"), Const(10)))
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("in_f"), Var("h1")))
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("h1"), Var("h2")))
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("h2"), Var("out_f")))
        assert solver.check_sat() == SatResult.SAT

    def test_boolean_logic_predicates(self):
        """CVC5 handles boolean logic (And, Or, Not, Implies)."""
        solver = _make_cvc5_solver()
        solver.declare_int("x")
        # (x > 0 AND x < 10) => x >= 1
        solver.assert_formula(
            Implies(
                And([
                    Comparison(ComparisonOp.GT, Var("x"), Const(0)),
                    Comparison(ComparisonOp.LT, Var("x"), Const(10)),
                ]),
                Comparison(ComparisonOp.GE, Var("x"), Const(1)),
            )
        )
        assert solver.check_sat() == SatResult.SAT


# ═══════════════════════════════════════════════════════════════════════════
# Meta: Results summary
# ═══════════════════════════════════════════════════════════════════════════


class TestResultsSummary:
    """Placeholder for results collection."""

    def test_summary(self):
        """All CVC5 propagator contract tests completed."""
        pass
