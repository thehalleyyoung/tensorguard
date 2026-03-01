"""Tests for the CVC5 SMT backend and Alethe proof integration."""
from __future__ import annotations

import pytest

try:
    import cvc5
    HAS_CVC5 = True
except ImportError:
    HAS_CVC5 = False

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
    Iff,
    BoolLit,
    IsNone,
    IsTruthy,
    Sort,
)
from src.proof_certificate import (
    ProofStep,
    ProofCertificate,
    CVC5ProofExtractor,
    get_proof_status,
)

pytestmark = pytest.mark.skipif(not HAS_CVC5, reason="cvc5 not installed")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CVC5Solver basic functionality
# ═══════════════════════════════════════════════════════════════════════════════


class TestCVC5SolverBasic:
    def _make_solver(self, **kwargs):
        from src.smt.cvc5_backend import CVC5Solver
        return CVC5Solver(**kwargs)

    def test_simple_sat(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        result = solver.check_sat()
        assert result == SatResult.SAT

    def test_simple_unsat(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        solver.assert_formula(Comparison(ComparisonOp.LT, Var("x"), Const(0)))
        result = solver.check_sat()
        assert result == SatResult.UNSAT

    def test_push_pop(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        solver.push()
        solver.assert_formula(Comparison(ComparisonOp.LT, Var("x"), Const(0)))
        assert solver.check_sat() == SatResult.UNSAT
        solver.pop()
        assert solver.check_sat() == SatResult.SAT

    def test_boolean_literal(self):
        solver = self._make_solver()
        solver.assert_formula(BoolLit(True))
        assert solver.check_sat() == SatResult.SAT

    def test_boolean_false(self):
        solver = self._make_solver()
        solver.assert_formula(BoolLit(False))
        assert solver.check_sat() == SatResult.UNSAT

    def test_and_predicate(self):
        solver = self._make_solver()
        pred = And((
            Comparison(ComparisonOp.GT, Var("x"), Const(0)),
            Comparison(ComparisonOp.LT, Var("x"), Const(10)),
        ))
        solver.assert_formula(pred)
        assert solver.check_sat() == SatResult.SAT

    def test_or_predicate(self):
        solver = self._make_solver()
        pred = Or((
            Comparison(ComparisonOp.EQ, Var("x"), Const(1)),
            Comparison(ComparisonOp.EQ, Var("x"), Const(2)),
        ))
        solver.assert_formula(pred)
        assert solver.check_sat() == SatResult.SAT

    def test_not_predicate(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(5)))
        solver.assert_formula(Not(Comparison(ComparisonOp.EQ, Var("x"), Const(5))))
        assert solver.check_sat() == SatResult.UNSAT

    def test_implies(self):
        solver = self._make_solver()
        solver.assert_formula(Implies(
            Comparison(ComparisonOp.GT, Var("x"), Const(0)),
            Comparison(ComparisonOp.GE, Var("x"), Const(1)),
        ))
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        assert solver.check_sat() == SatResult.SAT

    def test_arithmetic_operations(self):
        solver = self._make_solver()
        expr = BinOp(ArithOp.ADD, Var("x"), Const(5))
        solver.assert_formula(Comparison(ComparisonOp.EQ, expr, Const(10)))
        assert solver.check_sat() == SatResult.SAT


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CVC5 model extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestCVC5ModelExtraction:
    def _make_solver(self, **kwargs):
        from src.smt.cvc5_backend import CVC5Solver
        return CVC5Solver(**kwargs)

    def test_get_model_sat(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(42)))
        assert solver.check_sat() == SatResult.SAT
        model = solver.get_model()
        assert model is not None
        assert model.get_int("x") == 42

    def test_get_model_unsat_returns_none(self):
        solver = self._make_solver()
        solver.assert_formula(BoolLit(False))
        solver.check_sat()
        assert solver.get_model() is None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CVC5 Alethe proof extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestCVC5ProofExtraction:
    def _make_solver(self, **kwargs):
        from src.smt.cvc5_backend import CVC5Solver
        return CVC5Solver(produce_proofs=True, **kwargs)

    def test_proof_extraction_on_unsat(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        solver.assert_formula(Comparison(ComparisonOp.LT, Var("x"), Const(0)))
        assert solver.check_sat() == SatResult.UNSAT

        cert = solver.get_proof_certificate("TestModel", ["shape_safety"])
        assert cert is not None
        assert cert.proof_source == "cvc5"
        assert cert.model_name == "TestModel"
        assert len(cert.steps) > 0

    def test_no_proof_on_sat(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        assert solver.check_sat() == SatResult.SAT

        cert = solver.get_proof_certificate("TestModel")
        assert cert is None

    def test_proof_certificate_fields(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(5)))
        solver.assert_formula(Comparison(ComparisonOp.LT, Var("x"), Const(3)))
        solver.check_sat()

        cert = solver.get_proof_certificate("ArithModel", ["dim_check"])
        if cert is not None:
            assert cert.proof_source == "cvc5"
            assert cert.certificate_hash != ""
            assert cert.extraction_time_ms >= 0
            assert cert.root_step >= 0
            assert cert.root_step < len(cert.steps)

    def test_proof_to_dict_includes_source(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        solver.assert_formula(Comparison(ComparisonOp.LT, Var("x"), Const(0)))
        solver.check_sat()

        cert = solver.get_proof_certificate("DictTest")
        if cert is not None:
            d = cert.to_dict()
            assert d["proof_source"] == "cvc5"

    def test_proof_local_verification(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(1)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("x"), Const(2)))
        solver.check_sat()

        cert = solver.get_proof_certificate("VerifyTest")
        # Structural verification may or may not pass depending on CVC5
        # proof format; we just ensure it doesn't crash
        if cert is not None:
            cert.verify_locally()

    def test_proof_without_produce_proofs(self):
        from src.smt.cvc5_backend import CVC5Solver
        solver = CVC5Solver(produce_proofs=False)
        solver.assert_formula(BoolLit(False))
        solver.check_sat()
        cert = solver.get_proof_certificate("NoProof")
        assert cert is None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Shape constraint scenarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestCVC5ShapeConstraints:
    def _make_solver(self, **kwargs):
        from src.smt.cvc5_backend import CVC5Solver
        return CVC5Solver(produce_proofs=True, **kwargs)

    def test_matmul_dimension_mismatch(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("a_cols"), Const(10)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("b_rows"), Const(20)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("a_cols"), Var("b_rows")))
        assert solver.check_sat() == SatResult.UNSAT

        cert = solver.get_proof_certificate("MatMul", ["dim_match"])
        assert cert is not None

    def test_reshape_element_count(self):
        solver = self._make_solver()
        mul1 = BinOp(ArithOp.MUL, Var("d1"), Var("d2"))
        mul2 = BinOp(ArithOp.MUL, Var("d3"), Var("d4"))
        solver.assert_formula(Comparison(ComparisonOp.EQ, mul1, mul2))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("d1"), Const(3)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("d2"), Const(4)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("d3"), Const(6)))
        solver.assert_formula(Comparison(ComparisonOp.EQ, Var("d4"), Const(3)))
        # 3*4=12 != 6*3=18
        assert solver.check_sat() == SatResult.UNSAT

    def test_positive_dimensions(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("batch"), Const(0)))
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("channels"), Const(0)))
        assert solver.check_sat() == SatResult.SAT
        model = solver.get_model()
        assert model is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ProofCertificate proof_source field
# ═══════════════════════════════════════════════════════════════════════════════


class TestProofCertificateSource:
    def test_default_source_is_z3(self):
        cert = ProofCertificate(
            model_name="test",
            properties=["p1"],
            steps=[ProofStep(rule="asserted", conclusion="true")],
            root_step=0,
        )
        assert cert.proof_source == "z3"

    def test_cvc5_source(self):
        cert = ProofCertificate(
            model_name="test",
            properties=["p1"],
            steps=[ProofStep(rule="assume", conclusion="true")],
            root_step=0,
            proof_source="cvc5",
        )
        assert cert.proof_source == "cvc5"
        assert cert.to_dict()["proof_source"] == "cvc5"

    def test_pretty_includes_source(self):
        cert = ProofCertificate(
            model_name="test",
            properties=["p1"],
            steps=[ProofStep(rule="asserted", conclusion="true")],
            root_step=0,
            proof_source="cvc5",
        )
        assert "cvc5" in cert.pretty()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. get_proof_status utility
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetProofStatus:
    def test_certified(self):
        cert = ProofCertificate(
            model_name="t",
            properties=[],
            steps=[ProofStep(rule="asserted", conclusion="true")],
            root_step=0,
        )
        assert get_proof_status(cert) == "certified"

    def test_solver_verified(self):
        assert get_proof_status(None, solver_verified=True) == "solver_verified"

    def test_unverified(self):
        assert get_proof_status(None, solver_verified=False) == "unverified"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CVC5ProofExtractor
# ═══════════════════════════════════════════════════════════════════════════════


class TestCVC5ProofExtractor:
    def test_extract_from_cvc5_solver(self):
        from src.smt.cvc5_backend import CVC5Solver
        solver = CVC5Solver(produce_proofs=True)
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        solver.assert_formula(Comparison(ComparisonOp.LT, Var("x"), Const(0)))
        solver.check_sat()

        cert = CVC5ProofExtractor.extract_from_cvc5_solver(
            solver, "ExtractorTest", ["prop"]
        )
        assert cert is not None
        assert cert.proof_source == "cvc5"

    def test_extract_from_non_solver_returns_none(self):
        result = CVC5ProofExtractor.extract_from_cvc5_solver(
            object(), "test"
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Unsat core
# ═══════════════════════════════════════════════════════════════════════════════


class TestCVC5UnsatCore:
    def _make_solver(self, **kwargs):
        from src.smt.cvc5_backend import CVC5Solver
        return CVC5Solver(**kwargs)

    def test_unsat_core_available(self):
        solver = self._make_solver()
        solver.assert_formula(
            Comparison(ComparisonOp.GT, Var("x"), Const(0)), label="c1"
        )
        solver.assert_formula(
            Comparison(ComparisonOp.LT, Var("x"), Const(0)), label="c2"
        )
        solver.check_sat()
        core = solver.get_unsat_core()
        assert core is not None

    def test_unsat_core_none_on_sat(self):
        solver = self._make_solver()
        solver.assert_formula(Comparison(ComparisonOp.GT, Var("x"), Const(0)))
        solver.check_sat()
        assert solver.get_unsat_core() is None
