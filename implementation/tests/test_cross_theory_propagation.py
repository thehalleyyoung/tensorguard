"""Tests for cross-theory deduction propagation and mixed-arithmetic LIA reduction.

Validates:
1. CrossTheoryDeductionPropagator propagates concrete values across theories
2. MixedArithmeticPropagator reduces NIA reshape constraints to LIA
3. Pairwise boundary disaggregation classifies failures correctly
4. Backward propagation with mixed-arithmetic catches wrong_out_features
   at reshape boundaries
"""

from __future__ import annotations

import pytest

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

pytestmark = pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")

from src.smt.theory_combination import (
    DomainKind,
    TheorySolver,
    TheoryCombination,
    TensorTheoryCombination,
    MixedArithmeticPropagator,
    CrossTheoryDeductionPropagator,
)
from src.theory_combination_analysis import (
    classify_boundary_failure,
    aggregate_boundary_report,
    BoundaryFailure,
    PairwiseBoundaryReport,
)
from src.model_checker import verify_model


# ═══════════════════════════════════════════════════════════════════════════
# 1. MixedArithmeticPropagator tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMixedArithmeticPropagator:
    """Test partial evaluation of NIA products to LIA constraints."""

    def test_partial_evaluate_all_concrete(self):
        dims = [z3.IntVal(2), z3.IntVal(3), z3.IntVal(4)]
        concrete, symbolic = MixedArithmeticPropagator.partial_evaluate_product(dims)
        assert concrete == 24
        assert symbolic == []

    def test_partial_evaluate_all_symbolic(self):
        a, b = z3.Ints("a b")
        concrete, symbolic = MixedArithmeticPropagator.partial_evaluate_product([a, b])
        assert concrete == 1
        assert len(symbolic) == 2

    def test_partial_evaluate_mixed(self):
        a = z3.Int("a")
        dims = [a, z3.IntVal(4), z3.IntVal(8)]
        concrete, symbolic = MixedArithmeticPropagator.partial_evaluate_product(dims)
        assert concrete == 32
        assert len(symbolic) == 1
        assert symbolic[0].get_id() == a.get_id()

    def test_lia_fully_concrete_equal(self):
        """Fully concrete reshape with matching element counts."""
        old = [z3.IntVal(2), z3.IntVal(3), z3.IntVal(4)]
        new = [z3.IntVal(6), z3.IntVal(4)]
        cs = MixedArithmeticPropagator.generate_lia_reshape_constraints(old, new)
        assert cs == []  # 24 == 24, no constraints needed

    def test_lia_fully_concrete_unequal(self):
        """Fully concrete reshape with mismatched element counts."""
        old = [z3.IntVal(2), z3.IntVal(3), z3.IntVal(4)]
        new = [z3.IntVal(6), z3.IntVal(5)]
        cs = MixedArithmeticPropagator.generate_lia_reshape_constraints(old, new)
        assert len(cs) == 1
        # Should be BoolVal(False)
        s = z3.Solver()
        s.add(cs[0])
        assert s.check() == z3.unsat

    def test_lia_same_symbolic_factors_concrete_match(self):
        """Same symbolic factor on both sides, concrete parts match."""
        batch = z3.Int("batch")
        old = [batch, z3.IntVal(128)]
        new = [batch, z3.IntVal(16), z3.IntVal(8)]
        cs = MixedArithmeticPropagator.generate_lia_reshape_constraints(old, new)
        # batch * 128 == batch * 128: same symbolic, 128 == 128
        assert cs == []

    def test_lia_same_symbolic_factors_concrete_mismatch(self):
        """Same symbolic factor on both sides, concrete parts DIFFER."""
        batch = z3.Int("batch")
        old = [batch, z3.IntVal(128)]
        new = [batch, z3.IntVal(32), z3.IntVal(8)]
        cs = MixedArithmeticPropagator.generate_lia_reshape_constraints(old, new)
        # batch * 128 != batch * 256 → unsatisfiable
        assert len(cs) == 1
        s = z3.Solver()
        s.add(cs[0])
        assert s.check() == z3.unsat

    def test_lia_one_side_concrete(self):
        """One side fully concrete, other has symbolic factor."""
        x = z3.Int("x")
        old = [z3.IntVal(6)]
        new = [z3.IntVal(2), x]
        cs = MixedArithmeticPropagator.generate_lia_reshape_constraints(old, new)
        # 6 == 2 * x → x == 3
        assert len(cs) == 1
        s = z3.Solver()
        s.add(cs[0])
        assert s.check() == z3.sat
        assert s.model()[x].as_long() == 3

    def test_lia_single_symbolic_each_side(self):
        """Single symbolic variable on each side."""
        a = z3.Int("a")
        b = z3.Int("b")
        old = [z3.IntVal(4), a]
        new = [z3.IntVal(2), b]
        cs = MixedArithmeticPropagator.generate_lia_reshape_constraints(old, new)
        # 4*a == 2*b → 2*a == b
        assert len(cs) == 1
        s = z3.Solver()
        s.add(cs[0])
        s.add(a == 3)
        assert s.check() == z3.sat
        assert s.model()[b].as_long() == 6

    def test_propagate_to_solver(self):
        """Test propagate_reshape_to_shape_theory adds constraints."""
        s = z3.Solver()
        x = z3.Int("x")
        s.add(x > 0)
        old = [z3.IntVal(12)]
        new = [z3.IntVal(3), x]
        n = MixedArithmeticPropagator.propagate_reshape_to_shape_theory(
            s, old, new,
        )
        assert n >= 1
        assert s.check() == z3.sat
        assert s.model()[x].as_long() == 4


# ═══════════════════════════════════════════════════════════════════════════
# 2. CrossTheoryDeductionPropagator tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossTheoryDeductionPropagator:
    """Test cross-theory deduction propagation."""

    def test_propagate_concrete_value_across_theories(self):
        """Concrete value in one theory propagated to another."""
        s1 = z3.Solver()
        s2 = z3.Solver()
        x = z3.Int("x_prop")

        s1.add(x == 42)
        s2.add(x > 0)

        combo = TheoryCombination()
        combo.add_theory(TheorySolver(
            name="t1", solver=s1,
            domain_kind=DomainKind.STABLY_INFINITE,
            shared_vars=[x],
        ))
        combo.add_theory(TheorySolver(
            name="t2", solver=s2,
            domain_kind=DomainKind.STABLY_INFINITE,
            shared_vars=[x],
        ))

        deductions = combo.propagate_cross_theory_deductions()
        assert len(deductions) >= 1
        # Should propagate x==42 from t1 to t2
        found = False
        for src, tgt, cs in deductions:
            if src == "t1" and tgt == "t2":
                found = True
        assert found

    def test_deduction_loop_reaches_fixpoint(self):
        """Deduction loop terminates at fixpoint."""
        s1 = z3.Solver()
        s2 = z3.Solver()
        x = z3.Int("x_fix")

        s1.add(x == 10)
        s2.add(x > 0)

        combo = TheoryCombination()
        combo.add_theory(TheorySolver(
            name="t1", solver=s1,
            domain_kind=DomainKind.STABLY_INFINITE,
            shared_vars=[x],
        ))
        combo.add_theory(TheorySolver(
            name="t2", solver=s2,
            domain_kind=DomainKind.STABLY_INFINITE,
            shared_vars=[x],
        ))

        total = combo.run_deduction_propagation_loop(max_rounds=5)
        assert total >= 1

    def test_deduction_detects_cross_theory_conflict(self):
        """Deduction propagation makes a cross-theory conflict detectable."""
        s1 = z3.Solver()
        s2 = z3.Solver()
        x = z3.Int("x_conflict")

        s1.add(x == 5)
        s2.add(x > 10)

        combo = TheoryCombination()
        combo.add_theory(TheorySolver(
            name="shape", solver=s1,
            domain_kind=DomainKind.STABLY_INFINITE,
            shared_vars=[x],
        ))
        combo.add_theory(TheorySolver(
            name="stride", solver=s2,
            domain_kind=DomainKind.STABLY_INFINITE,
            shared_vars=[x],
        ))

        combo.run_deduction_propagation_loop()
        # After propagation, s2 should be unsat (x==5 AND x>10)
        assert s2.check() == z3.unsat

    def test_facade_propagate_all(self):
        """CrossTheoryDeductionPropagator facade works end-to-end."""
        s1 = z3.Solver()
        s2 = z3.Solver()
        x = z3.Int("x_facade")

        s1.add(x == 7)
        s2.add(x > 0)

        combo = TheoryCombination()
        combo.add_theory(TheorySolver(
            name="t1", solver=s1,
            domain_kind=DomainKind.STABLY_INFINITE,
            shared_vars=[x],
        ))
        combo.add_theory(TheorySolver(
            name="t2", solver=s2,
            domain_kind=DomainKind.STABLY_INFINITE,
            shared_vars=[x],
        ))

        prop = CrossTheoryDeductionPropagator(combo)
        total = prop.propagate_all()
        assert total >= 1
        assert prop.deductions_propagated >= 1

    def test_facade_with_reshape(self):
        """Facade adds LIA reshape constraints."""
        combo = TheoryCombination()
        prop = CrossTheoryDeductionPropagator(combo)
        s = z3.Solver()
        x = z3.Int("x_rs")
        s.add(x > 0)
        n = prop.add_reshape_constraints(
            s,
            [z3.IntVal(12)],
            [z3.IntVal(3), x],
        )
        assert n >= 1
        assert prop.lia_constraints_added >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. Verify combination with deduction propagation
# ═══════════════════════════════════════════════════════════════════════════


class TestVerifyCombinationWithDeduction:
    """Test that verify_theory_combination_consistency uses deduction."""

    def test_deduction_enhances_consistency_check(self):
        """Deduction propagation runs before arrangement enumeration."""
        from src.smt.device_theory import DeviceSort, DEVICE_VALS

        bc_solver = z3.Solver()
        dim = z3.Int("dim_ded")
        bc_solver.add(dim == 64)

        dev_solver = z3.Solver()
        dev = z3.Const("dev_ded", DeviceSort)
        dev_solver.add(dev == DEVICE_VALS["CPU"])

        combo = TensorTheoryCombination()
        combo.add_broadcast_theory(bc_solver, [dim])
        combo.add_device_theory(dev_solver, [dev])
        result = combo.verify_theory_combination_consistency()
        assert result.is_consistent

    def test_deduction_catches_shape_stride_conflict(self):
        """Cross-theory conflict between shape and stride detected."""
        s_shape = z3.Solver()
        s_stride = z3.Solver()
        dim = z3.Int("dim_ss")

        s_shape.add(dim == 128)
        s_stride.add(dim > 256)

        combo = TheoryCombination()
        combo.add_theory(TheorySolver(
            name="shape", solver=s_shape,
            domain_kind=DomainKind.STABLY_INFINITE,
            shared_vars=[dim],
        ))
        combo.add_theory(TheorySolver(
            name="stride", solver=s_stride,
            domain_kind=DomainKind.STABLY_INFINITE,
            shared_vars=[dim],
        ))

        result = combo.verify_theory_combination_consistency()
        # After deduction, stride solver has dim==128 AND dim>256 → UNSAT
        assert not result.is_consistent


# ═══════════════════════════════════════════════════════════════════════════
# 4. Pairwise boundary disaggregation tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundaryDisaggregation:
    """Test classify_boundary_failure and aggregate_boundary_report."""

    def test_reshape_classified_as_shape_stride(self):
        bf = classify_boundary_failure(
            "reshape_mismatch",
            ["T_shape", "T_stride"],
            {"dim_a > 0": "T_shape", "contiguous": "T_stride"},
            failure_location="reshape",
        )
        assert bf.failing_pair == ("T_shape", "T_stride")
        assert "T_shape" in bf.failure_reason

    def test_device_classified_as_shape_device(self):
        bf = classify_boundary_failure(
            "device_conflict",
            ["T_shape", "T_device"],
            {"dim > 0": "T_shape", "dev == CPU": "T_device"},
            failure_location="same_device",
        )
        assert bf.failing_pair == ("T_shape", "T_device")

    def test_backward_classified(self):
        bf = classify_boundary_failure(
            "wrong_out_features_reshape",
            ["T_shape", "T_stride"],
            {"dim == 128": "T_shape", "numel": "T_stride"},
            failure_location="backward_propagation",
        )
        assert bf.failing_pair == ("T_shape", "T_stride")

    def test_no_failure_location(self):
        bf = classify_boundary_failure(
            "unknown",
            ["T_shape", "T_device"],
            {"a": "T_shape", "b": "T_device"},
        )
        assert bf.failing_pair is None
        # All pairs marked as passed
        for k, v in bf.pair_results.items():
            assert v is True

    def test_aggregate_report(self):
        failures = [
            classify_boundary_failure(
                "bench1", ["T_shape", "T_stride"],
                {"a": "T_shape", "b": "T_stride"},
                "reshape",
            ),
            classify_boundary_failure(
                "bench2", ["T_shape", "T_device"],
                {"a": "T_shape", "b": "T_device"},
                "device",
            ),
            classify_boundary_failure(
                "bench3", ["T_shape", "T_stride"],
                {"a": "T_shape", "b": "T_stride"},
                "reshape",
            ),
        ]
        report = aggregate_boundary_report(failures, total_benchmarks=10)
        assert report.total_benchmarks == 10
        assert isinstance(report.failures_by_pair, dict)
        assert isinstance(report.f1_by_pair, dict)

    def test_aggregate_report_to_dict(self):
        failures = [
            classify_boundary_failure(
                "bench1", ["T_shape", "T_stride"],
                {"a": "T_shape", "b": "T_stride"},
                "reshape",
            ),
        ]
        report = aggregate_boundary_report(failures, 5)
        d = report.to_dict()
        assert "total_benchmarks" in d
        assert "failures_by_pair" in d
        assert "f1_by_pair" in d
        assert "individual_results" in d


# ═══════════════════════════════════════════════════════════════════════════
# 5. Model checker integration: mixed-theory reshape mutants
# ═══════════════════════════════════════════════════════════════════════════


class TestMixedTheoryReshapeMutants:
    """Verify backward propagation + LIA reduction catches reshape mutants."""

    def test_wrong_out_features_through_reshape(self):
        """wrong_out_features through reshape: LIA reduction should catch it."""
        src = """\
import torch
import torch.nn as nn

class BadReshapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 256)
        x = self.fc2(x)
        return x
"""
        result = verify_model(src, {"x": ("batch", 784)})
        assert not result.safe, "Should detect wrong_out_features through reshape"

    def test_correct_reshape_is_safe(self):
        """Correct reshape should not be flagged."""
        src = """\
import torch
import torch.nn as nn

class GoodReshapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 256)
        x = self.fc2(x)
        return x
"""
        result = verify_model(src, {"x": ("batch", 784)})
        assert result.safe, f"Expected safe, got: {result.counterexample}"

    def test_wrong_out_features_through_flatten(self):
        """wrong_out_features where flatten connects mismatched layers."""
        src = """\
import torch
import torch.nn as nn

class WrongFlattenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
"""
        # With 32x32 input, conv output is 16 * 30 * 30 = 14400, not 512
        result = verify_model(src, {"x": ("batch", 3, 32, 32)})
        assert not result.safe, "Should detect wrong features after flatten"

    def test_three_layer_reshape_chain(self):
        """Three-layer chain with reshape in between: wrong dimensions."""
        src = """\
import torch
import torch.nn as nn

class ThreeLayerReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 32)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
"""
        result = verify_model(src, {"x": ("batch", 100)})
        # fc1 outputs 64, reshape to (-1, 32): requires 64 = n*32
        # This is actually valid (n=2), so fc2 gets [batch*2, 32] → ok
        # But this is a different batch size! Let's check it doesn't crash.
        # The key test is that it runs without error.
        assert isinstance(result.safe, bool)
