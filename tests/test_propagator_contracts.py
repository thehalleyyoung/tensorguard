"""
Tests for the formal callback contracts (C1–C5) across all four
UserPropagator implementations:
  - BroadcastPropagator  (broadcast_theory.py)
  - DevicePropagator     (device_theory.py)
  - PhasePropagator      (phase_theory.py)
  - StridePropagator     (stride_theory.py)

At least 30 test cases covering:
  C1: Push-Pop Invertibility
  C2: Fixed Monotonicity / Propagation Soundness under assignment
  C3: Final Completeness
  C4: Conflict Soundness
  C5: Propagation Soundness
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Tuple

import pytest

# Ensure project root is importable
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
    ),
)

import z3

from src.smt.propagator_contracts import (
    ContractViolation,
    PropagatorInstrumentor,
    verify_push_pop_invertibility,
    verify_nested_push_pop,
    verify_fixed_soundness,
    verify_final_completeness,
    verify_conflict_soundness,
    verify_propagation_soundness,
)

from src.smt.broadcast_theory import (
    BroadcastPropagator,
    BroadcastTheoryPlugin,
    broadcast_result_dim,
    matmul_compatible,
    broadcast_compatible,
)
from src.smt.device_theory import (
    DevicePropagator,
    DeviceTheoryPlugin,
    DeviceSort,
    DEVICE_VALS,
    same_device,
    transfer_device,
    inherit_device,
)
from src.smt.phase_theory import (
    PhasePropagator,
    PhaseTheoryPlugin,
    set_phase,
    dropout_behavior,
    batchnorm_behavior,
)
from src.smt.stride_theory import (
    StridePropagator,
    StrideTheoryPlugin,
    contiguous_strides,
    reshape_valid,
    divisibility_constraint,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers / Factories
# ═══════════════════════════════════════════════════════════════════════════


def _make_broadcast_solver():
    """Factory: solver with a broadcast_result_dim(a, b, out) constraint."""
    s = z3.Solver()
    prop = BroadcastPropagator(s)
    a, b, out = z3.Ints("a b out")
    s.add(broadcast_result_dim(prop, a, b, out))
    return s, prop, {"a": a, "b": b, "out": out}


def _make_broadcast_matmul_solver():
    """Factory: solver with matmul_compatible constraint."""
    s = z3.Solver()
    prop = BroadcastPropagator(s)
    sa0, sa1 = z3.Ints("sa0 sa1")
    sb0, sb1 = z3.Ints("sb0 sb1")
    s.add(matmul_compatible(prop, [sa0, sa1], [sb0, sb1]))
    return s, prop, {"sa0": sa0, "sa1": sa1, "sb0": sb0, "sb1": sb1}


def _make_device_same_solver():
    """Factory: solver with same_device(da, db) constraint."""
    s = z3.Solver()
    prop = DevicePropagator(s)
    da = z3.Const("da", DeviceSort)
    db = z3.Const("db", DeviceSort)
    s.add(same_device(prop, da, db))
    return s, prop, {"da": da, "db": db}


def _make_device_transfer_solver():
    """Factory: solver with transfer_device constraint."""
    s = z3.Solver()
    prop = DevicePropagator(s)
    din = z3.Const("din", DeviceSort)
    dout = z3.Const("dout", DeviceSort)
    target = DEVICE_VALS["CUDA_0"]
    s.add(transfer_device(prop, din, dout, target))
    return s, prop, {"din": din, "dout": dout}


def _make_device_inherit_solver():
    """Factory: solver with inherit_device constraint."""
    s = z3.Solver()
    prop = DevicePropagator(s)
    din = z3.Const("din", DeviceSort)
    dout = z3.Const("dout", DeviceSort)
    s.add(inherit_device(prop, din, dout))
    return s, prop, {"din": din, "dout": dout}


def _make_phase_dropout_solver():
    """Factory: solver with dropout_behavior constraint (EVAL)."""
    s = z3.Solver()
    prop = PhasePropagator(s)
    phase = z3.Bool("phase")
    inp = z3.Bool("inp")
    out = z3.Bool("out")
    s.add(set_phase(prop, phase, False))   # EVAL
    s.add(dropout_behavior(prop, phase, inp, out))
    return s, prop, {"phase": phase, "inp": inp, "out": out}


def _make_phase_batchnorm_solver():
    """Factory: solver with batchnorm_behavior constraint."""
    s = z3.Solver()
    prop = PhasePropagator(s)
    phase = z3.Bool("phase")
    urs = z3.Bool("urs")
    s.add(batchnorm_behavior(prop, phase, urs))
    return s, prop, {"phase": phase, "urs": urs}


def _make_stride_contiguous_solver():
    """Factory: solver with contiguous_strides for shape (d0, d1, d2)."""
    s = z3.Solver()
    prop = StridePropagator(s)
    d0, d1, d2 = z3.Ints("d0 d1 d2")
    st0, st1, st2 = z3.Ints("st0 st1 st2")
    s.add(contiguous_strides(prop, [d0, d1, d2], [st0, st1, st2]))
    return s, prop, {"d0": d0, "d1": d1, "d2": d2,
                      "st0": st0, "st1": st1, "st2": st2}


def _make_stride_reshape_solver():
    """Factory: solver with reshape_valid constraint."""
    s = z3.Solver()
    prop = StridePropagator(s)
    r0, r1 = z3.Ints("r0 r1")
    t0, t1, t2 = z3.Ints("t0 t1 t2")
    s.add(reshape_valid(prop, [r0, r1], [t0, t1, t2]))
    return s, prop, {"r0": r0, "r1": r1, "t0": t0, "t1": t1, "t2": t2}


def _make_stride_divisibility_solver():
    """Factory: solver with divisibility_constraint."""
    s = z3.Solver()
    prop = StridePropagator(s)
    dd, dv = z3.Ints("dd dv")
    s.add(divisibility_constraint(prop, dd, dv))
    return s, prop, {"dd": dd, "dv": dv}


# ═══════════════════════════════════════════════════════════════════════════
# C1: Push-Pop Invertibility  (8 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestC1PushPopInvertibility:
    """C1: For all states σ, pop(push(σ)) = σ."""

    def test_c1_broadcast_basic(self):
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        violations = verify_push_pop_invertibility(prop, num_cycles=5)
        assert violations == [], [v.description for v in violations]

    def test_c1_broadcast_nested(self):
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        violations = verify_nested_push_pop(prop, depth=4)
        assert violations == [], [v.description for v in violations]

    def test_c1_device_basic(self):
        s = z3.Solver()
        prop = DevicePropagator(s)
        violations = verify_push_pop_invertibility(prop, num_cycles=5)
        assert violations == [], [v.description for v in violations]

    def test_c1_device_nested(self):
        s = z3.Solver()
        prop = DevicePropagator(s)
        violations = verify_nested_push_pop(prop, depth=4)
        assert violations == [], [v.description for v in violations]

    def test_c1_phase_basic(self):
        s = z3.Solver()
        prop = PhasePropagator(s)
        violations = verify_push_pop_invertibility(prop, num_cycles=5)
        assert violations == [], [v.description for v in violations]

    def test_c1_phase_nested(self):
        s = z3.Solver()
        prop = PhasePropagator(s)
        violations = verify_nested_push_pop(prop, depth=4)
        assert violations == [], [v.description for v in violations]

    def test_c1_stride_basic(self):
        s = z3.Solver()
        prop = StridePropagator(s)
        violations = verify_push_pop_invertibility(prop, num_cycles=5)
        assert violations == [], [v.description for v in violations]

    def test_c1_stride_nested(self):
        s = z3.Solver()
        prop = StridePropagator(s)
        violations = verify_nested_push_pop(prop, depth=4)
        assert violations == [], [v.description for v in violations]


# ═══════════════════════════════════════════════════════════════════════════
# C2: Fixed Monotonicity  (6 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestC2FixedMonotonicity:
    """C2: Propagated clauses are consequences of theory ∪ {v=val}."""

    def test_c2_broadcast_fix_a(self):
        axioms_a = z3.Ints("a b out")
        a, b, out = axioms_a
        theory = [z3.Or(a == b, a == 1, b == 1),
                  z3.Implies(a == b, out == a),
                  z3.Implies(a == 1, out == b),
                  z3.Implies(b == 1, out == a)]
        violations = verify_fixed_soundness(
            _make_broadcast_solver, "a", 3, theory_axioms=theory
        )
        assert violations == [], [v.description for v in violations]

    def test_c2_broadcast_fix_b(self):
        a, b, out = z3.Ints("a b out")
        theory = [z3.Or(a == b, a == 1, b == 1),
                  z3.Implies(a == b, out == a),
                  z3.Implies(a == 1, out == b),
                  z3.Implies(b == 1, out == a)]
        violations = verify_fixed_soundness(
            _make_broadcast_solver, "b", 1, theory_axioms=theory
        )
        assert violations == [], [v.description for v in violations]

    def test_c2_device_fix_da(self):
        violations = verify_fixed_soundness(
            _make_device_same_solver, "da", DEVICE_VALS["CPU"]
        )
        assert violations == [], [v.description for v in violations]

    def test_c2_phase_fix_phase_eval(self):
        violations = verify_fixed_soundness(
            _make_phase_batchnorm_solver, "phase", False
        )
        assert violations == [], [v.description for v in violations]

    def test_c2_stride_fix_d0(self):
        violations = verify_fixed_soundness(
            _make_stride_contiguous_solver, "d0", 2
        )
        assert violations == [], [v.description for v in violations]

    def test_c2_stride_fix_all_dims(self):
        """Fix all dimensions and check stride propagation."""
        def factory():
            s, prop, vd = _make_stride_contiguous_solver()
            s.add(vd["d0"] == 2)
            s.add(vd["d1"] == 3)
            return s, prop, vd
        violations = verify_fixed_soundness(factory, "d2", 4)
        assert violations == [], [v.description for v in violations]


# ═══════════════════════════════════════════════════════════════════════════
# C3: Final Completeness  (8 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestC3FinalCompleteness:
    """C3: If _on_final() does not conflict, assignment is satisfying."""

    def test_c3_broadcast_sat(self):
        violations = verify_final_completeness(
            _make_broadcast_solver,
            {"a": 3, "b": 1, "out": 3},
            is_satisfying=True,
        )
        assert violations == [], [v.description for v in violations]

    def test_c3_broadcast_unsat(self):
        violations = verify_final_completeness(
            _make_broadcast_solver,
            {"a": 3, "b": 2, "out": 5},
            is_satisfying=False,
        )
        assert violations == [], [v.description for v in violations]

    def test_c3_device_same_sat(self):
        violations = verify_final_completeness(
            _make_device_same_solver,
            {"da": DEVICE_VALS["CPU"], "db": DEVICE_VALS["CPU"]},
            is_satisfying=True,
        )
        assert violations == [], [v.description for v in violations]

    def test_c3_device_same_unsat(self):
        violations = verify_final_completeness(
            _make_device_same_solver,
            {"da": DEVICE_VALS["CPU"], "db": DEVICE_VALS["CUDA_0"]},
            is_satisfying=False,
        )
        assert violations == [], [v.description for v in violations]

    def test_c3_phase_dropout_eval_identity_sat(self):
        violations = verify_final_completeness(
            _make_phase_dropout_solver,
            {"phase": False, "inp": True, "out": True},
            is_satisfying=True,
        )
        assert violations == [], [v.description for v in violations]

    def test_c3_phase_dropout_eval_identity_unsat(self):
        violations = verify_final_completeness(
            _make_phase_dropout_solver,
            {"phase": False, "inp": True, "out": False},
            is_satisfying=False,
        )
        assert violations == [], [v.description for v in violations]

    def test_c3_stride_contiguous_sat(self):
        violations = verify_final_completeness(
            _make_stride_contiguous_solver,
            {"d0": 2, "d1": 3, "d2": 4, "st0": 12, "st1": 4, "st2": 1},
            is_satisfying=True,
        )
        assert violations == [], [v.description for v in violations]

    def test_c3_stride_contiguous_unsat(self):
        violations = verify_final_completeness(
            _make_stride_contiguous_solver,
            {"d0": 2, "d1": 3, "d2": 4, "st0": 12, "st1": 3, "st2": 1},
            is_satisfying=False,
        )
        assert violations == [], [v.description for v in violations]


# ═══════════════════════════════════════════════════════════════════════════
# C4: Conflict Soundness  (8 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestC4ConflictSoundness:
    """C4: Every conflict clause is a valid nogood."""

    def test_c4_broadcast_incompatible_dims(self):
        violations = verify_conflict_soundness(
            _make_broadcast_solver,
            {"a": 3, "b": 2},
        )
        assert violations == [], [v.description for v in violations]

    def test_c4_broadcast_wrong_output(self):
        violations = verify_conflict_soundness(
            _make_broadcast_solver,
            {"a": 3, "b": 1, "out": 7},
        )
        assert violations == [], [v.description for v in violations]

    def test_c4_device_same_conflict(self):
        violations = verify_conflict_soundness(
            _make_device_same_solver,
            {"da": DEVICE_VALS["CPU"], "db": DEVICE_VALS["CUDA_0"]},
        )
        assert violations == [], [v.description for v in violations]

    def test_c4_device_transfer_conflict(self):
        violations = verify_conflict_soundness(
            _make_device_transfer_solver,
            {"din": DEVICE_VALS["CPU"], "dout": DEVICE_VALS["CUDA_1"]},
        )
        assert violations == [], [v.description for v in violations]

    def test_c4_device_inherit_conflict(self):
        violations = verify_conflict_soundness(
            _make_device_inherit_solver,
            {"din": DEVICE_VALS["CUDA_0"], "dout": DEVICE_VALS["CPU"]},
        )
        assert violations == [], [v.description for v in violations]

    def test_c4_phase_batchnorm_eval_conflict(self):
        """EVAL requires uses_running_stats=True; setting False is conflict."""
        def factory():
            s = z3.Solver()
            prop = PhasePropagator(s)
            phase = z3.Bool("phase")
            urs = z3.Bool("urs")
            s.add(set_phase(prop, phase, False))
            s.add(batchnorm_behavior(prop, phase, urs))
            return s, prop, {"phase": phase, "urs": urs}
        violations = verify_conflict_soundness(
            factory, {"phase": False, "urs": False}
        )
        assert violations == [], [v.description for v in violations]

    def test_c4_stride_wrong_strides(self):
        violations = verify_conflict_soundness(
            _make_stride_contiguous_solver,
            {"d0": 2, "d1": 3, "d2": 4, "st0": 10, "st1": 4, "st2": 1},
        )
        assert violations == [], [v.description for v in violations]

    def test_c4_stride_divisibility_conflict(self):
        violations = verify_conflict_soundness(
            _make_stride_divisibility_solver,
            {"dd": 12, "dv": 5},
        )
        assert violations == [], [v.description for v in violations]


# ═══════════════════════════════════════════════════════════════════════════
# C5: Propagation Soundness  (8 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestC5PropagationSoundness:
    """C5: Propagated literals are implied by theory ∪ current_fixed."""

    def test_c5_broadcast_propagate_output(self):
        """Fix a=3, b=1 → out should be propagated to 3."""
        violations = verify_propagation_soundness(
            _make_broadcast_solver,
            {"a": 3, "b": 1},
            expected_propagations={"out": 3},
        )
        assert violations == [], [v.description for v in violations]

    def test_c5_broadcast_propagate_equal_dims(self):
        """Fix a=4, b=4 → out should be 4."""
        violations = verify_propagation_soundness(
            _make_broadcast_solver,
            {"a": 4, "b": 4},
            expected_propagations={"out": 4},
        )
        assert violations == [], [v.description for v in violations]

    def test_c5_device_same_propagation(self):
        """Fix da=CPU → db should be inferred as CPU."""
        violations = verify_propagation_soundness(
            _make_device_same_solver,
            {"da": DEVICE_VALS["CPU"]},
            expected_propagations={"db": DEVICE_VALS["CPU"]},
        )
        assert violations == [], [v.description for v in violations]

    def test_c5_device_transfer_propagation(self):
        """Transfer to CUDA_0: fix din=CPU → dout should be CUDA_0."""
        violations = verify_propagation_soundness(
            _make_device_transfer_solver,
            {"din": DEVICE_VALS["CPU"]},
            expected_propagations={"dout": DEVICE_VALS["CUDA_0"]},
        )
        assert violations == [], [v.description for v in violations]

    def test_c5_device_inherit_propagation(self):
        """Inherit: fix din=CUDA_1 → dout should be CUDA_1."""
        def factory():
            s = z3.Solver()
            prop = DevicePropagator(s)
            din = z3.Const("din", DeviceSort)
            dout = z3.Const("dout", DeviceSort)
            s.add(inherit_device(prop, din, dout))
            return s, prop, {"din": din, "dout": dout}
        violations = verify_propagation_soundness(
            factory,
            {"din": DEVICE_VALS["CUDA_1"]},
            expected_propagations={"dout": DEVICE_VALS["CUDA_1"]},
        )
        assert violations == [], [v.description for v in violations]

    def test_c5_phase_batchnorm_eval_propagation(self):
        """EVAL mode → uses_running_stats should be True."""
        violations = verify_propagation_soundness(
            _make_phase_batchnorm_solver,
            {"phase": False},
            expected_propagations={"urs": True},
        )
        assert violations == [], [v.description for v in violations]

    def test_c5_stride_propagate_strides(self):
        """Fix shape (2,3,4) → strides should be (12,4,1)."""
        violations = verify_propagation_soundness(
            _make_stride_contiguous_solver,
            {"d0": 2, "d1": 3, "d2": 4},
            expected_propagations={"st0": 12, "st1": 4, "st2": 1},
        )
        assert violations == [], [v.description for v in violations]

    def test_c5_stride_reshape_propagation(self):
        """Reshape (6,4) → (2,3,?) should infer ?=4."""
        def factory():
            s, prop, vd = _make_stride_reshape_solver()
            s.add(vd["t2"] > 0)
            return s, prop, vd
        violations = verify_propagation_soundness(
            factory,
            {"r0": 6, "r1": 4, "t0": 2, "t1": 3},
            expected_propagations={"t2": 4},
        )
        assert violations == [], [v.description for v in violations]


# ═══════════════════════════════════════════════════════════════════════════
# Additional cross-cutting / edge-case tests (2+ tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossCutting:
    """Cross-cutting tests for edge cases and combined scenarios."""

    def test_c1_broadcast_with_preexisting_state(self):
        """C1 with pre-existing fixed vars should still restore correctly."""
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        # Seed some state
        prop._fixed[1] = 10
        prop._fixed[2] = 20
        violations = verify_push_pop_invertibility(prop, num_cycles=3)
        assert violations == [], [v.description for v in violations]
        # Original state should still be there
        assert prop._fixed[1] == 10
        assert prop._fixed[2] == 20

    def test_c1_stride_incremental_push_pop(self):
        """C1: push, mutate, push, mutate, pop(1), check, pop(1), check."""
        s = z3.Solver()
        prop = StridePropagator(s)
        prop._fixed[100] = 5
        state0 = dict(prop._fixed)

        prop.push()
        prop._fixed[200] = 10
        state1 = dict(prop._fixed)

        prop.push()
        prop._fixed[300] = 15

        # Pop one level: should restore state1
        prop.pop(1)
        assert dict(prop._fixed) == state1

        # Pop another: should restore state0
        prop.pop(1)
        assert dict(prop._fixed) == state0

    def test_c3_matmul_compatible_sat(self):
        """C3: matmul with matching inner dims is SAT."""
        violations = verify_final_completeness(
            _make_broadcast_matmul_solver,
            {"sa0": 2, "sa1": 3, "sb0": 3, "sb1": 4},
            is_satisfying=True,
        )
        assert violations == [], [v.description for v in violations]

    def test_c4_matmul_incompatible(self):
        """C4: matmul with mismatched inner dims is UNSAT."""
        violations = verify_conflict_soundness(
            _make_broadcast_matmul_solver,
            {"sa0": 2, "sa1": 3, "sb0": 5, "sb1": 4},
        )
        assert violations == [], [v.description for v in violations]

    def test_c5_broadcast_unit_propagation(self):
        """C5: b=1 always, a=5 → out=5."""
        violations = verify_propagation_soundness(
            _make_broadcast_solver,
            {"a": 5, "b": 1},
            expected_propagations={"out": 5},
        )
        assert violations == [], [v.description for v in violations]

    def test_c3_phase_train_dropout_different_ok(self):
        """C3: In TRAIN mode, dropout output may differ from input."""
        def factory():
            s = z3.Solver()
            prop = PhasePropagator(s)
            phase = z3.Bool("phase")
            inp = z3.Bool("inp")
            out = z3.Bool("out")
            s.add(set_phase(prop, phase, True))   # TRAIN
            s.add(dropout_behavior(prop, phase, inp, out))
            return s, prop, {"phase": phase, "inp": inp, "out": out}
        violations = verify_final_completeness(
            factory,
            {"phase": True, "inp": True, "out": False},
            is_satisfying=True,
        )
        assert violations == [], [v.description for v in violations]

    def test_c3_stride_divisibility_sat(self):
        """C3: 12 divisible by 4 is SAT."""
        violations = verify_final_completeness(
            _make_stride_divisibility_solver,
            {"dd": 12, "dv": 4},
            is_satisfying=True,
        )
        assert violations == [], [v.description for v in violations]

    def test_c5_phase_batchnorm_train_propagation(self):
        """C5: TRAIN mode → uses_running_stats should be False."""
        violations = verify_propagation_soundness(
            _make_phase_batchnorm_solver,
            {"phase": True},
            expected_propagations={"urs": False},
        )
        assert violations == [], [v.description for v in violations]
