"""
Lean 4 ↔ Python Conformance Summary
=====================================

Maps each Lean theorem in ``lean/TheoryCombination.lean`` to its Python
conformance tests in ``tests/test_lean_conformance.py`` and documents the
correspondence between abstract Lean types and concrete Python data structures.

This module serves as the *conformance bridge* addressing the reviewers'
concern that "Lean 4 mechanization is metatheoretically disconnected from
the Python implementation."

While we cannot perform code extraction or bisimulation between Lean 4
and Python, we establish a *conformance testing* relation:

    For every property P proved in Lean, there exists a test suite T_P
    that exercises the Python implementation against P using concrete
    and random inputs, verifying behavioral equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Type Correspondence Table
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TypeCorrespondence:
    """Maps a Lean type to its Python implementation counterpart."""
    lean_type: str
    lean_description: str
    python_type: str
    python_module: str
    notes: str = ""


TYPE_CORRESPONDENCE: List[TypeCorrespondence] = [
    TypeCorrespondence(
        lean_type="Fin n → Nat",
        lean_description="Shape as function from index to positive natural",
        python_type="Tuple[int, ...]",
        python_module="src.smt.broadcast_theory",
        notes="Lean uses Fin-indexed functions; Python uses tuples of ints",
    ),
    TypeCorrespondence(
        lean_type="broadcastConsistent n a b",
        lean_description="∀ i, a[i]==b[i] ∨ a[i]==1 ∨ b[i]==1",
        python_type="_are_dims_broadcast_compatible(a, b) → bool",
        python_module="src.smt.broadcast_theory",
        notes="Lean Prop, Python bool; broadcastDim_sound proves equivalence",
    ),
    TypeCorrespondence(
        lean_type="broadcastResult n a b",
        lean_description="fun i => max(a[i], b[i])",
        python_type="_broadcast_result(a, b) → int / _broadcast_shape(a, b) → tuple",
        python_module="src.smt.broadcast_theory",
        notes="Element-wise max; Lean at Fin-indexed level, Python at tuple level",
    ),
    TypeCorrespondence(
        lean_type="broadcastDimCheck a b → Bool",
        lean_description="a == b || a == 1 || b == 1",
        python_type="_are_dims_broadcast_compatible(a: int, b: int) → bool",
        python_module="src.smt.broadcast_theory",
        notes="Direct correspondence: same boolean expression",
    ),
    TypeCorrespondence(
        lean_type="strideConsistent h_in pad kernel stride h_out",
        lean_description="stride > 0 ∧ h_out = (h_in + 2*pad - kernel)/stride + 1",
        python_type="Conv2d output dim formula in model_checker + stride_theory",
        python_module="src.smt.stride_theory / src.model_checker",
        notes="Same integer arithmetic formula in both",
    ),
    TypeCorrespondence(
        lean_type="deviceConsistent n devices",
        lean_description="∀ i j, devices[i] == devices[j]",
        python_type="DevicePropagator.same_device(a, b) → Z3 constraint",
        python_module="src.smt.device_theory",
        notes="Lean quantifies over Fin; Python uses Z3 UserPropagateBase",
    ),
    TypeCorrespondence(
        lean_type="matmulConsistent k_a k_b",
        lean_description="k_a = k_b (inner dims match)",
        python_type="matmul_compatible(shape_a, shape_b) → Z3 constraint",
        python_module="src.smt.broadcast_theory",
        notes="shape_a[-1] == shape_b[-2] in Python",
    ),
    TypeCorrespondence(
        lean_type="CEGARState { numActive, converged }",
        lean_description="Loop state with active predicate count",
        python_type="ShapeCEGARLoop state (pred_set, iterations)",
        python_module="src.shape_cegar",
        notes="Lean abstract; Python concrete PredicateSet + iteration counter",
    ),
    TypeCorrespondence(
        lean_type="SubsetProduct weights target",
        lean_description="∃ mask, ∏(if mask[i] then w[i] else 1) = target",
        python_type="Brute-force subset product check",
        python_module="tests.test_lean_conformance",
        notes="Lean proves ↔ ReshapeDimSat; tests verify both directions",
    ),
    TypeCorrespondence(
        lean_type="Arrangement k",
        lean_description="Function classOf : Fin k → Nat with bound",
        python_type="List[int] (partition assignment)",
        python_module="src.smt.theory_combination",
        notes="_enumerate_partitions generates restricted growth strings",
    ),
    TypeCorrespondence(
        lean_type="TheorySolver k",
        lean_description="Abstract solver with isSatisfiable and sound",
        python_type="TheorySolver dataclass (name, solver, domain_kind)",
        python_module="src.smt.theory_combination",
        notes="Lean typeclass; Python dataclass wrapping Z3 Solver",
    ),
    TypeCorrespondence(
        lean_type="ProductTheory { shapeSig, deviceSig, phaseSig }",
        lean_description="TensorGuard's three-theory product",
        python_type="TheoryCombination with shape/device/phase solvers",
        python_module="src.smt.theory_combination",
        notes="Lean struct with disjointness proofs; Python runtime combination",
    ),
    TypeCorrespondence(
        lean_type="phaseConsistent isTraining hasDropout",
        lean_description="hasDropout → isTraining = true",
        python_type="Phase constraint in model_checker",
        python_module="src.model_checker",
        notes="Lean Prop; Python boolean check",
    ),
    TypeCorrespondence(
        lean_type="mhaConsistent embed_dim num_heads",
        lean_description="num_heads > 0 ∧ embed_dim % num_heads = 0",
        python_type="MHA divisibility check in model_checker",
        python_module="src.model_checker",
        notes="Same arithmetic condition",
    ),
    TypeCorrespondence(
        lean_type="UserPropagatorSpec",
        lean_description="semanticConsistency / checkerResult / soundness",
        python_type="Z3 UserPropagateBase subclass",
        python_module="src.smt.broadcast_theory",
        notes="Lean separates spec from checker; Python implements checker directly",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Theorem → Test Mapping
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TheoremTestMapping:
    """Maps a Lean theorem to its conformance tests."""
    lean_theorem: str
    lean_line: int
    lean_statement: str
    test_class: str
    test_methods: List[str]
    coverage_notes: str = ""


THEOREM_TEST_MAP: List[TheoremTestMapping] = [
    TheoremTestMapping(
        lean_theorem="broadcast_sound",
        lean_line=214,
        lean_statement="∃ result, ∀ i, result i = max (a i) (b i)",
        test_class="TestBroadcastSound",
        test_methods=[
            "test_broadcast_output_is_max",
            "test_broadcast_sound_random",
            "test_broadcast_sound_z3",
        ],
        coverage_notes=(
            "Parametric tests + 50 random shapes + Z3 propagator verification"
        ),
    ),
    TheoremTestMapping(
        lean_theorem="broadcast_symmetric",
        lean_line=250,
        lean_statement="broadcastConsistent n a b → broadcastConsistent n b a",
        test_class="TestBroadcastSymmetric",
        test_methods=[
            "test_symmetry_compatible",
            "test_symmetry_incompatible",
            "test_symmetry_output_shape",
            "test_symmetry_dim_level",
        ],
        coverage_notes=(
            "Tests symmetry for compatible pairs, incompatible pairs, "
            "output shapes, and exhaustive dimension-level check"
        ),
    ),
    TheoremTestMapping(
        lean_theorem="broadcast_assoc",
        lean_line=599,
        lean_statement=(
            "broadcastResult (broadcastResult a b) c = "
            "broadcastResult a (broadcastResult b c)"
        ),
        test_class="TestBroadcastAssoc",
        test_methods=[
            "test_assoc_dim_level",
            "test_assoc_shape_level",
            "test_assoc_via_propagator",
        ],
        coverage_notes=(
            "Dimension-level parametric + 30 random shapes + "
            "BroadcastPropagator.verify_broadcast_associativity"
        ),
    ),
    TheoremTestMapping(
        lean_theorem="broadcast_idempotent / broadcastResult_idempotent",
        lean_line=530,
        lean_statement="broadcastConsistent n a a ∧ broadcastResult n a a = a",
        test_class="TestBroadcastIdempotent",
        test_methods=[
            "test_idempotent_dim_level",
            "test_idempotent_shape_level",
            "test_idempotent_z3",
        ],
        coverage_notes="Exhaustive dims 1..19, 20 random shapes, Z3 verification",
    ),
    TheoremTestMapping(
        lean_theorem="broadcastDim_sound / broadcastDim_complete",
        lean_line=444,
        lean_statement=(
            "broadcastDimCheck a b = true ↔ broadcastDimSpec a b"
        ),
        test_class="TestBroadcastDimSoundComplete",
        test_methods=["test_sound_and_complete"],
        coverage_notes="Exhaustive 10×10 grid checking checker ↔ spec",
    ),
    TheoremTestMapping(
        lean_theorem="stride_sound",
        lean_line=227,
        lean_statement="h_out = (h_in + 2*pad - kernel) / stride + 1",
        test_class="TestStrideSoundness",
        test_methods=[
            "test_conv_output_formula",
            "test_stride_contiguous_layout",
            "test_stride_sound_z3",
            "test_conv_output_random",
        ],
        coverage_notes=(
            "6 parametric conv configs + contiguous stride verification + "
            "Z3 formula check + 30 random convolutions"
        ),
    ),
    TheoremTestMapping(
        lean_theorem="device_consistent_transitive",
        lean_line=237,
        lean_statement="devices i = devices k (via transitivity)",
        test_class="TestDeviceConsistentTransitive",
        test_methods=[
            "test_transitive_same_device_chain",
            "test_transitive_conflict",
            "test_same_device_for_each",
            "test_inherit_preserves_device",
            "test_transfer_overrides_device",
        ],
        coverage_notes=(
            "Transitive chain, conflict detection, all 5 device values, "
            "inherit and transfer operations"
        ),
    ),
    TheoremTestMapping(
        lean_theorem="matmul_sound",
        lean_line=283,
        lean_statement="matmulConsistent k_a k_b → k_a = k_b",
        test_class="TestMatmulSound",
        test_methods=[
            "test_matmul_inner_dims_match",
            "test_matmul_inner_dims_mismatch",
            "test_matmul_chain_dims",
            "test_matmul_batch_preserved",
        ],
        coverage_notes="SAT/UNSAT Z3 checks + dimension chain rule + batch preservation",
    ),
    TheoremTestMapping(
        lean_theorem="linear_output_dim",
        lean_line=324,
        lean_statement="x_last = in_features ∧ out_features = out_features",
        test_class="TestLinearOutputDim",
        test_methods=["test_linear_constraint"],
        coverage_notes="Direct constraint check",
    ),
    TheoremTestMapping(
        lean_theorem="mha_head_dim_sound",
        lean_line=347,
        lean_statement="num_heads * (embed_dim / num_heads) + embed_dim % num_heads = embed_dim",
        test_class="TestMHAHeadDim",
        test_methods=["test_mha_divisibility"],
        coverage_notes="5 standard MHA configurations",
    ),
    TheoremTestMapping(
        lean_theorem="cegar_terminates",
        lean_line=660,
        lean_statement=(
            "∃ k ≤ N, (iterN step k s₀).converged = true ∨ "
            "(iterN step k s₀).numActive = N"
        ),
        test_class="TestCEGARTerminates",
        test_methods=[
            "test_cegar_terminates_simple_model",
            "test_cegar_terminates_within_bound",
            "test_cegar_monotone_predicate_growth",
            "test_cegar_finite_universe_bound",
        ],
        coverage_notes=(
            "Real nn.Module CEGAR runs + budget bound check + "
            "monotone growth verification + abstract simulation"
        ),
    ),
    TheoremTestMapping(
        lean_theorem="reshape_np_hard (subset_product_forward + reverse)",
        lean_line=824,
        lean_statement="SubsetProduct weights T ↔ ReshapeDimSat weights T",
        test_class="TestReshapeNPHard",
        test_methods=[
            "test_equivalence",
            "test_forward_direction",
            "test_reverse_direction",
            "test_negative_case",
        ],
        coverage_notes=(
            "6 parametric instances + forward/reverse directions + negative case"
        ),
    ),
    TheoremTestMapping(
        lean_theorem="combination_soundness / tensorguard_combination_sound",
        lean_line=104,
        lean_statement=(
            "combinationSAT solvers arrangements → "
            "combinedSatisfiable solvers"
        ),
        test_class="TestCombinationSoundness",
        test_methods=[
            "test_consistent_arrangement_exists",
            "test_inconsistent_theories",
            "test_finite_domain_arrangement",
            "test_arrangement_count_bounded",
        ],
        coverage_notes=(
            "Consistent/inconsistent theory pairs + finite-domain "
            "arrangement + Stirling number bound"
        ),
    ),
    TheoremTestMapping(
        lean_theorem="phaseConsistent",
        lean_line=204,
        lean_statement="hasDropout → isTraining = true",
        test_class="TestPhaseConsistent",
        test_methods=["test_phase_consistency"],
        coverage_notes="4-case truth table (all combinations of training × dropout)",
    ),
    TheoremTestMapping(
        lean_theorem="propagator_output_sound",
        lean_line=510,
        lean_statement=(
            "checkerResult = true → ∃ result, result i = max(a_i, b_i)"
        ),
        test_class="TestPropagatorOutputSound",
        test_methods=["test_propagator_produces_max"],
        coverage_notes="4 dimension pairs verified through Z3 propagator",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

def get_summary() -> Dict:
    """Return a summary dict of the conformance mapping."""
    total_tests = sum(len(m.test_methods) for m in THEOREM_TEST_MAP)
    return {
        "lean_file": "lean/TheoryCombination.lean",
        "lean_lines": 922,
        "lean_theorems_covered": len(THEOREM_TEST_MAP),
        "python_test_file": "tests/test_lean_conformance.py",
        "total_test_methods": total_tests,
        "type_correspondences": len(TYPE_CORRESPONDENCE),
        "sorry_obligations": 0,
        "approach": "conformance_testing",
        "rationale": (
            "Since Lean 4 does not support extraction to Python, "
            "and bisimulation requires a shared formal semantics, "
            "we bridge the gap via systematic conformance testing: "
            "for each proved property, we create tests that exercise "
            "the Python code against the property using concrete, "
            "parametric, and random inputs."
        ),
    }


def print_summary() -> None:
    """Print a human-readable conformance summary."""
    s = get_summary()
    print("=" * 72)
    print("Lean 4 ↔ Python Conformance Summary")
    print("=" * 72)
    print(f"Lean file:              {s['lean_file']} ({s['lean_lines']} lines)")
    print(f"Theorems covered:       {s['lean_theorems_covered']}")
    print(f"Total test methods:     {s['total_test_methods']}")
    print(f"Type correspondences:   {s['type_correspondences']}")
    print(f"Sorry obligations:      {s['sorry_obligations']}")
    print(f"Approach:               {s['approach']}")
    print()
    print("Theorem → Test Mapping:")
    print("-" * 72)
    for m in THEOREM_TEST_MAP:
        print(f"  {m.lean_theorem} (line {m.lean_line})")
        print(f"    → {m.test_class}")
        for t in m.test_methods:
            print(f"      • {t}")
        print(f"    Coverage: {m.coverage_notes}")
        print()
    print("Type Correspondence Table:")
    print("-" * 72)
    for tc in TYPE_CORRESPONDENCE:
        print(f"  Lean:   {tc.lean_type}")
        print(f"  Python: {tc.python_type}")
        print(f"  Module: {tc.python_module}")
        if tc.notes:
            print(f"  Notes:  {tc.notes}")
        print()


if __name__ == "__main__":
    print_summary()
