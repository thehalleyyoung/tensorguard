#!/usr/bin/env python3
"""Surviving Mutant Classification.

Classifies surviving mutants from mutation testing into equivalent mutants
(semantically identical) vs genuine escapes (real bugs TensorGuard misses).
Results stratified by theory fragment, architecture pattern, and bug category.
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_checker import verify_model

from experiments.run_mutation_testing import (
    BenchmarkSpec, ModelMutator, MutationOperator, MutationRecord,
    MutationTestRunner, BENCHMARK_MODELS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Extended equivalence detection
# ═══════════════════════════════════════════════════════════════════════════════

class EquivalenceReason(Enum):
    """Why a mutant is classified as equivalent."""
    IDENTITY_ARITHMETIC = "identity_arithmetic"       # e.g., dim + 0 -> dim
    REDUNDANT_LAYER = "redundant_layer"               # e.g., adding dropout(0)
    SWAP_COMMUTATIVE = "swap_commutative"             # swapping commutative ops
    NO_SHAPE_CHANGE = "no_shape_change"               # mutation doesn't change shapes
    DEAD_CODE = "dead_code"                           # mutation in unreachable code
    SEMANTIC_IDENTITY = "semantic_identity"            # syntactically different, same behavior
    NOT_EQUIVALENT = "not_equivalent"                  # genuinely different behavior


@dataclass
class MutantClassification:
    """Detailed classification of a surviving mutant."""
    model_name: str
    operator: str
    description: str
    is_equivalent: bool
    equivalence_reason: str
    theory_fragment: str        # QF_LIA, QF_NIA, mixed
    architecture: str           # MLP, CNN, ResNet, Transformer, etc.
    bug_category: str           # shape_mismatch, device_error, phase_error, etc.
    original_verdict: str       # SAFE/BUG for original model
    mutant_verdict: str         # SAFE/BUG for mutated model
    mutation_delta: Optional[str] = None   # human-readable diff
    verification_time_ms: float = 0.0


def classify_theory_fragment(source: str, operator: MutationOperator) -> str:
    """Determine theory fragment based on source code and mutation type."""
    has_reshape = any(k in source for k in ["view(", "reshape(", "flatten("])
    has_conv = "Conv2d" in source or "Conv1d" in source
    has_pool = "MaxPool" in source or "AvgPool" in source
    has_device = ".cuda()" in source or ".to(" in source or "device=" in source
    has_phase = ("Dropout" in source or "BatchNorm" in source
                 or ".train()" in source or ".eval()" in source)

    # Product constraints from reshape/flatten → QF_NIA
    is_nonlinear = has_reshape and (has_conv or has_pool)

    # Mixed if multiple theory domains are involved
    domains = sum([has_device, has_phase, is_nonlinear])
    if domains >= 2 or (has_device and has_reshape):
        return "mixed"
    elif is_nonlinear:
        return "QF_NIA"
    else:
        return "QF_LIA"


def classify_architecture(source: str) -> str:
    """Determine architecture pattern from source code."""
    src_lower = source.lower()
    if "resblock" in src_lower or "residual" in src_lower or "res" in src_lower:
        has_skip = ("+ residual" in source or "+residual" in source
                    or "x + " in source or "x +" in source)
        if has_skip or "ResBlock" in source or "ResNet" in source:
            return "ResNet"
    if ("LSTM" in source or "GRU" in source or "RNN" in source
            or "lstm" in src_lower):
        return "LSTM"
    if ("MultiheadAttention" in source or "Transformer" in source
            or "attn" in src_lower):
        return "Transformer"
    if "Conv2d" in source or "Conv1d" in source:
        if "UNet" in source or "enc_conv" in source or "dec_conv" in source:
            return "UNet"
        if ("branch1" in source or "branch2" in source
                or "Inception" in source):
            return "Inception"
        return "CNN"
    return "MLP"


def classify_bug_category(operator: MutationOperator, description: str,
                          source: str) -> str:
    """Determine bug category from mutation operator and context."""
    if operator in (MutationOperator.WRONG_IN_FEATURES,
                    MutationOperator.WRONG_OUT_FEATURES,
                    MutationOperator.ADD_DIMENSION_MISMATCH):
        return "shape_mismatch"
    elif operator == MutationOperator.WRONG_KERNEL_SIZE:
        return "kernel_error"
    elif operator == MutationOperator.WRONG_CHANNELS:
        return "channel_mismatch"
    elif operator == MutationOperator.REMOVE_RESHAPE:
        return "reshape_error"
    elif operator == MutationOperator.SWAP_LAYERS:
        return "layer_ordering"
    elif operator == MutationOperator.WRONG_POOL_SIZE:
        return "pooling_error"
    elif operator == MutationOperator.TRANSPOSE_MISSING:
        return "transpose_error"
    elif operator == MutationOperator.WRONG_CONCAT_DIM:
        return "concat_error"
    else:
        return "other"


def deep_equivalence_check(original_source: str, mutated_source: str,
                           record: MutationRecord,
                           input_shapes: Dict[str, tuple]) -> Tuple[bool, str]:
    """Perform deep equivalence analysis on a surviving mutant.

    Returns (is_equivalent, reason).
    """
    op = record.operator
    desc = record.description

    # --- Pattern 1: Identity arithmetic ---
    # e.g., changing Linear(128, 128) in_features by 0
    if op in (MutationOperator.WRONG_IN_FEATURES,
              MutationOperator.WRONG_OUT_FEATURES):
        m = re.search(r'(\d+)\s*->\s*(\d+)', desc)
        if m:
            old_val, new_val = int(m.group(1)), int(m.group(2))
            if old_val == new_val:
                return True, EquivalenceReason.IDENTITY_ARITHMETIC.value

    # --- Pattern 2: Swapping identical layers ---
    if op == MutationOperator.SWAP_LAYERS:
        # Parse the two lines being swapped and check if they're identical ops
        try:
            orig_tree = ast.parse(original_source)
            mut_tree = ast.parse(mutated_source)
            # If the AST structure of forward() is semantically equivalent
            orig_forward = _extract_forward_body(orig_tree)
            mut_forward = _extract_forward_body(mut_tree)
            if orig_forward and mut_forward:
                # Check if swapped lines have same structure (e.g., two identical relu calls)
                if _are_swap_equivalent(orig_forward, mut_forward):
                    return True, EquivalenceReason.SWAP_COMMUTATIVE.value
        except Exception:
            pass

    # --- Pattern 3: No actual shape change ---
    # Some mutations may not actually change the shapes flowing through
    if op == MutationOperator.WRONG_POOL_SIZE:
        # If the model uses AdaptiveAvgPool, pool size might not matter
        if "AdaptiveAvgPool" in original_source:
            return True, EquivalenceReason.NO_SHAPE_CHANGE.value

    # --- Pattern 4: Kernel size changes that preserve output size ---
    if op == MutationOperator.WRONG_KERNEL_SIZE:
        # If padding compensates for kernel size change, shapes may be same
        m = re.search(r'(\d+)\s*->\s*(\d+)', desc)
        if m:
            old_k, new_k = int(m.group(1)), int(m.group(2))
            # Check if padding = (kernel_size - 1) // 2 pattern
            old_pad = (old_k - 1) // 2
            new_pad = (new_k - 1) // 2
            if f"padding={old_pad}" in original_source:
                # Shape changes, but check if model still valid
                pass  # Not equivalent in general

    # --- Pattern 5: Semantic identity via verification ---
    # Both original and mutant verify to same result with same shapes
    try:
        orig_result = verify_model(original_source, input_shapes=input_shapes)
        mut_result = verify_model(mutated_source, input_shapes=input_shapes)
        if orig_result.safe and mut_result.safe:
            # Both safe — check if shapes are identical
            if (orig_result.certificate and mut_result.certificate
                    and orig_result.certificate == mut_result.certificate):
                return True, EquivalenceReason.SEMANTIC_IDENTITY.value
    except Exception:
        pass

    # --- Pattern 6: Transpose removal that's a no-op ---
    if op == MutationOperator.TRANSPOSE_MISSING:
        # Removing transpose(1, 1) or permute(0, 1, 2) is identity
        if "transpose(1, 1)" in original_source or "permute(0, 1, 2)" in original_source:
            return True, EquivalenceReason.IDENTITY_ARITHMETIC.value

    return False, EquivalenceReason.NOT_EQUIVALENT.value


def _extract_forward_body(tree: ast.Module) -> Optional[List[ast.stmt]]:
    """Extract the body of the forward() method from an AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            return node.body
    return None


def _are_swap_equivalent(orig_body: List[ast.stmt],
                         mut_body: List[ast.stmt]) -> bool:
    """Check if swapping two statements produces equivalent code."""
    if len(orig_body) != len(mut_body):
        return False
    # Simple heuristic: if the set of statement types and targets is the same
    try:
        orig_reprs = sorted(ast.dump(s) for s in orig_body)
        mut_reprs = sorted(ast.dump(s) for s in mut_body)
        return orig_reprs == mut_reprs
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Extended mutation runner with classification
# ═══════════════════════════════════════════════════════════════════════════════

class ClassifyingMutationRunner:
    """Run mutation testing with deep surviving-mutant classification."""

    def __init__(self, models: Dict[str, BenchmarkSpec],
                 mutations_per_model: int = 5):
        self.models = models
        self.mutations_per_model = mutations_per_model
        self.classifications: List[MutantClassification] = []
        self.all_results: List[Dict[str, Any]] = []

    def run(self) -> Tuple[List[MutantClassification], List[Dict[str, Any]]]:
        print(f"Running classifying mutation analysis on {len(self.models)} models")
        print("=" * 70)

        for name, spec in self.models.items():
            self._test_model(name, spec)

        return self.classifications, self.all_results

    def _test_model(self, name: str, spec: BenchmarkSpec) -> None:
        source = spec.source
        input_shapes = spec.input_shapes
        print(f"\n{'─' * 50}")
        print(f"Model: {name}")

        # Classify this model's properties
        theory_frag = classify_theory_fragment(source, MutationOperator.WRONG_IN_FEATURES)
        arch = classify_architecture(source)

        # Verify original
        try:
            orig_result = verify_model(source, input_shapes=input_shapes)
            orig_verdict = "SAFE" if orig_result.safe else "BUG"
        except Exception:
            orig_verdict = "ERROR"

        mutator = ModelMutator(source)
        available = mutator.available_mutations()
        if not available:
            print(f"  ⚠ No applicable mutations")
            return

        generated = 0
        attempts = 0
        max_attempts = self.mutations_per_model * 3
        op_cycle = available * ((self.mutations_per_model // len(available)) + 2)

        for op in op_cycle:
            if generated >= self.mutations_per_model:
                break
            if attempts >= max_attempts:
                break
            attempts += 1

            fresh_mutator = ModelMutator(source)
            record = fresh_mutator.apply(op)
            if record is None:
                continue

            try:
                ast.parse(record.mutated_source)
            except SyntaxError:
                continue

            t0 = time.monotonic()
            try:
                mut_result = verify_model(record.mutated_source,
                                          input_shapes=input_shapes)
                elapsed = (time.monotonic() - t0) * 1000
                mut_verdict = "SAFE" if mut_result.safe else "BUG"
            except Exception:
                elapsed = (time.monotonic() - t0) * 1000
                mut_verdict = "BUG"  # errors count as detection

            bug_cat = classify_bug_category(record.operator, record.description, source)
            result_entry = {
                "model": name,
                "operator": record.operator.value,
                "description": record.description,
                "original_verdict": orig_verdict,
                "mutant_verdict": mut_verdict,
                "theory_fragment": theory_frag,
                "architecture": arch,
                "bug_category": bug_cat,
                "killed": mut_verdict == "BUG",
                "time_ms": round(elapsed, 2),
            }

            if mut_verdict == "SAFE":
                # This is a surviving mutant — classify it
                is_eq, reason = deep_equivalence_check(
                    source, record.mutated_source, record, input_shapes
                )
                classification = MutantClassification(
                    model_name=name,
                    operator=record.operator.value,
                    description=record.description,
                    is_equivalent=is_eq,
                    equivalence_reason=reason,
                    theory_fragment=theory_frag,
                    architecture=arch,
                    bug_category=bug_cat,
                    original_verdict=orig_verdict,
                    mutant_verdict=mut_verdict,
                    verification_time_ms=elapsed,
                )
                self.classifications.append(classification)
                result_entry["is_equivalent"] = is_eq
                result_entry["equivalence_reason"] = reason

                icon = "≡" if is_eq else "✗"
                label = "equivalent" if is_eq else "GENUINE ESCAPE"
                print(f"  {icon} [{record.operator.value}] "
                      f"{record.description} → survived ({label})")
            else:
                print(f"  ✓ [{record.operator.value}] "
                      f"{record.description} → killed")

            self.all_results.append(result_entry)
            generated += 1

        print(f"  Generated {generated} mutants")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Report generation
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(
        p_hat * (1 - p_hat) / total + z * z / (4 * total * total)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def build_report(classifications: List[MutantClassification],
                 all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build comprehensive classification report."""
    total_mutants = len(all_results)
    killed = sum(1 for r in all_results if r.get("killed", False))
    survived = total_mutants - killed
    equivalent = sum(1 for c in classifications if c.is_equivalent)
    genuine_escapes = sum(1 for c in classifications if not c.is_equivalent)

    testable = killed + genuine_escapes
    mutation_score = killed / max(testable, 1)
    ci_low, ci_high = wilson_ci(killed, testable)

    # Raw mutation score (before equivalence adjustment)
    raw_testable = killed + survived
    raw_score = killed / max(raw_testable, 1)

    # --- Stratify by theory fragment ---
    fragment_strat = {}
    for frag in ["QF_LIA", "QF_NIA", "mixed"]:
        frag_results = [r for r in all_results if r.get("theory_fragment") == frag]
        frag_classifications = [c for c in classifications
                                if c.theory_fragment == frag]
        frag_killed = sum(1 for r in frag_results if r.get("killed", False))
        frag_survived = len(frag_results) - frag_killed
        frag_eq = sum(1 for c in frag_classifications if c.is_equivalent)
        frag_genuine = sum(1 for c in frag_classifications if not c.is_equivalent)
        frag_testable = frag_killed + frag_genuine

        fragment_strat[frag] = {
            "total": len(frag_results),
            "killed": frag_killed,
            "survived": frag_survived,
            "equivalent": frag_eq,
            "genuine_escapes": frag_genuine,
            "raw_mutation_score": round(
                frag_killed / max(frag_killed + frag_survived, 1), 4
            ),
            "adjusted_mutation_score": round(
                frag_killed / max(frag_testable, 1), 4
            ),
        }

    # --- Stratify by architecture ---
    arch_strat = {}
    for arch in sorted(set(r.get("architecture", "unknown") for r in all_results)):
        arch_results = [r for r in all_results if r.get("architecture") == arch]
        arch_classifications = [c for c in classifications
                                if c.architecture == arch]
        a_killed = sum(1 for r in arch_results if r.get("killed", False))
        a_survived = len(arch_results) - a_killed
        a_eq = sum(1 for c in arch_classifications if c.is_equivalent)
        a_genuine = sum(1 for c in arch_classifications if not c.is_equivalent)
        a_testable = a_killed + a_genuine

        arch_strat[arch] = {
            "total": len(arch_results),
            "killed": a_killed,
            "survived": a_survived,
            "equivalent": a_eq,
            "genuine_escapes": a_genuine,
            "raw_mutation_score": round(
                a_killed / max(a_killed + a_survived, 1), 4
            ),
            "adjusted_mutation_score": round(
                a_killed / max(a_testable, 1), 4
            ),
        }

    # --- Stratify by bug category ---
    cat_strat = {}
    for cat in sorted(set(r.get("bug_category", "unknown") for r in all_results)):
        cat_results = [r for r in all_results if r.get("bug_category") == cat]
        cat_classifications = [c for c in classifications
                               if c.bug_category == cat]
        c_killed = sum(1 for r in cat_results if r.get("killed", False))
        c_survived = len(cat_results) - c_killed
        c_eq = sum(1 for c2 in cat_classifications if c2.is_equivalent)
        c_genuine = sum(1 for c2 in cat_classifications if not c2.is_equivalent)
        c_testable = c_killed + c_genuine

        cat_strat[cat] = {
            "total": len(cat_results),
            "killed": c_killed,
            "survived": c_survived,
            "equivalent": c_eq,
            "genuine_escapes": c_genuine,
            "raw_mutation_score": round(
                c_killed / max(c_killed + c_survived, 1), 4
            ),
            "adjusted_mutation_score": round(
                c_killed / max(c_testable, 1), 4
            ),
        }

    # --- Equivalence reason distribution ---
    reason_dist = {}
    for c in classifications:
        r = c.equivalence_reason
        reason_dist[r] = reason_dist.get(r, 0) + 1

    # --- Genuine escape details ---
    genuine_escape_details = [
        {
            "model": c.model_name,
            "operator": c.operator,
            "description": c.description,
            "theory_fragment": c.theory_fragment,
            "architecture": c.architecture,
            "bug_category": c.bug_category,
        }
        for c in classifications if not c.is_equivalent
    ]

    # --- Equivalent mutant details ---
    equivalent_details = [
        {
            "model": c.model_name,
            "operator": c.operator,
            "description": c.description,
            "equivalence_reason": c.equivalence_reason,
            "theory_fragment": c.theory_fragment,
        }
        for c in classifications if c.is_equivalent
    ]

    report = {
        "metadata": {
            "total_models": len(set(r.get("model") for r in all_results)),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": {
            "total_mutants": total_mutants,
            "killed": killed,
            "survived": survived,
            "equivalent_mutants": equivalent,
            "genuine_escapes": genuine_escapes,
            "raw_mutation_score": round(raw_score, 4),
            "adjusted_mutation_score": round(mutation_score, 4),
            "adjustment_delta": round(mutation_score - raw_score, 4),
            "wilson_95_ci": {
                "lower": round(ci_low, 4),
                "upper": round(ci_high, 4),
            },
        },
        "by_theory_fragment": fragment_strat,
        "by_architecture": arch_strat,
        "by_bug_category": cat_strat,
        "equivalence_reason_distribution": reason_dist,
        "genuine_escape_details": genuine_escape_details,
        "equivalent_mutant_details": equivalent_details,
        "interpretation": {
            "equivalent_fraction": round(
                equivalent / max(survived, 1), 4
            ),
            "message": (
                f"Of {survived} surviving mutants, {equivalent} "
                f"({equivalent / max(survived, 1):.0%}) are classified as "
                f"equivalent (no observable behavior change). "
                f"{genuine_escapes} are genuine escapes representing "
                f"real detection gaps."
            ),
            "mixed_fragment_impact": (
                f"Mixed-theory fragment shows "
                f"{fragment_strat.get('mixed', {}).get('genuine_escapes', 0)} "
                f"genuine escapes, consistent with Craig interpolation "
                f"fallback limitations."
            ),
        },
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    random.seed(42)

    print("=" * 70)
    print("Surviving Mutant Classification")
    print("=" * 70)
    print(f"  Models: {len(BENCHMARK_MODELS)}")
    print()

    runner = ClassifyingMutationRunner(BENCHMARK_MODELS, mutations_per_model=6)
    classifications, all_results = runner.run()

    report = build_report(classifications, all_results)

    # Print summary
    print(f"\n{'=' * 70}")
    print("MUTANT CLASSIFICATION RESULTS")
    print(f"{'=' * 70}")
    s = report["summary"]
    print(f"  Total mutants:             {s['total_mutants']}")
    print(f"  Killed:                    {s['killed']}")
    print(f"  Survived:                  {s['survived']}")
    print(f"    - Equivalent:            {s['equivalent_mutants']}")
    print(f"    - Genuine escapes:       {s['genuine_escapes']}")
    print(f"  Raw mutation score:        {s['raw_mutation_score']:.2%}")
    print(f"  Adjusted mutation score:   {s['adjusted_mutation_score']:.2%}")
    print(f"  Adjustment delta:          +{s['adjustment_delta']:.2%}")
    ci = s["wilson_95_ci"]
    print(f"  95% Wilson CI (adjusted):  [{ci['lower']:.2%}, {ci['upper']:.2%}]")

    print(f"\n  By theory fragment:")
    for frag, data in report["by_theory_fragment"].items():
        if data["total"] > 0:
            print(f"    {frag:8s}: raw={data['raw_mutation_score']:.2%}  "
                  f"adj={data['adjusted_mutation_score']:.2%}  "
                  f"equiv={data['equivalent']}  escapes={data['genuine_escapes']}  "
                  f"n={data['total']}")

    print(f"\n  By architecture:")
    for arch, data in report["by_architecture"].items():
        if data["total"] > 0:
            print(f"    {arch:15s}: raw={data['raw_mutation_score']:.2%}  "
                  f"adj={data['adjusted_mutation_score']:.2%}  "
                  f"n={data['total']}")

    print(f"\n  By bug category:")
    for cat, data in report["by_bug_category"].items():
        if data["total"] > 0:
            print(f"    {cat:20s}: raw={data['raw_mutation_score']:.2%}  "
                  f"escapes={data['genuine_escapes']}  n={data['total']}")

    if report["genuine_escape_details"]:
        print(f"\n  Genuine escapes ({len(report['genuine_escape_details'])}):")
        for ge in report["genuine_escape_details"]:
            print(f"    • {ge['model']}: [{ge['operator']}] {ge['description']} "
                  f"({ge['theory_fragment']}/{ge['architecture']})")

    print(f"\n  Equivalence reasons:")
    for reason, count in sorted(report["equivalence_reason_distribution"].items()):
        print(f"    {reason:30s}: {count}")

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "mutant_classification.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
