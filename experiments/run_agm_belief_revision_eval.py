"""
AGM Belief Revision Evaluation.

Evaluates the impact of AGM belief revision (contraction, revision,
stale invalidation) on knowledge base precision across schema drift.

Protocol
--------
1. Build up a KB across multiple verification sessions for a ResNet family.
2. Introduce "schema drift" — architecture changes that invalidate old
   predicates (e.g. changing hidden dim from 64→128, changing output classes).
3. Measure KB precision with and without belief revision.
4. Save results to ``agm_belief_revision_results.json``.

Reference: Gärdenfors, P. (1988). *Knowledge in Flux*.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Any, Dict, List

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.knowledge_base import (
    VerificationKnowledgeBase,
    compute_arch_hash,
    PredicateEntry,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Model source generators
# ═══════════════════════════════════════════════════════════════════════════════

def _make_model_source(
    name: str, hidden: int, num_classes: int, num_blocks: int = 2,
    use_dropout: bool = False,
) -> str:
    """Generate nn.Module source for a model variant."""
    layers = []
    layers.append(f"        self.conv1 = nn.Conv2d(3, {hidden}, kernel_size=3, padding=1)")
    layers.append(f"        self.bn1 = nn.BatchNorm2d({hidden})")
    layers.append(f"        self.relu = nn.ReLU()")
    for i in range(num_blocks):
        layers.append(f"        self.conv{i+2} = nn.Conv2d({hidden}, {hidden}, kernel_size=3, padding=1)")
        layers.append(f"        self.bn{i+2} = nn.BatchNorm2d({hidden})")
    if use_dropout:
        layers.append(f"        self.dropout = nn.Dropout(0.5)")
    layers.append(f"        self.fc = nn.Linear({hidden}, {num_classes})")

    forward_lines = []
    forward_lines.append("        x = self.relu(self.bn1(self.conv1(x)))")
    for i in range(num_blocks):
        forward_lines.append(f"        x = self.relu(self.bn{i+2}(self.conv{i+2}(x)))")
    forward_lines.append("        x = x.mean(dim=[2, 3])")
    if use_dropout:
        forward_lines.append("        x = self.dropout(x)")
    forward_lines.append("        x = self.fc(x)")
    forward_lines.append("        return x")

    return f"""import torch
import torch.nn as nn

class {name}(nn.Module):
    def __init__(self):
        super().__init__()
{chr(10).join(layers)}

    def forward(self, x):
{chr(10).join(forward_lines)}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Schema evolution phases
# ═══════════════════════════════════════════════════════════════════════════════

def _predicates_for_model(hidden: int, num_classes: int,
                          use_dropout: bool = False) -> List[str]:
    """Generate the ground-truth predicates for a model configuration."""
    preds = [
        f"x.shape[1] == 3",
        f"conv1.weight.shape[0] == {hidden}",
        f"conv1.weight.shape[1] == 3",
        f"bn1.num_features == {hidden}",
        f"fc.in_features == {hidden}",
        f"fc.out_features == {num_classes}",
        f"output.shape[-1] == {num_classes}",
    ]
    if use_dropout:
        preds.append("dropout.p == 0.5")
    return preds


# Phase definitions: (phase_name, hidden, num_classes, use_dropout)
PHASES = [
    ("v1_initial",       64,  10,  False),
    ("v2_wider",         128, 10,  False),
    ("v3_more_classes",  128, 100, False),
    ("v4_dropout",       128, 100, True),
    ("v5_narrow",        32,  100, True),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _build_kb_across_phases(
    phases: List[tuple],
    apply_revision: bool,
) -> List[Dict[str, Any]]:
    """Build KB across phases, optionally applying AGM revision.

    Returns per-phase metrics.
    """
    kb = VerificationKnowledgeBase()
    arch_hash = "eval_family_hash"
    results = []

    base_time = time.time()

    for idx, (phase_name, hidden, num_classes, use_dropout) in enumerate(phases):
        current_predicates = _predicates_for_model(hidden, num_classes, use_dropout)

        if apply_revision:
            # AGM revision: revise each new predicate (contracts contradictions)
            for pred in current_predicates:
                kb.revise(arch_hash, pred)

            # Invalidate stale predicates with aggressive thresholds for eval
            # (threshold_age=0 so age is not a barrier, min_uses=0)
            kb.invalidate_stale(
                arch_hash, threshold_age=0, min_uses=1000,
                entrenchment_threshold=0.15,
            )
        else:
            # Naive expansion only (original behavior)
            kb.record(arch_hash, predicates=current_predicates)

        # Measure precision against current ground truth
        test_cases = [{"valid_predicates": current_predicates}]
        precision_result = kb.measure_kb_precision(arch_hash, test_cases)

        record = kb.get_family_record(arch_hash)
        total_stored = len(record.predicates) if record else 0

        results.append({
            "phase": phase_name,
            "phase_index": idx,
            "hidden": hidden,
            "num_classes": num_classes,
            "use_dropout": use_dropout,
            "ground_truth_count": len(current_predicates),
            "stored_count": total_stored,
            "precision": precision_result["precision"],
            "recall": precision_result["recall"],
            "valid_count": precision_result["valid_count"],
            "invalid_count": precision_result["invalid_count"],
        })

    return results


def run_evaluation() -> Dict[str, Any]:
    """Run full AGM belief revision evaluation."""
    print("=" * 70)
    print("AGM Belief Revision Evaluation")
    print("=" * 70)

    # Run without revision (baseline)
    print("\n[1/2] Running baseline (expansion only, no revision)...")
    baseline_results = _build_kb_across_phases(PHASES, apply_revision=False)

    # Run with revision
    print("[2/2] Running with AGM belief revision...")
    revision_results = _build_kb_across_phases(PHASES, apply_revision=True)

    # Compute summary statistics
    baseline_precisions = [r["precision"] for r in baseline_results]
    revision_precisions = [r["precision"] for r in revision_results]

    avg_baseline = sum(baseline_precisions) / len(baseline_precisions)
    avg_revision = sum(revision_precisions) / len(revision_precisions)

    # Final-phase comparison (most drift accumulated)
    final_baseline = baseline_results[-1]
    final_revision = revision_results[-1]

    summary = {
        "avg_precision_baseline": round(avg_baseline, 4),
        "avg_precision_revision": round(avg_revision, 4),
        "precision_improvement": round(avg_revision - avg_baseline, 4),
        "final_phase_baseline_precision": final_baseline["precision"],
        "final_phase_revision_precision": final_revision["precision"],
        "final_phase_baseline_stored": final_baseline["stored_count"],
        "final_phase_revision_stored": final_revision["stored_count"],
        "final_phase_baseline_invalid": final_baseline["invalid_count"],
        "final_phase_revision_invalid": final_revision["invalid_count"],
        "num_phases": len(PHASES),
    }

    output = {
        "experiment": "agm_belief_revision",
        "reference": "Gärdenfors 1988, Knowledge in Flux",
        "description": (
            "Evaluates AGM belief revision (contraction + Levi identity) "
            "for preventing stale predicate accumulation under schema drift."
        ),
        "phases": [p[0] for p in PHASES],
        "baseline_results": baseline_results,
        "revision_results": revision_results,
        "summary": summary,
    }

    # Print summary
    print("\n" + "─" * 70)
    print("Results Summary")
    print("─" * 70)
    print(f"  Phases evaluated:             {len(PHASES)}")
    print(f"  Avg precision (baseline):     {avg_baseline:.4f}")
    print(f"  Avg precision (AGM revision): {avg_revision:.4f}")
    print(f"  Precision improvement:        {avg_revision - avg_baseline:+.4f}")
    print(f"  Final phase stored (baseline):{final_baseline['stored_count']}")
    print(f"  Final phase stored (revision):{final_revision['stored_count']}")
    print(f"  Final phase invalid (baseline):{final_baseline['invalid_count']}")
    print(f"  Final phase invalid (revision):{final_revision['invalid_count']}")

    # Per-phase detail
    print("\n  Per-phase precision:")
    print(f"  {'Phase':<20} {'Baseline':>10} {'Revision':>10} {'Δ':>10}")
    for b, r in zip(baseline_results, revision_results):
        delta = r["precision"] - b["precision"]
        print(f"  {b['phase']:<20} {b['precision']:>10.4f} {r['precision']:>10.4f} {delta:>+10.4f}")

    return output


def main() -> None:
    results = run_evaluation()

    out_path = os.path.join(os.path.dirname(__file__), "agm_belief_revision_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
