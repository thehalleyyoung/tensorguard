"""
Knowledge Transfer Benchmark: Empirical cross-session verification speedup.

Measures the speedup obtained by transferring verification knowledge (predicates,
strategies, proof schemas) across sessions within an architectural family.

Protocol
--------
1. Define a ResNet family (ResNet-18, -34, -50, -101, -152) with torch.nn.
2. **Cold pass**: Verify each model from scratch, recording iterations, Z3 queries,
   wall-clock time, and predicates discovered.
3. Build a VerificationKnowledgeBase from cold-pass results.
4. **Warm pass**: Verify the same models with KB-primed CEGAR, recording same metrics.
5. Compute per-model and aggregate speedup ratios.
6. Run anti-unification to extract shared proof schemas.
7. Save detailed results to ``.benchmarks/knowledge_transfer_results.json``.

Results are REAL — if no speedup is observed the report says so honestly.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.knowledge_base import (
    VerificationKnowledgeBase,
    compute_arch_hash,
    anti_unify_proof_certificates,
)
from src.shape_cegar import ShapeCEGARLoop


# ═══════════════════════════════════════════════════════════════════════════════
# ResNet family definitions
# ═══════════════════════════════════════════════════════════════════════════════

def _make_resnet_source(
    name: str,
    block_counts: List[int],
    base_width: int,
    num_classes: int = 1000,
) -> str:
    """Generate nn.Module source for a ResNet variant.

    Uses residual (skip) connections and ends with view + Linear (no global
    pooling).  With symbolic spatial dimensions the view creates a symbolic
    flat dimension, forcing CEGAR to discover that the flattened size must
    match the Linear's in_features.  The predicate ``x.shape[-1] == in_features``
    transfers across the architectural family.

    All operations use propagators registered in model_checker.py:
    Conv2d, BatchNorm2d, ReLU, Linear, ADD, RESHAPE (view).
    """
    init_lines = []
    forward_lines = []

    init_lines.append(f"        self.conv1 = nn.Conv2d(3, {base_width}, kernel_size=3, padding=1)")
    init_lines.append(f"        self.bn1 = nn.BatchNorm2d({base_width})")
    init_lines.append(f"        self.relu = nn.ReLU()")

    block_idx = 0
    cur_width = base_width
    for stage, count in enumerate(block_counts):
        for b in range(count):
            init_lines.append(
                f"        self.conv{block_idx + 2} = nn.Conv2d({cur_width}, {cur_width}, kernel_size=3, padding=1)"
            )
            init_lines.append(f"        self.bn{block_idx + 2} = nn.BatchNorm2d({cur_width})")
            block_idx += 1

    init_lines.append(f"        self.fc = nn.Linear({cur_width}, {num_classes})")

    forward_lines.append("        x = self.relu(self.bn1(self.conv1(x)))")
    block_idx = 0
    for stage, count in enumerate(block_counts):
        for b in range(count):
            # Residual connection: save input, apply conv+bn+relu, add skip
            forward_lines.append(f"        residual = x")
            forward_lines.append(
                f"        x = self.relu(self.bn{block_idx + 2}(self.conv{block_idx + 2}(x)))"
            )
            forward_lines.append(f"        x = x + residual")
            block_idx += 1
    # Flatten with view — creates symbolic flat dim with symbolic spatial inputs
    forward_lines.append("        x = x.view(x.size(0), -1)")
    forward_lines.append("        x = self.fc(x)")
    forward_lines.append("        return x")

    return f"""import torch
import torch.nn as nn

class {name}(nn.Module):
    def __init__(self):
        super().__init__()
{chr(10).join(init_lines)}

    def forward(self, x):
{chr(10).join(forward_lines)}
"""


RESNET_FAMILY = [
    ("ResNet18",  [2, 2, 2, 2], 64,  1000),
    ("ResNet34",  [3, 4, 6, 3], 64,  1000),
    ("ResNet50",  [3, 4, 6, 3], 64,  1000),
    ("ResNet101", [3, 4, 23, 3], 64, 1000),
    ("ResNet152", [3, 8, 36, 3], 64, 1000),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Metric collection
# ═══════════════════════════════════════════════════════════════════════════════

def _run_verification(
    source: str,
    name: str,
    kb: Optional[VerificationKnowledgeBase] = None,
) -> Dict[str, Any]:
    """Run CEGAR verification and collect metrics."""
    t0 = time.monotonic()
    loop = ShapeCEGARLoop(
        source,
        input_shapes={"x": ("batch", 3, "H", "W")},
        knowledge_base=kb,
    )
    result = loop.run()
    elapsed_ms = (time.monotonic() - t0) * 1000

    # Count Z3 queries from the verification result
    z3_queries = 0
    if result.verification_result and result.verification_result.certificate:
        z3_queries = result.verification_result.certificate.z3_queries

    pred_strs = [p.pretty() for p in result.discovered_predicates]

    return {
        "name": name,
        "verdict": result.verdict.name,
        "iterations": result.iterations,
        "z3_queries": z3_queries,
        "time_ms": round(elapsed_ms, 2),
        "predicates_discovered": len(result.discovered_predicates),
        "predicate_strings": pred_strs,
        "total_time_ms": round(result.total_time_ms, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Cold pass
# ═══════════════════════════════════════════════════════════════════════════════

def run_cold_pass(
    sources: List[str],
    names: List[str],
) -> Tuple[List[Dict[str, Any]], VerificationKnowledgeBase]:
    """Verify all models from scratch (cold start), build KB from results."""
    kb = VerificationKnowledgeBase()
    results = []

    for name, source in zip(names, sources):
        arch_hash = compute_arch_hash(source)
        metrics = _run_verification(source, name, kb=None)
        metrics["pass"] = "cold"
        metrics["arch_hash"] = arch_hash

        # Record to KB for warm pass
        kb.record(
            arch_hash,
            predicates=metrics["predicate_strings"],
            strategies=[{
                "propagator_type": "shape_cegar",
                "iteration_count": metrics["iterations"],
                "total_time_ms": metrics["time_ms"],
            }],
            proof_certificate={
                "model_name": name,
                "properties": ["shape_safety"],
                "steps": [
                    {"rule": "verify", "conclusion": f"{name}_safe",
                     "dims": metrics["predicates_discovered"]},
                ],
                "certificate_hash": hashlib.sha256(
                    f"{name}_{metrics['verdict']}".encode()
                ).hexdigest(),
            },
        )

        results.append(metrics)
        print(f"  [cold] {name}: {metrics['verdict']} "
              f"({metrics['iterations']} iters, {metrics['z3_queries']} Z3q, "
              f"{metrics['time_ms']:.0f}ms, {metrics['predicates_discovered']} preds)")

    return results, kb


# ═══════════════════════════════════════════════════════════════════════════════
# Warm pass
# ═══════════════════════════════════════════════════════════════════════════════

def run_warm_pass(
    sources: List[str],
    names: List[str],
    kb: VerificationKnowledgeBase,
) -> List[Dict[str, Any]]:
    """Verify all models with KB loaded (warm start)."""
    results = []

    for name, source in zip(names, sources):
        arch_hash = compute_arch_hash(source)
        transferred = kb.lookup(arch_hash)

        metrics = _run_verification(source, name, kb=kb)
        metrics["pass"] = "warm"
        metrics["arch_hash"] = arch_hash
        metrics["predicates_transferred"] = len(transferred.predicates)

        results.append(metrics)
        print(f"  [warm] {name}: {metrics['verdict']} "
              f"({metrics['iterations']} iters, {metrics['z3_queries']} Z3q, "
              f"{metrics['time_ms']:.0f}ms, "
              f"{metrics['predicates_transferred']} transferred)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Speedup computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_speedups(
    cold: List[Dict[str, Any]],
    warm: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute per-model and aggregate speedup ratios."""
    per_model = []
    for c, w in zip(cold, warm):
        entry = {"name": c["name"]}
        for metric in ("iterations", "z3_queries", "time_ms"):
            c_val = c[metric]
            w_val = w[metric]
            if c_val > 0:
                ratio = c_val / max(w_val, 1e-9)
                reduction_pct = (1 - w_val / c_val) * 100
            else:
                ratio = 1.0
                reduction_pct = 0.0
            entry[f"{metric}_cold"] = c_val
            entry[f"{metric}_warm"] = w_val
            entry[f"{metric}_speedup"] = round(ratio, 2)
            entry[f"{metric}_reduction_pct"] = round(reduction_pct, 1)
        per_model.append(entry)

    # Aggregates
    agg = {}
    for metric in ("iterations", "z3_queries", "time_ms"):
        total_cold = sum(c[metric] for c in cold)
        total_warm = sum(w[metric] for w in warm)
        if total_cold > 0:
            agg[f"{metric}_total_cold"] = total_cold
            agg[f"{metric}_total_warm"] = total_warm
            agg[f"{metric}_speedup"] = round(total_cold / max(total_warm, 1e-9), 2)
            agg[f"{metric}_reduction_pct"] = round(
                (1 - total_warm / total_cold) * 100, 1
            )
        else:
            agg[f"{metric}_total_cold"] = 0
            agg[f"{metric}_total_warm"] = 0
            agg[f"{metric}_speedup"] = 1.0
            agg[f"{metric}_reduction_pct"] = 0.0

    return {"per_model": per_model, "aggregate": agg}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("Knowledge Transfer Benchmark: ResNet Family Cross-Session Speedup")
    print("=" * 70)

    names = [r[0] for r in RESNET_FAMILY]
    sources = [_make_resnet_source(*r) for r in RESNET_FAMILY]

    # Architectural hashes — should be same for same-skeleton models
    print("\nArchitectural hashes:")
    hashes = []
    for name, source in zip(names, sources):
        h = compute_arch_hash(source)
        hashes.append(h)
        print(f"  {name}: {h[:16]}...")
    unique_hashes = set(hashes)
    print(f"  Unique hashes: {len(unique_hashes)}")

    # Pass 1: Cold
    print("\n--- Pass 1: Cold start (no KB) ---")
    cold_results, kb = run_cold_pass(sources, names)

    # Save KB
    kb_path = os.path.join(os.path.dirname(__file__), ".benchmarks", "knowledge_transfer_kb.json")
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    kb.save(kb_path)
    print(f"\nKB saved: {kb.family_count} families, {kb.total_predicates} predicates")

    # Anti-unification
    print("\n--- Anti-unification: extracting shared proof schemas ---")
    for fam_hash, record in kb.families.items():
        certs = record.proof_certificates
        if len(certs) >= 2:
            schema = anti_unify_proof_certificates(certs, fam_hash)
            print(f"  Family {fam_hash[:16]}: "
                  f"{schema.source_count} certs → "
                  f"{len(schema.rule_skeleton)} skeleton steps, "
                  f"{sum(len(v) for v in schema.variable_positions)} variable positions")
        else:
            print(f"  Family {fam_hash[:16]}: {len(certs)} cert(s) — not enough for anti-unification")

    # Pass 2: Warm
    print("\n--- Pass 2: Warm start (with KB) ---")
    kb_loaded = VerificationKnowledgeBase.load(kb_path)
    warm_results = run_warm_pass(sources, names, kb_loaded)

    # Speedup
    speedups = compute_speedups(cold_results, warm_results)

    print("\n--- Speedup Summary ---")
    print(f"{'Model':<12} {'Iter Cold':>10} {'Iter Warm':>10} {'Iter Speedup':>13} "
          f"{'Time Cold':>10} {'Time Warm':>10} {'Time Speedup':>13}")
    print("-" * 82)
    for pm in speedups["per_model"]:
        print(f"{pm['name']:<12} "
              f"{pm['iterations_cold']:>10} {pm['iterations_warm']:>10} "
              f"{pm['iterations_speedup']:>10.2f}x "
              f"{pm['time_ms_cold']:>10.0f} {pm['time_ms_warm']:>10.0f} "
              f"{pm['time_ms_speedup']:>10.2f}x")

    agg = speedups["aggregate"]
    print("-" * 82)
    print(f"{'TOTAL':<12} "
          f"{agg['iterations_total_cold']:>10} {agg['iterations_total_warm']:>10} "
          f"{agg['iterations_speedup']:>10.2f}x "
          f"{agg['time_ms_total_cold']:>10.0f} {agg['time_ms_total_warm']:>10.0f} "
          f"{agg['time_ms_speedup']:>10.2f}x")

    # Honestly report findings
    iter_speedup = agg.get("iterations_speedup", 1.0)
    time_speedup = agg.get("time_ms_speedup", 1.0)
    print(f"\nIteration speedup: {iter_speedup:.2f}x "
          f"({'speedup observed' if iter_speedup > 1.01 else 'no significant speedup'})")
    print(f"Time speedup:      {time_speedup:.2f}x "
          f"({'speedup observed' if time_speedup > 1.01 else 'no significant speedup'})")

    # Save results
    output = {
        "benchmark": "knowledge_transfer",
        "family": "ResNet",
        "models": names,
        "cold_results": cold_results,
        "warm_results": warm_results,
        "speedups": speedups,
        "kb_stats": {
            "family_count": kb.family_count,
            "total_predicates": kb.total_predicates,
            "kb_path": kb_path,
        },
        "architectural_hashes": {n: h for n, h in zip(names, hashes)},
        "unique_hash_count": len(unique_hashes),
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", ".benchmarks", "knowledge_transfer_results.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    out_path2 = os.path.join(
        os.path.dirname(__file__), "results", "knowledge_transfer_results.json"
    )
    os.makedirs(os.path.dirname(out_path2), exist_ok=True)
    with open(out_path2, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results also saved to {out_path2}")


if __name__ == "__main__":
    main()
