"""
Cross-session knowledge transfer evaluation.

Demonstrates that the VerificationKnowledgeBase enables faster verification
of architecturally similar models by transferring predicates, strategies,
and failure modes across sessions.

Protocol
--------
1. Define 5 ResNet-like variants (same layer-type skeleton, different depths/widths).
2. **Pass 1 (cold)**: Verify all 5 from scratch, recording predicates and timing.
3. Save KB to disk.
4. **Pass 2 (warm)**: Verify the same 5 with KB loaded, measuring speedup.
5. Report: transferred predicates, CEGAR iterations saved, timing.

Results are saved to ``.benchmarks/cross_session_results.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.knowledge_base import VerificationKnowledgeBase, compute_arch_hash
from src.shape_cegar import ShapeCEGARLoop


# ── ResNet-like model variants ────────────────────────────────────────────

def _make_resnet_variant(
    name: str, num_blocks: int, hidden: int, num_classes: int = 10
) -> str:
    """Generate source code for a ResNet-like model variant.

    All variants share the same layer-type skeleton (Conv2d → BatchNorm2d →
    ReLU → Linear) but differ in depth (num_blocks), width (hidden), and
    number of output classes.
    """
    layers = []
    layers.append(f"        self.conv1 = nn.Conv2d(3, {hidden}, kernel_size=3, padding=1)")
    layers.append(f"        self.bn1 = nn.BatchNorm2d({hidden})")
    layers.append(f"        self.relu = nn.ReLU()")
    for i in range(num_blocks):
        layers.append(f"        self.conv{i+2} = nn.Conv2d({hidden}, {hidden}, kernel_size=3, padding=1)")
        layers.append(f"        self.bn{i+2} = nn.BatchNorm2d({hidden})")
    layers.append(f"        self.fc = nn.Linear({hidden}, {num_classes})")

    forward_lines = []
    forward_lines.append("        x = self.relu(self.bn1(self.conv1(x)))")
    for i in range(num_blocks):
        forward_lines.append(f"        x = self.relu(self.bn{i+2}(self.conv{i+2}(x)))")
    forward_lines.append("        x = x.mean(dim=[2, 3])")
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


VARIANTS = [
    ("ResNetTiny", 1, 16, 10),
    ("ResNetSmall", 2, 32, 10),
    ("ResNetMedium", 3, 64, 100),
    ("ResNetLarge", 4, 128, 100),
    ("ResNetWide", 2, 256, 1000),
]


def run_cold_pass(
    sources: List[str],
    names: List[str],
) -> tuple[List[Dict[str, Any]], VerificationKnowledgeBase]:
    """Verify all models from scratch (no KB), build KB."""
    kb = VerificationKnowledgeBase()
    results = []

    for name, source in zip(names, sources):
        arch_hash = compute_arch_hash(source)
        t0 = time.monotonic()
        loop = ShapeCEGARLoop(source, input_shapes={"x": ("batch", 3, 32, 32)})
        cegar_result = loop.run()
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Record discovered predicates to KB
        pred_strs = [p.pretty() for p in cegar_result.discovered_predicates]
        kb.record(
            arch_hash,
            predicates=pred_strs,
            strategies=[{
                "propagator_type": "shape_cegar",
                "iteration_count": cegar_result.iterations,
                "total_time_ms": elapsed_ms,
            }],
        )

        results.append({
            "name": name,
            "arch_hash": arch_hash,
            "verdict": cegar_result.verdict.name,
            "iterations": cegar_result.iterations,
            "predicates_discovered": len(cegar_result.discovered_predicates),
            "time_ms": round(elapsed_ms, 2),
            "pass": "cold",
        })
        print(f"  [cold] {name}: {cegar_result.verdict.name} "
              f"({cegar_result.iterations} iters, {elapsed_ms:.0f}ms)")

    return results, kb


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
        t0 = time.monotonic()
        loop = ShapeCEGARLoop(
            source,
            input_shapes={"x": ("batch", 3, 32, 32)},
            knowledge_base=kb,
        )
        cegar_result = loop.run()
        elapsed_ms = (time.monotonic() - t0) * 1000

        results.append({
            "name": name,
            "arch_hash": arch_hash,
            "verdict": cegar_result.verdict.name,
            "iterations": cegar_result.iterations,
            "predicates_discovered": len(cegar_result.discovered_predicates),
            "predicates_transferred": len(transferred.predicates),
            "time_ms": round(elapsed_ms, 2),
            "pass": "warm",
        })
        print(f"  [warm] {name}: {cegar_result.verdict.name} "
              f"({cegar_result.iterations} iters, {elapsed_ms:.0f}ms, "
              f"{len(transferred.predicates)} transferred)")

    return results


def main() -> None:
    print("=" * 60)
    print("Cross-Session Knowledge Transfer Evaluation")
    print("=" * 60)

    names = [v[0] for v in VARIANTS]
    sources = [_make_resnet_variant(*v) for v in VARIANTS]

    # Show architectural hashes
    print("\nArchitectural hashes:")
    for name, source in zip(names, sources):
        h = compute_arch_hash(source)
        print(f"  {name}: {h[:16]}...")

    # Pass 1: Cold
    print("\n--- Pass 1: Cold (no KB) ---")
    cold_results, kb = run_cold_pass(sources, names)

    # Save KB
    kb_path = os.path.join(os.path.dirname(__file__), ".benchmarks", "cross_session_kb.json")
    kb.save(kb_path)
    print(f"\nKB saved: {kb.family_count} families, {kb.total_predicates} predicates")

    # Pass 2: Warm
    print("\n--- Pass 2: Warm (with KB) ---")
    kb_loaded = VerificationKnowledgeBase.load(kb_path)
    warm_results = run_warm_pass(sources, names, kb_loaded)

    # Summary
    print("\n--- Summary ---")
    all_results = {"cold": cold_results, "warm": warm_results, "kb_path": kb_path}

    cold_iters = sum(r["iterations"] for r in cold_results)
    warm_iters = sum(r["iterations"] for r in warm_results)
    cold_time = sum(r["time_ms"] for r in cold_results)
    warm_time = sum(r["time_ms"] for r in warm_results)

    print(f"  Cold total iterations: {cold_iters}")
    print(f"  Warm total iterations: {warm_iters}")
    if cold_iters > 0:
        print(f"  Iteration reduction:   {cold_iters - warm_iters} "
              f"({(1 - warm_iters / cold_iters) * 100:.1f}%)")
    print(f"  Cold total time:       {cold_time:.0f}ms")
    print(f"  Warm total time:       {warm_time:.0f}ms")
    if cold_time > 0:
        print(f"  Time reduction:        {cold_time - warm_time:.0f}ms "
              f"({(1 - warm_time / cold_time) * 100:.1f}%)")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), ".benchmarks", "cross_session_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
