#!/usr/bin/env python3
"""Experiment: Craig Interpolation Predicate Discovery Evaluation.

Evaluates whether the Craig interpolation module now discovers predicates
during CEGAR refinement, complementing template-based discovery.

Measures:
- Number of predicates discovered via interpolation
- Whether interpolation closes the 4.3% gap between CEGAR and BMC
- Comparison of predicate quality (interpolation vs template)
"""
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import (
    ComputationGraph, ComputationStep, LayerDef,
    LayerKind, OpKind,
)
from src.shape_cegar import (
    ShapeCEGARLoop, ShapeCEGARResult, ShapePredicate,
    UnsatCorePredicateExtractor,
)

try:
    from src.craig_interpolation import InterpolationPredicateDiscovery, DimMapping
    HAS_INTERP = True
except ImportError:
    HAS_INTERP = False

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


def make_linear_graph(in_f, out_f, input_dim):
    """Create a graph: input -> Linear(in_f, out_f)."""
    layer = LayerDef(attr_name="fc1", kind=LayerKind.LINEAR,
                     in_features=in_f, out_features=out_f)
    step = ComputationStep(op=OpKind.LAYER_CALL, inputs=["x"], output="y",
                           layer_ref="fc1")
    return ComputationGraph(
        class_name="LinearModel", layers={"fc1": layer},
        steps=[step], input_names=["x"], output_names=["y"],
    )


def make_reshape_linear_graph(in_features):
    """Create: input -> reshape(B, in_features) -> Linear(in_features, 10)."""
    layer = LayerDef(attr_name="fc1", kind=LayerKind.LINEAR,
                     in_features=in_features, out_features=10)
    reshape = ComputationStep(op=OpKind.RESHAPE, inputs=["x"], output="x_flat",
                              params={"shape": [-1, in_features]})
    fc = ComputationStep(op=OpKind.LAYER_CALL, inputs=["x_flat"], output="y",
                         layer_ref="fc1")
    return ComputationGraph(
        class_name="ReshapeLinearModel", layers={"fc1": layer},
        steps=[reshape, fc], input_names=["x"], output_names=["y"],
    )


def make_conv_linear_graph():
    """Conv2d(3, 64, 3) -> Flatten -> Linear(64*6*6, 128)."""
    conv = LayerDef(attr_name="conv1", kind=LayerKind.CONV2D,
                    in_channels=3, out_channels=64, kernel_size=(3, 3))
    flat = LayerDef(attr_name="flat", kind=LayerKind.FLATTEN)
    fc = LayerDef(attr_name="fc1", kind=LayerKind.LINEAR,
                  in_features=2304, out_features=128)
    s1 = ComputationStep(op=OpKind.LAYER_CALL, inputs=["x"], output="c1",
                         layer_ref="conv1")
    s2 = ComputationStep(op=OpKind.FLATTEN, inputs=["c1"], output="f1",
                         params={"start_dim": 1})
    s3 = ComputationStep(op=OpKind.LAYER_CALL, inputs=["f1"], output="y",
                         layer_ref="fc1")
    return ComputationGraph(
        class_name="ConvLinearModel",
        layers={"conv1": conv, "flat": flat, "fc1": fc},
        steps=[s1, s2, s3], input_names=["x"], output_names=["y"],
    )


def make_two_linear_graph():
    """Linear(768, 256) -> Linear(256, 10) — tests multi-step transitions."""
    fc1 = LayerDef(attr_name="fc1", kind=LayerKind.LINEAR,
                   in_features=768, out_features=256)
    fc2 = LayerDef(attr_name="fc2", kind=LayerKind.LINEAR,
                   in_features=256, out_features=10)
    s1 = ComputationStep(op=OpKind.LAYER_CALL, inputs=["x"], output="h",
                         layer_ref="fc1")
    s2 = ComputationStep(op=OpKind.LAYER_CALL, inputs=["h"], output="y",
                         layer_ref="fc2")
    return ComputationGraph(
        class_name="TwoLinearModel",
        layers={"fc1": fc1, "fc2": fc2},
        steps=[s1, s2], input_names=["x"], output_names=["y"],
    )


BENCHMARKS = [
    {
        "name": "simple_linear_mismatch",
        "graph_fn": lambda: make_linear_graph(768, 256, 512),
        "input_shapes": {"x": (32, 512)},
        "failing_step": 0,
        "cex_dims": {"__ci_x_d1": 512},
        "expected_constraint": 768,
    },
    {
        "name": "reshape_linear_mismatch",
        "graph_fn": lambda: make_reshape_linear_graph(784),
        "input_shapes": {"x": (32, 1, 28, 27)},
        "failing_step": 1,
        "cex_dims": {"__ci_x_d0": 32, "__ci_x_d1": 1, "__ci_x_d2": 28, "__ci_x_d3": 27},
        "expected_constraint": None,
    },
    {
        "name": "conv_linear_channels",
        "graph_fn": lambda: make_conv_linear_graph(),
        "input_shapes": {"x": (1, 1, 8, 8)},
        "failing_step": 0,
        "cex_dims": {"__ci_x_d1": 1},
        "expected_constraint": 3,
    },
    {
        "name": "two_linear_deep",
        "graph_fn": lambda: make_two_linear_graph(),
        "input_shapes": {"x": (32, 512)},
        "failing_step": 0,
        "cex_dims": {"__ci_x_d1": 512},
        "expected_constraint": 768,
    },
    {
        "name": "large_linear_mismatch",
        "graph_fn": lambda: make_linear_graph(1024, 512, 256),
        "input_shapes": {"x": (16, 256)},
        "failing_step": 0,
        "cex_dims": {"__ci_x_d1": 256},
        "expected_constraint": 1024,
    },
]


def run_interpolation_benchmark(bench):
    """Run interpolation on a single benchmark."""
    if not HAS_Z3 or not HAS_INTERP:
        return {"name": bench["name"], "status": "skipped", "reason": "z3/interp not available"}

    graph = bench["graph_fn"]()
    pe = UnsatCorePredicateExtractor(graph, {})
    ipd = InterpolationPredicateDiscovery()

    t0 = time.monotonic()
    path_cs, cex_cs, dm = pe._build_interpolation_query(
        graph, bench["failing_step"], bench["input_shapes"],
        concrete_dims=bench["cex_dims"],
    )
    preds = ipd.discover_via_interpolation(path_cs, cex_cs, dm)
    elapsed_ms = (time.monotonic() - t0) * 1000

    result = {
        "name": bench["name"],
        "status": "ok",
        "path_constraints": len(path_cs),
        "cex_constraints": len(cex_cs),
        "predicates_discovered": len(preds),
        "predicate_details": [
            p.pretty() if hasattr(p, 'pretty') else str(p) for p in preds
        ],
        "interpolation_stats": ipd.stats,
        "time_ms": round(elapsed_ms, 2),
    }

    if bench["expected_constraint"] is not None:
        found = any(
            (hasattr(p, 'value') and abs(p.value) == bench["expected_constraint"]) or
            (hasattr(p, 'rhs') and abs(p.rhs) == bench["expected_constraint"])
            for p in preds
        )
        result["expected_found"] = found

    return result


def main():
    results = []
    total_preds = 0
    total_attempted = 0
    total_succeeded = 0

    print("=" * 70)
    print("Craig Interpolation Predicate Discovery Experiment")
    print("=" * 70)

    for bench in BENCHMARKS:
        r = run_interpolation_benchmark(bench)
        results.append(r)
        if r["status"] == "ok":
            total_preds += r["predicates_discovered"]
            total_attempted += r["interpolation_stats"]["interpolations_attempted"]
            total_succeeded += r["interpolation_stats"]["interpolations_succeeded"]
            print(f"\n  {r['name']}:")
            print(f"    Predicates discovered: {r['predicates_discovered']}")
            print(f"    Details: {r['predicate_details']}")
            print(f"    Time: {r['time_ms']:.1f}ms")
            if "expected_found" in r:
                status = "✓" if r["expected_found"] else "✗"
                print(f"    Expected constraint: {status}")
        else:
            print(f"\n  {r['name']}: SKIPPED ({r.get('reason', '')})")

    summary = {
        "total_benchmarks": len(BENCHMARKS),
        "total_predicates_discovered": total_preds,
        "interpolations_attempted": total_attempted,
        "interpolations_succeeded": total_succeeded,
        "success_rate": total_succeeded / max(total_attempted, 1),
        "benchmark_results": results,
    }

    print(f"\n{'=' * 70}")
    print(f"Summary: {total_preds} predicates from {total_succeeded}/{total_attempted} interpolations")
    print(f"{'=' * 70}")

    out_path = os.path.join(os.path.dirname(__file__), "..", ".benchmarks",
                            "craig_interpolation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
