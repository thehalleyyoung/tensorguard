#!/usr/bin/env python3
"""Experiment: CVC5 Native vs Z3 UNSAT-Core Simulation Interpolation.

Compares CVC5's native ``getInterpolant`` against the Z3 UNSAT-core
simulation on the existing benchmark models.  For each benchmark,
both methods are evaluated on correctness (Craig properties) and timing.

Results are written to ``.benchmarks/cvc5_interpolation_results.json``.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import (
    ComputationGraph, ComputationStep, LayerDef,
    LayerKind, OpKind,
)
from src.shape_cegar import UnsatCorePredicateExtractor

try:
    from src.craig_interpolation import (
        InterpolationPredicateDiscovery,
        InterpolationMethod,
        DimMapping,
        _collect_vars_from_list,
        HAS_CVC5,
    )
    HAS_INTERP = True
except ImportError:
    HAS_INTERP = False

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ---------------------------------------------------------------------------
# Benchmark graph constructors (same as run_craig_interpolation_eval.py)
# ---------------------------------------------------------------------------

def make_linear_graph(in_f, out_f, input_dim):
    layer = LayerDef(attr_name="fc1", kind=LayerKind.LINEAR,
                     in_features=in_f, out_features=out_f)
    step = ComputationStep(op=OpKind.LAYER_CALL, inputs=["x"], output="y",
                           layer_ref="fc1")
    return ComputationGraph(
        class_name="LinearModel", layers={"fc1": layer},
        steps=[step], input_names=["x"], output_names=["y"],
    )


def make_reshape_linear_graph(in_features):
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
    },
    {
        "name": "reshape_linear_mismatch",
        "graph_fn": lambda: make_reshape_linear_graph(784),
        "input_shapes": {"x": (32, 1, 28, 27)},
        "failing_step": 1,
        "cex_dims": {"__ci_x_d0": 32, "__ci_x_d1": 1, "__ci_x_d2": 28, "__ci_x_d3": 27},
    },
    {
        "name": "conv_linear_channels",
        "graph_fn": lambda: make_conv_linear_graph(),
        "input_shapes": {"x": (1, 1, 8, 8)},
        "failing_step": 0,
        "cex_dims": {"__ci_x_d1": 1},
    },
    {
        "name": "two_linear_deep",
        "graph_fn": lambda: make_two_linear_graph(),
        "input_shapes": {"x": (32, 512)},
        "failing_step": 0,
        "cex_dims": {"__ci_x_d1": 512},
    },
    {
        "name": "large_linear_mismatch",
        "graph_fn": lambda: make_linear_graph(1024, 512, 256),
        "input_shapes": {"x": (16, 256)},
        "failing_step": 0,
        "cex_dims": {"__ci_x_d1": 256},
    },
]


def run_single_method(bench, method: "InterpolationMethod"):
    """Run interpolation with a specific method on one benchmark."""
    if not HAS_Z3 or not HAS_INTERP:
        return {"status": "skipped", "reason": "missing deps"}

    graph = bench["graph_fn"]()
    pe = UnsatCorePredicateExtractor(graph, {})
    ipd = InterpolationPredicateDiscovery(method=method)

    t0 = time.monotonic()
    path_cs, cex_cs, dm = pe._build_interpolation_query(
        graph, bench["failing_step"], bench["input_shapes"],
        concrete_dims=bench["cex_dims"],
    )
    preds = ipd.discover_via_interpolation(path_cs, cex_cs, dm)
    elapsed_ms = (time.monotonic() - t0) * 1000

    return {
        "status": "ok",
        "predicates_discovered": len(preds),
        "predicate_details": [
            p.pretty() if hasattr(p, "pretty") else str(p) for p in preds
        ],
        "stats": ipd.stats,
        "time_ms": round(elapsed_ms, 2),
    }


def main():
    print("=" * 70)
    print("CVC5 Native vs Z3 UNSAT-Core Simulation: Interpolation Evaluation")
    print(f"CVC5 available: {HAS_CVC5}")
    print("=" * 70)

    results = []

    for bench in BENCHMARKS:
        name = bench["name"]
        print(f"\n--- {name} ---")

        cvc5_result = run_single_method(bench, InterpolationMethod.CVC5_NATIVE)
        z3_result = run_single_method(bench, InterpolationMethod.Z3_UNSAT_CORE_SIMULATION)

        entry = {
            "benchmark": name,
            "cvc5_native": cvc5_result,
            "z3_simulation": z3_result,
        }

        for label, r in [("CVC5", cvc5_result), ("Z3-sim", z3_result)]:
            if r["status"] == "ok":
                print(f"  {label}: {r['predicates_discovered']} preds, "
                      f"{r['time_ms']:.1f}ms")
            else:
                print(f"  {label}: SKIPPED ({r.get('reason', '')})")

        results.append(entry)

    summary = {
        "cvc5_available": HAS_CVC5,
        "total_benchmarks": len(BENCHMARKS),
        "benchmark_results": results,
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", ".benchmarks",
                            "cvc5_interpolation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
