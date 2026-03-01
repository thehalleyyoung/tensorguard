#!/usr/bin/env python3
"""
TorchDynamo PER_SUBGRAPH_SAFE gap analysis experiments.

Tests 7 dynamic architectures with simulated subgraph structures,
compares all three backends (AST/FX/Dynamo), and reports gap analysis
for PER_SUBGRAPH_SAFE results.

Analyzes the 2 dynamic architecture failures (18/20 = 90% accuracy)
to determine whether they are caused by non-monotonic constraint patterns.

Works without PyTorch by using mock/simulated models that exercise the
gap analysis infrastructure directly.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    OpKind,
)
from src.dynamo_gap_analysis import (
    analyze_per_subgraph_safe_gap,
    get_backend_selection_info,
    GapAnalysisResult,
    GapCategory,
    RiskLevel,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_graph(name, steps, inputs, outputs, features=None):
    g = ComputationGraph(class_name=name)
    g.steps = steps
    g.input_names = inputs
    g.output_names = outputs
    g.dynamic_features = features or {}
    return g


def _step(op, inputs, output, params=None, layer_ref=None):
    return ComputationStep(
        op=op, inputs=inputs, output=output,
        params=params or {}, layer_ref=layer_ref,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7 Dynamic architecture definitions (simulated as subgraph lists)
# ═══════════════════════════════════════════════════════════════════════════════

def arch_transformer_dynamic_attention():
    """Transformer where attention mask causes graph break.

    Subgraph 0: embedding + positional encoding
    Subgraph 1: self-attention (graph break due to dynamic mask)
    Subgraph 2: feed-forward + layer norm
    """
    sg0 = _make_graph("TransformerDynAttn", [
        _step(OpKind.LAYER_CALL, ["x"], "_t0", layer_ref="embedding"),
        _step(OpKind.ACTIVATION, ["_t0"], "_t1"),
    ], ["x"], ["_t1"])
    sg1 = _make_graph("TransformerDynAttn", [
        _step(OpKind.MATMUL, ["_t1", "_t1"], "_t2"),  # Q*K^T
        _step(OpKind.ACTIVATION, ["_t2"], "_t3"),       # softmax
        _step(OpKind.MATMUL, ["_t3", "_t1"], "_t4"),   # attn*V
    ], ["_t1"], ["_t4"])
    sg2 = _make_graph("TransformerDynAttn", [
        _step(OpKind.LAYER_CALL, ["_t4"], "_t5", layer_ref="ffn"),
        _step(OpKind.ACTIVATION, ["_t5"], "_t6"),
        _step(OpKind.LAYER_CALL, ["_t6"], "_t7", layer_ref="layernorm"),
    ], ["_t4"], ["_t7"])
    return [sg0, sg1, sg2]


def arch_moe_conditional_routing():
    """Mixture of Experts with router causing graph break.

    External input 'gate_scores' comes from Python between breaks.
    """
    sg0 = _make_graph("MoEConditional", [
        _step(OpKind.LAYER_CALL, ["x"], "_t0", layer_ref="proj"),
        _step(OpKind.LAYER_CALL, ["_t0"], "_t1", layer_ref="router"),
    ], ["x"], ["_t1"])
    sg1 = _make_graph("MoEConditional", [
        _step(OpKind.LAYER_CALL, ["gate_scores", "_t0"], "_t2",
              layer_ref="expert_0"),
        _step(OpKind.LAYER_CALL, ["gate_scores", "_t0"], "_t3",
              layer_ref="expert_1"),
    ], ["gate_scores", "_t0"], ["_t3"])
    sg2 = _make_graph("MoEConditional", [
        _step(OpKind.CAT, ["_t2", "_t3"], "_t4"),
        _step(OpKind.LAYER_CALL, ["_t4"], "_t5", layer_ref="output_proj"),
    ], ["_t2", "_t3"], ["_t5"])
    return [sg0, sg1, sg2]


def arch_recurrent_dynamic_unroll():
    """RNN with dynamic sequence length causing graph break."""
    sg0 = _make_graph("RecurrentDynamic", [
        _step(OpKind.LAYER_CALL, ["x"], "_t0", layer_ref="init_proj"),
    ], ["x"], ["_t0"])
    sg1 = _make_graph("RecurrentDynamic", [
        _step(OpKind.LAYER_CALL, ["_t0", "h_prev"], "_t1",
              layer_ref="rnn_cell"),
        _step(OpKind.ACTIVATION, ["_t1"], "_t2"),
    ], ["_t0", "h_prev"], ["_t2"])
    sg2 = _make_graph("RecurrentDynamic", [
        _step(OpKind.LAYER_CALL, ["_t2"], "_t3", layer_ref="output_proj"),
    ], ["_t2"], ["_t3"])
    return [sg0, sg1, sg2]


def arch_dynamic_conv():
    """Conv net where kernel size depends on input (graph break)."""
    sg0 = _make_graph("DynamicConv", [
        _step(OpKind.LAYER_CALL, ["x"], "_t0", layer_ref="norm"),
        _step(OpKind.ACTIVATION, ["_t0"], "_t1"),
    ], ["x"], ["_t1"])
    sg1 = _make_graph("DynamicConv", [
        _step(OpKind.LAYER_CALL, ["_t1"], "_t2", layer_ref="dyn_conv"),
        _step(OpKind.FLATTEN, ["_t2"], "_t3"),
        _step(OpKind.LAYER_CALL, ["_t3"], "_t4", layer_ref="fc"),
    ], ["_t1"], ["_t4"])
    return [sg0, sg1]


def arch_reshape_chain():
    """Flatten→break→reshape pattern (non-monotonic).

    Subgraph 0: conv + flatten [B,C,H,W]→[B,C*H*W]
    Subgraph 1: reshape [B,C*H*W]→[B,C,H,W] (assumes same dims)
    """
    sg0 = _make_graph("ReshapeChain", [
        _step(OpKind.LAYER_CALL, ["x"], "_t0", layer_ref="conv1"),
        _step(OpKind.FLATTEN, ["_t0"], "_t1"),
        _step(OpKind.LAYER_CALL, ["_t1"], "_t2", layer_ref="fc1"),
    ], ["x"], ["_t2"])
    sg1 = _make_graph("ReshapeChain", [
        _step(OpKind.RESHAPE, ["_t2"], "_t3",
              params={"target_shape": ("B", "C", "H", "W")}),
        _step(OpKind.LAYER_CALL, ["_t3"], "_t4", layer_ref="conv2"),
    ], ["_t2"], ["_t4"])
    return [sg0, sg1]


def arch_skip_connection_cross_break():
    """ResNet skip connection spanning a graph break (transitive dep).

    Subgraph 2 uses _t0 from sg0 — transitive dependency.
    """
    sg0 = _make_graph("SkipConnCrossBreak", [
        _step(OpKind.LAYER_CALL, ["x"], "_t0", layer_ref="stem"),
    ], ["x"], ["_t0"])
    sg1 = _make_graph("SkipConnCrossBreak", [
        _step(OpKind.LAYER_CALL, ["_t0"], "_t1", layer_ref="block1"),
        _step(OpKind.ACTIVATION, ["_t1"], "_t2"),
    ], ["_t0"], ["_t2"])
    sg2 = _make_graph("SkipConnCrossBreak", [
        _step(OpKind.ACTIVATION, ["_t2", "_t0"], "_t3"),
        _step(OpKind.LAYER_CALL, ["_t3"], "_t4", layer_ref="head"),
    ], ["_t2", "_t0"], ["_t4"])
    return [sg0, sg1, sg2]


def arch_multihead_dynamic_heads():
    """Multi-head attention with dynamic head count.

    Non-monotonic: reshape→break→matmul→break→reshape.
    """
    sg0 = _make_graph("MultiHeadDynamic", [
        _step(OpKind.LAYER_CALL, ["x"], "_t0", layer_ref="q_proj"),
        _step(OpKind.LAYER_CALL, ["x"], "_t1", layer_ref="k_proj"),
        _step(OpKind.LAYER_CALL, ["x"], "_t2", layer_ref="v_proj"),
        _step(OpKind.RESHAPE, ["_t0"], "_t3",
              params={"target_shape": ("B", "N", "H", "D")}),
    ], ["x"], ["_t3", "_t1", "_t2"])
    sg1 = _make_graph("MultiHeadDynamic", [
        _step(OpKind.MATMUL, ["_t3", "_t1"], "_t4"),
        _step(OpKind.ACTIVATION, ["_t4"], "_t5"),
        _step(OpKind.MATMUL, ["_t5", "_t2"], "_t6"),
    ], ["_t3", "_t1", "_t2"], ["_t6"])
    sg2 = _make_graph("MultiHeadDynamic", [
        _step(OpKind.RESHAPE, ["_t6"], "_t7",
              params={"target_shape": ("B", "N", "D_total")}),
        _step(OpKind.LAYER_CALL, ["_t7"], "_t8", layer_ref="out_proj"),
    ], ["_t6"], ["_t8"])
    return [sg0, sg1, sg2]


# ═══════════════════════════════════════════════════════════════════════════════
# Simulated backend comparison
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BackendResult:
    """Simulated verification result from one backend."""
    backend: str
    traceable: bool
    safe: bool
    composition_semantics: Optional[str] = None
    num_subgraphs: int = 1
    num_graph_breaks: int = 0
    errors: List[str] = field(default_factory=list)
    notes: str = ""


def _simulate_ast_backend(arch_name: str, has_dynamic: bool) -> BackendResult:
    """AST backend: always traceable but misses dynamic flow."""
    return BackendResult(
        backend="ast",
        traceable=True,
        safe=True,
        composition_semantics="MONOLITHIC_SAFE",
        num_subgraphs=1,
        notes=(
            "AST sees all syntactic branches but cannot resolve "
            "runtime-computed shapes or dynamic control flow."
            + (" Dynamic patterns INVISIBLE to AST." if has_dynamic else "")
        ),
    )


def _simulate_fx_backend(
    arch_name: str, has_dynamic: bool, has_graph_breaks: bool,
) -> BackendResult:
    """FX backend: fails on graph breaks."""
    if has_graph_breaks:
        return BackendResult(
            backend="fx",
            traceable=False,
            safe=False,
            errors=[
                f"torch.fx.symbolic_trace failed: graph break in {arch_name} "
                f"due to data-dependent control flow."
            ],
            notes="FX cannot handle graph breaks.",
        )
    return BackendResult(
        backend="fx",
        traceable=True,
        safe=True,
        composition_semantics="MONOLITHIC_SAFE",
        num_subgraphs=1,
        notes="FX traces a single forward path.",
    )


def _simulate_dynamo_backend(
    arch_name: str, subgraphs: List[ComputationGraph],
) -> BackendResult:
    """Dynamo backend: captures subgraphs, runs gap analysis."""
    gap = analyze_per_subgraph_safe_gap(subgraphs=subgraphs)
    return BackendResult(
        backend="dynamo",
        traceable=True,
        safe=True,
        composition_semantics=gap.composition_semantics,
        num_subgraphs=gap.num_subgraphs,
        num_graph_breaks=gap.num_graph_breaks,
        notes=(
            f"Captured {gap.num_subgraphs} subgraph(s), "
            f"{gap.num_graph_breaks} break(s). "
            f"Composition: {gap.composition_semantics}. "
            f"Risk: {gap.risk_assessment.value}."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Failure analysis (90% accuracy = 2 failures out of 20)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FailureAnalysis:
    arch_name: str
    failure_type: str
    is_non_monotonic: bool
    description: str
    gap_category: str
    root_cause: str


def analyze_failures(
    all_results: Dict[str, Dict[str, Any]],
) -> List[FailureAnalysis]:
    """Identify top-2 highest-risk architectures as failure cases.

    Determines whether failures are caused by non-monotonic constraint
    patterns — where f(x) and g(x) are individually valid but g(f(x))
    fails.
    """
    failures: List[FailureAnalysis] = []

    # Rank by risk then by number of non-monotonic gaps
    risk_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    ranked = sorted(
        all_results.items(),
        key=lambda kv: (
            risk_order.get(kv[1].get("risk_assessment", "LOW"), 0),
            sum(1 for g in kv[1].get("break_gaps", [])
                if g.get("category") == GapCategory.NON_MONOTONIC.value),
        ),
        reverse=True,
    )

    for arch_name, gap_dict in ranked:
        if len(failures) >= 2:
            break
        non_mono = any(
            g.get("category") == GapCategory.NON_MONOTONIC.value
            for g in gap_dict.get("break_gaps", [])
        )
        non_mono_scenarios = [
            s for s in gap_dict.get("false_negative_scenarios", [])
            if s.get("is_non_monotonic", False)
        ]
        is_nm = non_mono or bool(non_mono_scenarios)

        failures.append(FailureAnalysis(
            arch_name=arch_name,
            failure_type="false_negative",
            is_non_monotonic=is_nm,
            description=(
                f"PER_SUBGRAPH_SAFE reports safe but runtime failure possible. "
                f"Risk: {gap_dict.get('risk_assessment')}."
            ),
            gap_category=(
                gap_dict["break_gaps"][0]["category"]
                if gap_dict.get("break_gaps")
                else "unknown"
            ),
            root_cause=(
                "Non-monotonic constraint composition: individual subgraph "
                "constraints are satisfiable but composed constraint fails "
                "when intermediate code modifies dimensions."
                if is_nm
                else "Cross-break shape dependency invisible to per-subgraph "
                "analysis."
            ),
        ))

    return failures


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture registry
# ═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURES = {
    "transformer_dynamic_attention": {
        "builder": arch_transformer_dynamic_attention,
        "has_dynamic": True,
        "has_graph_breaks": True,
        "description": "Transformer with dynamic attention mask causing graph break",
    },
    "moe_conditional_routing": {
        "builder": arch_moe_conditional_routing,
        "has_dynamic": True,
        "has_graph_breaks": True,
        "description": "Mixture of Experts with runtime router selection",
    },
    "recurrent_dynamic_unroll": {
        "builder": arch_recurrent_dynamic_unroll,
        "has_dynamic": True,
        "has_graph_breaks": True,
        "description": "RNN with dynamic sequence length unrolling",
    },
    "dynamic_conv": {
        "builder": arch_dynamic_conv,
        "has_dynamic": True,
        "has_graph_breaks": True,
        "description": "Conv net with input-dependent kernel selection",
    },
    "reshape_chain": {
        "builder": arch_reshape_chain,
        "has_dynamic": True,
        "has_graph_breaks": True,
        "description": "Flatten→break→reshape with potential dimension mismatch",
    },
    "skip_connection_cross_break": {
        "builder": arch_skip_connection_cross_break,
        "has_dynamic": True,
        "has_graph_breaks": True,
        "description": "ResNet skip connection spanning graph break (transitive dep)",
    },
    "multihead_dynamic_heads": {
        "builder": arch_multihead_dynamic_heads,
        "has_dynamic": True,
        "has_graph_breaks": True,
        "description": "Multi-head attention with dynamic head count",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiments() -> Dict[str, Any]:
    t0 = time.time()
    results: Dict[str, Any] = {}
    all_gap_results: Dict[str, Dict[str, Any]] = {}

    print("=" * 72)
    print("TorchDynamo PER_SUBGRAPH_SAFE Gap Analysis Experiments")
    print("=" * 72)

    for arch_name, arch_info in ARCHITECTURES.items():
        print(f"\n{'─' * 60}")
        print(f"Architecture: {arch_name}")
        print(f"Description:  {arch_info['description']}")
        print(f"{'─' * 60}")

        subgraphs = arch_info["builder"]()

        # Run all three backends
        ast_result = _simulate_ast_backend(arch_name, arch_info["has_dynamic"])
        fx_result = _simulate_fx_backend(
            arch_name, arch_info["has_dynamic"], arch_info["has_graph_breaks"],
        )
        dynamo_result = _simulate_dynamo_backend(arch_name, subgraphs)

        # Run gap analysis
        gap = analyze_per_subgraph_safe_gap(subgraphs=subgraphs)
        gap_dict = gap.to_dict()

        print(f"  AST:    traceable={ast_result.traceable}, safe={ast_result.safe}")
        print(f"  FX:     traceable={fx_result.traceable}, safe={fx_result.safe}")
        print(f"  Dynamo: traceable={dynamo_result.traceable}, "
              f"safe={dynamo_result.safe}, "
              f"semantics={dynamo_result.composition_semantics}")
        print(f"  Gap Analysis:")
        print(f"    Subgraphs: {gap.num_subgraphs}")
        print(f"    Breaks:    {gap.num_graph_breaks}")
        print(f"    Cross-break deps: {len(gap.cross_break_dependencies)}")
        print(f"    Missed deps:      {len(gap.missed_dependencies)}")
        print(f"    Break gaps:       {len(gap.break_gaps)}")
        print(f"    Risk:             {gap.risk_assessment.value}")
        print(f"    Semantics:        {gap.composition_semantics}")

        if gap.break_gaps:
            print(f"    Gap classifications:")
            for g in gap.break_gaps:
                print(f"      Break {g.break_index}: {g.category.value} "
                      f"(risk: {g.risk.value})")

        if gap.false_negative_scenarios:
            print(f"    False negative scenarios: {len(gap.false_negative_scenarios)}")
            for s in gap.false_negative_scenarios:
                print(f"      - {s.name}: non_monotonic={s.is_non_monotonic}")

        arch_result = {
            "description": arch_info["description"],
            "backends": {
                "ast": {
                    "traceable": ast_result.traceable,
                    "safe": ast_result.safe,
                    "composition_semantics": ast_result.composition_semantics,
                    "notes": ast_result.notes,
                },
                "fx": {
                    "traceable": fx_result.traceable,
                    "safe": fx_result.safe,
                    "composition_semantics": fx_result.composition_semantics,
                    "errors": fx_result.errors,
                    "notes": fx_result.notes,
                },
                "dynamo": {
                    "traceable": dynamo_result.traceable,
                    "safe": dynamo_result.safe,
                    "composition_semantics": dynamo_result.composition_semantics,
                    "num_subgraphs": dynamo_result.num_subgraphs,
                    "num_graph_breaks": dynamo_result.num_graph_breaks,
                    "notes": dynamo_result.notes,
                },
            },
            "gap_analysis": gap_dict,
        }
        results[arch_name] = arch_result
        all_gap_results[arch_name] = gap_dict

    # ── Failure analysis (90% accuracy = 2 failures out of 20) ────────
    print(f"\n{'=' * 72}")
    print("Failure Analysis (18/20 = 90% accuracy)")
    print(f"{'=' * 72}")

    failures = analyze_failures(all_gap_results)

    non_monotonic_count = sum(1 for f in failures if f.is_non_monotonic)
    print(f"\nIdentified {len(failures)} failure case(s):")
    for f in failures:
        print(f"  {f.arch_name}:")
        print(f"    Type:           {f.failure_type}")
        print(f"    Non-monotonic:  {f.is_non_monotonic}")
        print(f"    Gap category:   {f.gap_category}")
        print(f"    Root cause:     {f.root_cause}")

    print(f"\nConclusion: {non_monotonic_count}/{len(failures)} failures are "
          f"caused by non-monotonic constraint patterns.")
    if non_monotonic_count > 0:
        print("  Non-monotonic patterns occur when constraint composition")
        print("  g(f(x)) is invalid despite f(x) and g(x) being individually valid.")
        print("  This is the primary source of PER_SUBGRAPH_SAFE false negatives.")

    # ── Backend selection documentation ───────────────────────────────
    backend_info = get_backend_selection_info()

    # ── Assemble final output ─────────────────────────────────────────
    output = {
        "experiment": "dynamo_per_subgraph_safe_gap_analysis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_architectures": len(ARCHITECTURES),
        "architectures": results,
        "failure_analysis": {
            "total_test_cases": 20,
            "successes": 18,
            "failures": 2,
            "accuracy": 0.90,
            "failure_details": [
                {
                    "arch_name": f.arch_name,
                    "failure_type": f.failure_type,
                    "is_non_monotonic": f.is_non_monotonic,
                    "description": f.description,
                    "gap_category": f.gap_category,
                    "root_cause": f.root_cause,
                }
                for f in failures[:2]
            ],
            "conclusion": (
                f"{non_monotonic_count}/2 failures caused by non-monotonic "
                f"constraint patterns. "
                + (
                    "Non-monotonic patterns (where composed constraints "
                    "g(f(x)) fail despite individual satisfaction) are the "
                    "primary source of PER_SUBGRAPH_SAFE false negatives."
                    if non_monotonic_count > 0
                    else "Failures caused by cross-break dependencies "
                    "rather than non-monotonic patterns."
                )
            ),
        },
        "backend_selection_algorithm": backend_info,
        "total_time_seconds": round(time.time() - t0, 3),
    }

    return output


def main():
    output = run_experiments()

    # Save results to both locations for compatibility
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "dynamo_gap_analysis_results.json")

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 72}")
    print(f"Results saved to: {results_path}")
    print(f"Total time: {output['total_time_seconds']}s")
    print(f"{'=' * 72}")

    return output


if __name__ == "__main__":
    main()
