"""
Thread-modular verification evaluation experiment.

Tests thread-modular composition verification on 10 model architectures:
  - 5 models with no graph breaks (MONOLITHIC_SAFE)
  - 3 models with graph breaks but safe composition (COMPOSITION_VERIFIED)
  - 2 models with graph breaks and cross-break issues (GAP_DETECTED)

Uses synthetic ComputationGraph instances directly since torch._dynamo
may not be available.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the implementation src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    LayerDef,
    LayerKind,
    OpKind,
)
from src.thread_modular import (
    CompositionVerdict,
    ThreadModularVerifier,
    verify_thread_modular,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_graph(class_name, steps=None, inputs=None, outputs=None,
                layers=None, features=None):
    g = ComputationGraph(class_name=class_name)
    g.steps = steps or []
    g.input_names = inputs or []
    g.output_names = outputs or []
    if layers:
        g.layers = layers
    g.dynamic_features = features or {}
    return g


def _step(op, inputs, output, params=None, layer_ref=None):
    return ComputationStep(
        op=op, inputs=inputs, output=output,
        params=params or {}, layer_ref=layer_ref,
    )


def _linear(name, in_f, out_f):
    return LayerDef(
        attr_name=name, kind=LayerKind.LINEAR,
        in_features=in_f, out_features=out_f,
    )


def _conv2d(name, in_c, out_c):
    return LayerDef(
        attr_name=name, kind=LayerKind.CONV2D,
        in_channels=in_c, out_channels=out_c,
        kernel_size=(3, 3),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Model definitions: 5 monolithic (no breaks)
# ═══════════════════════════════════════════════════════════════════════════════

def model_simple_mlp() -> Tuple[str, List[ComputationGraph], Dict[str, Tuple]]:
    """Simple 3-layer MLP — no graph breaks."""
    layers = {
        "fc1": _linear("fc1", 784, 256),
        "fc2": _linear("fc2", 256, 128),
        "fc3": _linear("fc3", 128, 10),
    }
    sg = _make_graph("SimpleMLP", steps=[
        _step(OpKind.LAYER_CALL, ["x"], "h1", layer_ref="fc1"),
        _step(OpKind.ACTIVATION, ["h1"], "h1_relu"),
        _step(OpKind.LAYER_CALL, ["h1_relu"], "h2", layer_ref="fc2"),
        _step(OpKind.ACTIVATION, ["h2"], "h2_relu"),
        _step(OpKind.LAYER_CALL, ["h2_relu"], "out", layer_ref="fc3"),
    ], inputs=["x"], outputs=["out"], layers=layers)
    return "SimpleMLP", [sg], {"x": ("batch", 784)}


def model_conv_classifier() -> Tuple[str, List[ComputationGraph], Dict[str, Tuple]]:
    """CNN classifier — no graph breaks."""
    layers = {
        "conv1": _conv2d("conv1", 3, 16),
        "conv2": _conv2d("conv2", 16, 32),
        "fc": _linear("fc", 32, 10),
    }
    sg = _make_graph("ConvClassifier", steps=[
        _step(OpKind.LAYER_CALL, ["x"], "c1", layer_ref="conv1"),
        _step(OpKind.ACTIVATION, ["c1"], "c1_relu"),
        _step(OpKind.LAYER_CALL, ["c1_relu"], "c2", layer_ref="conv2"),
        _step(OpKind.ACTIVATION, ["c2"], "c2_relu"),
        _step(OpKind.FLATTEN, ["c2_relu"], "flat"),
        _step(OpKind.LAYER_CALL, ["flat"], "out", layer_ref="fc"),
    ], inputs=["x"], outputs=["out"], layers=layers)
    return "ConvClassifier", [sg], {"x": ("batch", 3, 32, 32)}


def model_residual_block() -> Tuple[str, List[ComputationGraph], Dict[str, Tuple]]:
    """Residual block with skip connection — no graph breaks."""
    layers = {
        "fc1": _linear("fc1", 64, 64),
        "fc2": _linear("fc2", 64, 64),
    }
    sg = _make_graph("ResidualBlock", steps=[
        _step(OpKind.LAYER_CALL, ["x"], "h1", layer_ref="fc1"),
        _step(OpKind.ACTIVATION, ["h1"], "h1_relu"),
        _step(OpKind.LAYER_CALL, ["h1_relu"], "h2", layer_ref="fc2"),
        _step(OpKind.ADD, ["h2", "x"], "out"),
    ], inputs=["x"], outputs=["out"], layers=layers)
    return "ResidualBlock", [sg], {"x": ("batch", 64)}


def model_attention_head() -> Tuple[str, List[ComputationGraph], Dict[str, Tuple]]:
    """Single attention head — no graph breaks."""
    layers = {
        "q_proj": _linear("q_proj", 64, 64),
        "k_proj": _linear("k_proj", 64, 64),
        "v_proj": _linear("v_proj", 64, 64),
        "out_proj": _linear("out_proj", 64, 64),
    }
    sg = _make_graph("AttentionHead", steps=[
        _step(OpKind.LAYER_CALL, ["x"], "q", layer_ref="q_proj"),
        _step(OpKind.LAYER_CALL, ["x"], "k", layer_ref="k_proj"),
        _step(OpKind.LAYER_CALL, ["x"], "v", layer_ref="v_proj"),
        _step(OpKind.MATMUL, ["q", "k"], "attn_scores"),
        _step(OpKind.SOFTMAX, ["attn_scores"], "attn_weights"),
        _step(OpKind.MATMUL, ["attn_weights", "v"], "attn_out"),
        _step(OpKind.LAYER_CALL, ["attn_out"], "out", layer_ref="out_proj"),
    ], inputs=["x"], outputs=["out"], layers=layers)
    return "AttentionHead", [sg], {"x": ("batch", "seq_len", 64)}


def model_autoencoder() -> Tuple[str, List[ComputationGraph], Dict[str, Tuple]]:
    """Autoencoder — no graph breaks."""
    layers = {
        "enc1": _linear("enc1", 784, 256),
        "enc2": _linear("enc2", 256, 64),
        "dec1": _linear("dec1", 64, 256),
        "dec2": _linear("dec2", 256, 784),
    }
    sg = _make_graph("Autoencoder", steps=[
        _step(OpKind.LAYER_CALL, ["x"], "e1", layer_ref="enc1"),
        _step(OpKind.ACTIVATION, ["e1"], "e1r"),
        _step(OpKind.LAYER_CALL, ["e1r"], "latent", layer_ref="enc2"),
        _step(OpKind.ACTIVATION, ["latent"], "lr"),
        _step(OpKind.LAYER_CALL, ["lr"], "d1", layer_ref="dec1"),
        _step(OpKind.ACTIVATION, ["d1"], "d1r"),
        _step(OpKind.LAYER_CALL, ["d1r"], "out", layer_ref="dec2"),
    ], inputs=["x"], outputs=["out"], layers=layers)
    return "Autoencoder", [sg], {"x": ("batch", 784)}


# ═══════════════════════════════════════════════════════════════════════════════
# Model definitions: 3 with graph breaks but safe composition
# ═══════════════════════════════════════════════════════════════════════════════

def model_two_stage_mlp() -> Tuple[str, List[ComputationGraph], Dict[str, Tuple]]:
    """Two-stage MLP split at graph break — safe composition."""
    sg1 = _make_graph("TwoStageMLP_G0", steps=[
        _step(OpKind.LAYER_CALL, ["x"], "h1", layer_ref="fc1"),
        _step(OpKind.ACTIVATION, ["h1"], "h1_relu"),
    ], inputs=["x"], outputs=["h1_relu"],
       layers={"fc1": _linear("fc1", 64, 128)})

    sg2 = _make_graph("TwoStageMLP_G1", steps=[
        _step(OpKind.LAYER_CALL, ["h1_relu"], "out", layer_ref="fc2"),
    ], inputs=["h1_relu"], outputs=["out"],
       layers={"fc2": _linear("fc2", 128, 10)})

    return "TwoStageMLP", [sg1, sg2], {"x": ("batch", 64)}


def model_encoder_decoder_split() -> Tuple[str, List[ComputationGraph], Dict[str, Tuple]]:
    """Encoder-decoder split at graph break — safe composition."""
    sg1 = _make_graph("EncDecSplit_Enc", steps=[
        _step(OpKind.LAYER_CALL, ["x"], "e1", layer_ref="enc1"),
        _step(OpKind.ACTIVATION, ["e1"], "e1r"),
        _step(OpKind.LAYER_CALL, ["e1r"], "latent", layer_ref="enc2"),
    ], inputs=["x"], outputs=["latent"],
       layers={"enc1": _linear("enc1", 256, 128),
               "enc2": _linear("enc2", 128, 32)})

    sg2 = _make_graph("EncDecSplit_Dec", steps=[
        _step(OpKind.LAYER_CALL, ["latent"], "d1", layer_ref="dec1"),
        _step(OpKind.ACTIVATION, ["d1"], "d1r"),
        _step(OpKind.LAYER_CALL, ["d1r"], "out", layer_ref="dec2"),
    ], inputs=["latent"], outputs=["out"],
       layers={"dec1": _linear("dec1", 32, 128),
               "dec2": _linear("dec2", 128, 256)})

    return "EncoderDecoderSplit", [sg1, sg2], {"x": ("batch", 256)}


def model_three_stage_pipeline() -> Tuple[str, List[ComputationGraph], Dict[str, Tuple]]:
    """Three-stage pipeline with clean breaks — safe composition."""
    sg1 = _make_graph("Pipeline_G0", steps=[
        _step(OpKind.LAYER_CALL, ["x"], "h1", layer_ref="fc1"),
        _step(OpKind.ACTIVATION, ["h1"], "h1r"),
    ], inputs=["x"], outputs=["h1r"],
       layers={"fc1": _linear("fc1", 64, 64)})

    sg2 = _make_graph("Pipeline_G1", steps=[
        _step(OpKind.LAYER_CALL, ["h1r"], "h2", layer_ref="fc2"),
        _step(OpKind.ACTIVATION, ["h2"], "h2r"),
    ], inputs=["h1r"], outputs=["h2r"],
       layers={"fc2": _linear("fc2", 64, 64)})

    sg3 = _make_graph("Pipeline_G2", steps=[
        _step(OpKind.LAYER_CALL, ["h2r"], "out", layer_ref="fc3"),
    ], inputs=["h2r"], outputs=["out"],
       layers={"fc3": _linear("fc3", 64, 10)})

    return "ThreeStagePipeline", [sg1, sg2, sg3], {"x": ("batch", 64)}


# ═══════════════════════════════════════════════════════════════════════════════
# Model definitions: 2 with graph breaks and cross-break issues
# ═══════════════════════════════════════════════════════════════════════════════

def model_reshape_across_break() -> Tuple[str, List[ComputationGraph], Dict[str, Tuple]]:
    """Reshape across graph break — non-monotonic gap."""
    sg1 = _make_graph("ReshapeBreak_G0", steps=[
        _step(OpKind.LAYER_CALL, ["x"], "h", layer_ref="fc1"),
        _step(OpKind.RESHAPE, ["h"], "h_reshaped",
              params={"target_shape": ("batch", 4, 16)}),
    ], inputs=["x"], outputs=["h_reshaped"],
       layers={"fc1": _linear("fc1", 64, 64)})

    sg2 = _make_graph("ReshapeBreak_G1", steps=[
        _step(OpKind.RESHAPE, ["h_reshaped"], "h_flat",
              params={"target_shape": ("batch", 64)}),
        _step(OpKind.LAYER_CALL, ["h_flat"], "out", layer_ref="fc2"),
    ], inputs=["h_reshaped"], outputs=["out"],
       layers={"fc2": _linear("fc2", 64, 10)})

    return "ReshapeAcrossBreak", [sg1, sg2], {"x": ("batch", 64)}


def model_external_input_break() -> Tuple[str, List[ComputationGraph], Dict[str, Tuple]]:
    """External input at graph break — dynamic routing gap."""
    sg1 = _make_graph("ExtInput_G0", steps=[
        _step(OpKind.LAYER_CALL, ["x"], "features", layer_ref="encoder"),
        _step(OpKind.SUBSCRIPT, ["features"], "selected"),
    ], inputs=["x"], outputs=["selected"],
       layers={"encoder": _linear("encoder", 64, 128)})

    # Second subgraph receives an external input not from sg1
    sg2 = _make_graph("ExtInput_G1", steps=[
        _step(OpKind.LAYER_CALL, ["routing_mask"], "routed", layer_ref="router"),
        _step(OpKind.MATMUL, ["routed", "selected"], "out"),
    ], inputs=["routing_mask", "selected"], outputs=["out"],
       layers={"router": _linear("router", 32, 128)})

    return "ExternalInputBreak", [sg1, sg2], {"x": ("batch", 64)}


# ═══════════════════════════════════════════════════════════════════════════════
# Run all evaluations
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation() -> Dict[str, Any]:
    """Run thread-modular verification on all 10 model architectures."""
    monolithic_models = [
        model_simple_mlp,
        model_conv_classifier,
        model_residual_block,
        model_attention_head,
        model_autoencoder,
    ]
    safe_break_models = [
        model_two_stage_mlp,
        model_encoder_decoder_split,
        model_three_stage_pipeline,
    ]
    gap_models = [
        model_reshape_across_break,
        model_external_input_break,
    ]

    results: Dict[str, Any] = {
        "experiment": "thread_modular_verification",
        "models": [],
        "summary": {},
    }

    all_correct = True

    # Monolithic models
    for model_fn in monolithic_models:
        name, subgraphs, input_shapes = model_fn()
        t0 = time.monotonic()
        result = verify_thread_modular(subgraphs, input_shapes)
        elapsed = (time.monotonic() - t0) * 1000

        expected = CompositionVerdict.MONOLITHIC_SAFE
        correct = result.verdict == expected
        if not correct:
            all_correct = False

        results["models"].append({
            "name": name,
            "category": "monolithic",
            "num_subgraphs": result.num_subgraphs,
            "verdict": result.verdict.name,
            "expected": expected.name,
            "correct": correct,
            "num_gaps": len(result.gaps),
            "time_ms": round(elapsed, 2),
        })
        print(f"  {'✓' if correct else '✗'} {name}: {result.verdict.name} "
              f"(expected {expected.name}) [{elapsed:.1f}ms]")

    # Safe break models
    for model_fn in safe_break_models:
        name, subgraphs, input_shapes = model_fn()
        t0 = time.monotonic()
        result = verify_thread_modular(subgraphs, input_shapes)
        elapsed = (time.monotonic() - t0) * 1000

        expected = CompositionVerdict.COMPOSITION_VERIFIED
        correct = result.verdict == expected
        if not correct:
            all_correct = False

        results["models"].append({
            "name": name,
            "category": "safe_break",
            "num_subgraphs": result.num_subgraphs,
            "verdict": result.verdict.name,
            "expected": expected.name,
            "correct": correct,
            "num_gaps": len(result.gaps),
            "time_ms": round(elapsed, 2),
        })
        print(f"  {'✓' if correct else '✗'} {name}: {result.verdict.name} "
              f"(expected {expected.name}) [{elapsed:.1f}ms]")

    # Gap models
    for model_fn in gap_models:
        name, subgraphs, input_shapes = model_fn()
        t0 = time.monotonic()
        result = verify_thread_modular(subgraphs, input_shapes)
        elapsed = (time.monotonic() - t0) * 1000

        expected = CompositionVerdict.GAP_DETECTED
        correct = result.verdict == expected
        if not correct:
            all_correct = False

        gap_details = [
            {"category": g.category.value, "desc": g.description[:80]}
            for g in result.gaps
        ]

        results["models"].append({
            "name": name,
            "category": "gap",
            "num_subgraphs": result.num_subgraphs,
            "verdict": result.verdict.name,
            "expected": expected.name,
            "correct": correct,
            "num_gaps": len(result.gaps),
            "gap_details": gap_details,
            "time_ms": round(elapsed, 2),
        })
        print(f"  {'✓' if correct else '✗'} {name}: {result.verdict.name} "
              f"(expected {expected.name}, {len(result.gaps)} gaps) [{elapsed:.1f}ms]")

    # Summary
    total = len(results["models"])
    correct_count = sum(1 for m in results["models"] if m["correct"])
    results["summary"] = {
        "total_models": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0,
        "all_correct": all_correct,
        "monolithic_correct": sum(
            1 for m in results["models"]
            if m["category"] == "monolithic" and m["correct"]
        ),
        "safe_break_correct": sum(
            1 for m in results["models"]
            if m["category"] == "safe_break" and m["correct"]
        ),
        "gap_correct": sum(
            1 for m in results["models"]
            if m["category"] == "gap" and m["correct"]
        ),
    }

    return results


def main():
    print("=" * 70)
    print("Thread-Modular Verification Evaluation")
    print("=" * 70)
    print()

    results = run_evaluation()

    print()
    print("-" * 70)
    s = results["summary"]
    print(f"Results: {s['correct']}/{s['total_models']} correct "
          f"(accuracy: {s['accuracy']:.0%})")
    print(f"  Monolithic:  {s['monolithic_correct']}/5")
    print(f"  Safe breaks: {s['safe_break_correct']}/3")
    print(f"  Gap models:  {s['gap_correct']}/2")
    print("-" * 70)

    # Save results
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "thread_modular_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return 0 if s["all_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
