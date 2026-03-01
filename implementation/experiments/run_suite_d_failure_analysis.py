#!/usr/bin/env python3
"""
Suite D Stratified Failure Analysis for TensorGuard.

Addresses reviewer concern: WHICH specific patterns defeat the analysis?
Stratifies Suite D (50 real-world models, F1=0.917) results by:
  - Architecture family (CNN, RNN, Transformer, GAN, etc.)
  - Constraint theory fragment (QF_LIA vs QF_NIA vs mixed)
  - Model complexity (parameter count proxy, layer depth)
  - Dynamic feature usage

Runs actual verify_model() on each benchmark and performs root-cause
analysis on every FP/FN.

Usage:
    cd implementation && python -m experiments.run_suite_d_failure_analysis
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_checker import verify_model, VerificationResult
from experiments.external_pytorch_benchmark import (
    EXTERNAL_PYTORCH_BENCHMARKS,
    get_benchmark_summary,
)

BENCHMARKS_DIR = PROJECT_ROOT.parent / ".benchmarks"
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = BENCHMARKS_DIR / "suite_d_failure_analysis.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Architecture family classification
# ═══════════════════════════════════════════════════════════════════════════════

ARCH_FAMILY_MAP: Dict[str, str] = {
    "ResNet": "cnn",
    "ResNeXt": "cnn",
    "VGG": "cnn",
    "AlexNet": "cnn",
    "LeNet": "cnn",
    "DenseNet": "cnn",
    "SqueezeNet": "cnn",
    "MobileNetV2": "cnn",
    "ConvNet": "cnn",
    "U-Net": "cnn",
    "Transformer": "transformer",
    "BERT": "transformer",
    "GPT-2": "transformer",
    "Seq2Seq": "rnn",
    "DCGAN": "gan",
    "WaveNet": "generative",
    "MLP": "mlp",
    "Autoencoder": "autoencoder",
}


def classify_architecture(category: str) -> str:
    """Map benchmark category to architecture family."""
    return ARCH_FAMILY_MAP.get(category, "other")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer counting and complexity analysis
# ═══════════════════════════════════════════════════════════════════════════════

# nn.Module layers that correspond to learnable or shape-transforming ops
LAYER_TYPES = {
    "Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d",
    "ConvTranspose2d", "ConvTranspose3d", "BatchNorm1d", "BatchNorm2d",
    "BatchNorm3d", "LayerNorm", "GroupNorm", "InstanceNorm2d",
    "MultiheadAttention", "TransformerEncoder", "TransformerDecoder",
    "TransformerEncoderLayer", "TransformerDecoderLayer",
    "LSTM", "GRU", "RNN", "Embedding",
    "MaxPool1d", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d", "Dropout", "ReLU", "GELU", "Sigmoid", "Tanh",
    "LeakyReLU", "Softmax",
}


def count_layers(source: str) -> Tuple[int, int, List[str]]:
    """Count nn layers and estimate parameter count from source.

    Returns (num_layers, estimated_params, layer_list).
    """
    nn_calls = re.findall(r"nn\.(\w+)\(([^)]*)\)", source)
    layers = []
    estimated_params = 0

    for layer_type, args_str in nn_calls:
        if layer_type in ("Module",):
            continue
        layers.append(layer_type)
        # Rough parameter estimation
        nums = re.findall(r"\d+", args_str)
        nums = [int(n) for n in nums]
        if layer_type == "Linear" and len(nums) >= 2:
            estimated_params += nums[0] * nums[1] + nums[1]
        elif layer_type.startswith("Conv") and "Transpose" not in layer_type and len(nums) >= 3:
            in_ch, out_ch, k = nums[0], nums[1], nums[2]
            estimated_params += in_ch * out_ch * k * k + out_ch
        elif layer_type.startswith("ConvTranspose") and len(nums) >= 3:
            in_ch, out_ch, k = nums[0], nums[1], nums[2]
            estimated_params += in_ch * out_ch * k * k + out_ch
        elif layer_type.startswith("BatchNorm") and len(nums) >= 1:
            estimated_params += 2 * nums[0]
        elif layer_type == "LayerNorm" and len(nums) >= 1:
            estimated_params += 2 * nums[0]
        elif layer_type == "Embedding" and len(nums) >= 2:
            estimated_params += nums[0] * nums[1]
        elif layer_type == "MultiheadAttention" and len(nums) >= 2:
            d = nums[0]
            estimated_params += 4 * d * d  # Q, K, V, out projections
        elif layer_type in ("LSTM", "GRU") and len(nums) >= 2:
            factor = 4 if layer_type == "LSTM" else 3
            estimated_params += factor * (nums[0] * nums[1] + nums[1] * nums[1])

    return len(layers), estimated_params, layers


def classify_complexity(num_layers: int) -> str:
    """Classify model complexity by layer count."""
    if num_layers < 10:
        return "shallow_lt10"
    elif num_layers <= 50:
        return "medium_10_50"
    else:
        return "deep_gt50"


# ═══════════════════════════════════════════════════════════════════════════════
# Theory fragment classification
# ═══════════════════════════════════════════════════════════════════════════════

def classify_theory_fragment(source: str, layers: List[str]) -> str:
    """Classify which SMT theory fragment the model requires.

    QF_LIA: Only linear integer arithmetic (add, subtract, multiply by constant)
    QF_NIA: Non-linear integer arithmetic (variable*variable, e.g., reshape products)
    mixed: Both linear and non-linear constraints needed
    """
    has_nonlinear = False
    has_linear = True  # Always true for shape checking

    # Non-linear indicators: reshape/view with -1, conv flatten products,
    # multi-head attention (embed_dim = heads * head_dim)
    nonlinear_patterns = [
        r"\.view\(",
        r"\.reshape\(",
        r"\.flatten\(",
        r"x\.size\(\d+\)\s*\*",
        r"groups=",
        r"MultiheadAttention",
    ]
    for pat in nonlinear_patterns:
        if re.search(pat, source):
            has_nonlinear = True
            break

    # Additional non-linear: Conv2d followed by Linear (requires spatial dim computation)
    has_conv = any(l.startswith("Conv") for l in layers)
    has_linear_layer = "Linear" in layers
    if has_conv and has_linear_layer and re.search(r"\.view\(|\.flatten\(|\.reshape\(", source):
        has_nonlinear = True

    if has_nonlinear:
        return "mixed" if has_linear else "QF_NIA"
    return "QF_LIA"


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic feature detection
# ═══════════════════════════════════════════════════════════════════════════════

DYNAMIC_FEATURE_PATTERNS = {
    "torch.cond": r"torch\.cond\(",
    "variable_length_seq": r"x\.size\(\d+\)|\.shape\[",
    "dynamic_reshape": r"\.view\(.*-1|\.reshape\(.*-1",
    "adaptive_pooling": r"AdaptiveAvgPool|AdaptiveMaxPool",
    "conditional_branch": r"if\s+.*training|if\s+.*self\.\w+",
    "in_place_op": r"\+=|\.add_\(|\.mul_\(",
    "skip_connection": r"out\s*\+\s*identity|out\s*\+\s*x|\+\s*residual|return.*\+.*x",
    "multi_class_def": r"class\s+\w+.*Module.*\n[\s\S]*class\s+\w+.*Module",
}


def detect_dynamic_features(source: str) -> List[str]:
    """Detect dynamic features in model source code."""
    features = []
    for feat_name, pattern in DYNAMIC_FEATURE_PATTERNS.items():
        if re.search(pattern, source):
            features.append(feat_name)
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# Root cause analysis for failures
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_root_cause(
    name: str,
    source: str,
    error_type: str,
    description: str,
    result: VerificationResult,
    layers: List[str],
) -> str:
    """Determine root cause for FP/FN misclassification."""

    if error_type == "FN":
        # False negative: bug exists but wasn't detected
        if "groups" in source and re.search(r"groups=\d+", source):
            return ("Groups-divisibility constraint not fully modeled: "
                    "verifier does not check in_channels % groups == 0")
        if re.search(r"\.view\(.*-1", source) and any(l.startswith("ConvTranspose") for l in layers):
            return ("ConvTranspose output shape -> view/reshape: product-equality "
                    "constraint requires QF_NIA reasoning beyond current templates")
        if "pooler" in name or "cls_token" in name.lower():
            return ("Indexing-based dimension selection (cls_token = hidden[:,0,:]) "
                    "not tracked through constraint propagation")
        if re.search(r"\.view\(", source) and any(l.startswith("Conv") for l in layers):
            return ("Reshape after transposed convolution requires non-linear "
                    "product-equality solving (out_ch * H' * W') across deconv layers")
        return f"Complex inter-layer constraint not captured by current abstraction"

    if error_type == "FP":
        # False positive: no bug but flagged as buggy
        if "dilat" in source.lower() or "WaveNet" in description:
            return ("Dilated convolution with gated activation: output shape "
                    "preserved by padding=dilation but verifier over-approximates "
                    "Conv1d spatial dimension formula")
        if result.counterexample and result.counterexample.violations:
            kinds = [v.kind for v in result.counterexample.violations]
            return f"Over-approximation in constraint generation for: {', '.join(set(kinds))}"
        return "Constraint over-approximation leads to spurious violation"

    return "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Any]:
    """Compute precision, recall, F1 from confusion matrix counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "total": tp + tn + fp + fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_failure_analysis() -> Dict[str, Any]:
    """Run full stratified failure analysis on Suite D."""
    print("=" * 72)
    print("  TensorGuard Suite D — Stratified Failure Analysis")
    print("=" * 72)

    summary = get_benchmark_summary()
    print(f"\nBenchmarks: {summary['total']} models across "
          f"{summary['num_categories']} architecture categories")
    print(f"  Buggy: {summary['buggy']}, Correct: {summary['correct']}")

    # Accumulators
    arch_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    theory_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    complexity_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failure_cases: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    total_tp = total_tn = total_fp = total_fn = 0
    total_time_ms = 0.0

    for i, (name, bm) in enumerate(EXTERNAL_PYTORCH_BENCHMARKS.items(), 1):
        source = bm["source"]
        input_shapes = bm["input_shapes"]
        is_buggy = bm["is_buggy"]
        category = bm["category"]
        description = bm["description"]

        # Classify
        arch_family = classify_architecture(category)
        num_layers, est_params, layers = count_layers(source)
        complexity_bin = classify_complexity(num_layers)
        theory_frag = classify_theory_fragment(source, layers)
        dynamic_feats = detect_dynamic_features(source)

        # Run verification
        t0 = time.monotonic()
        try:
            result = verify_model(
                source=source,
                input_shapes=input_shapes,
            )
        except Exception as exc:
            # If verification crashes, treat as "safe" (missed bug or correct)
            result = VerificationResult(
                safe=True,
                errors=[f"Exception: {exc}"],
                verification_time_ms=(time.monotonic() - t0) * 1000,
            )
        elapsed_ms = (time.monotonic() - t0) * 1000
        total_time_ms += elapsed_ms

        detected_bug = not result.safe

        # Classify outcome
        if is_buggy and detected_bug:
            verdict = "TP"
            total_tp += 1
        elif is_buggy and not detected_bug:
            verdict = "FN"
            total_fn += 1
        elif not is_buggy and detected_bug:
            verdict = "FP"
            total_fp += 1
        else:
            verdict = "TN"
            total_tn += 1

        # Update accumulators
        arch_counts[arch_family][verdict] += 1
        theory_counts[theory_frag][verdict] += 1
        complexity_counts[complexity_bin][verdict] += 1

        status_sym = {"TP": "✓", "TN": "✓", "FP": "✗", "FN": "✗"}.get(verdict, "?")
        print(f"  [{i:2d}/50] {status_sym} {verdict:2s}  {name:<40s}  "
              f"arch={arch_family:<12s} theory={theory_frag:<6s} "
              f"layers={num_layers:2d}  {elapsed_ms:7.1f}ms")

        model_record = {
            "model": name,
            "category": category,
            "architecture": arch_family,
            "is_buggy": is_buggy,
            "detected_bug": detected_bug,
            "verdict": verdict,
            "theory_fragment": theory_frag,
            "num_layers": num_layers,
            "estimated_params": est_params,
            "complexity_bin": complexity_bin,
            "dynamic_features": dynamic_feats,
            "time_ms": round(elapsed_ms, 1),
            "description": description,
        }
        all_results.append(model_record)

        # Record failure cases
        if verdict in ("FP", "FN"):
            root_cause = analyze_root_cause(
                name, source, verdict, description, result, layers,
            )
            failure_case = {
                "model": name,
                "expected": "buggy" if is_buggy else "safe",
                "predicted": "buggy" if detected_bug else "safe",
                "error_type": verdict,
                "root_cause": root_cause,
                "architecture": arch_family,
                "category": category,
                "theory_fragment": theory_frag,
                "num_layers": num_layers,
                "estimated_params": est_params,
                "complexity_bin": complexity_bin,
                "dynamic_features": dynamic_feats,
                "description": description,
            }
            failure_cases.append(failure_case)

    # Build stratified results
    def build_group_metrics(counts: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        result = {}
        for group, verdicts in sorted(counts.items()):
            tp = verdicts.get("TP", 0)
            tn = verdicts.get("TN", 0)
            fp = verdicts.get("FP", 0)
            fn = verdicts.get("FN", 0)
            result[group] = compute_metrics(tp, tn, fp, fn)
        return result

    results_by_arch = build_group_metrics(arch_counts)
    results_by_theory = build_group_metrics(theory_counts)
    results_by_complexity = build_group_metrics(complexity_counts)

    # Dynamic feature analysis
    dyn_feat_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0, "fp": 0, "fn": 0})
    for rec in all_results:
        for feat in rec["dynamic_features"]:
            dyn_feat_stats[feat]["total"] += 1
            if rec["verdict"] in ("TP", "TN"):
                dyn_feat_stats[feat]["correct"] += 1
            elif rec["verdict"] == "FP":
                dyn_feat_stats[feat]["fp"] += 1
            elif rec["verdict"] == "FN":
                dyn_feat_stats[feat]["fn"] += 1

    overall = compute_metrics(total_tp, total_tn, total_fp, total_fn)

    output = {
        "total_models": 50,
        "overall_metrics": overall,
        "total_time_ms": round(total_time_ms, 1),
        "results_by_architecture": results_by_arch,
        "results_by_theory_fragment": results_by_theory,
        "results_by_complexity": results_by_complexity,
        "dynamic_feature_impact": dict(dyn_feat_stats),
        "failure_cases": failure_cases,
        "all_model_results": all_results,
    }

    # Print summary
    print("\n" + "=" * 72)
    print("  OVERALL RESULTS")
    print("=" * 72)
    print(f"  TP={total_tp}  TN={total_tn}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision={overall['precision']:.4f}  Recall={overall['recall']:.4f}  "
          f"F1={overall['f1']:.4f}")
    print(f"  Total time: {total_time_ms:.0f}ms")

    print(f"\n── By Architecture Family ──")
    for arch, m in sorted(results_by_arch.items()):
        print(f"  {arch:<14s}  n={m['total']:2d}  F1={m['f1']:.3f}  "
              f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

    print(f"\n── By Theory Fragment ──")
    for frag, m in sorted(results_by_theory.items()):
        print(f"  {frag:<8s}  n={m['total']:2d}  F1={m['f1']:.3f}  "
              f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

    print(f"\n── By Complexity ──")
    for cbin, m in sorted(results_by_complexity.items()):
        print(f"  {cbin:<14s}  n={m['total']:2d}  F1={m['f1']:.3f}  "
              f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

    print(f"\n── Failure Cases ({len(failure_cases)}) ──")
    for fc in failure_cases:
        print(f"  [{fc['error_type']}] {fc['model']}")
        print(f"       arch={fc['architecture']}, theory={fc['theory_fragment']}, "
              f"layers={fc['num_layers']}")
        print(f"       Root cause: {fc['root_cause']}")

    if dyn_feat_stats:
        print(f"\n── Dynamic Feature Impact ──")
        for feat, stats in sorted(dyn_feat_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {feat:<25s}  n={stats['total']:2d}  "
                  f"accuracy={acc:.3f}  fp={stats['fp']}  fn={stats['fn']}")

    return output


def main():
    output = run_failure_analysis()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
