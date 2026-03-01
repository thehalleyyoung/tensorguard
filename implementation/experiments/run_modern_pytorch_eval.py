#!/usr/bin/env python3
"""
Runner for modern PyTorch pattern benchmarks.

Evaluates TensorGuard's detection and verification of modern PyTorch
patterns: torch.compile, mixed-precision, DDP, DataParallel,
data-dependent control flow, JIT, and torch.export.
"""

import json
import os
import sys
import time

# Ensure the implementation root is on the path
IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, IMPL_ROOT)

from benchmarks.modern_pytorch.modern_pytorch_benchmarks import (
    MODERN_PYTORCH_BENCHMARKS,
)
from src.model_checker import verify_model


def run_benchmarks():
    results = {}
    summary = {
        "total": 0,
        "verified_safe": 0,
        "verified_unsafe": 0,
        "errors": 0,
        "correct_verdicts": 0,
        "features_detected": 0,
        "features_expected": 0,
        "feature_detection_recall": 0.0,
    }
    total_features_expected = 0
    total_features_detected = 0

    for name, bench in MODERN_PYTORCH_BENCHMARKS.items():
        summary["total"] += 1
        t0 = time.monotonic()

        try:
            result = verify_model(
                source=bench["source"],
                input_shapes=bench["input_shapes"],
            )
            elapsed = (time.monotonic() - t0) * 1000

            # Check verdict correctness
            is_buggy = bench["is_buggy"]
            if is_buggy:
                correct = not result.safe  # should detect bug
            else:
                correct = result.safe  # should be safe

            if correct:
                summary["correct_verdicts"] += 1

            if result.safe:
                summary["verified_safe"] += 1
            else:
                summary["verified_unsafe"] += 1

            # Check feature detection
            expected_features = bench.get("expected_features", [])
            detected_features = result.dynamic_features
            detected_set = set()
            for ef in expected_features:
                total_features_expected += 1
                if ef in detected_features and detected_features[ef]:
                    total_features_detected += 1
                    detected_set.add(ef)

            results[name] = {
                "verdict": "safe" if result.safe else "unsafe",
                "expected_buggy": is_buggy,
                "correct": correct,
                "category": bench["category"],
                "description": bench["description"],
                "dynamic_features_detected": {
                    k: v for k, v in detected_features.items()
                    if not isinstance(v, list)  # skip large lists for JSON
                },
                "dynamic_feature_warnings": result.dynamic_feature_warnings,
                "expected_features": expected_features,
                "expected_features_found": list(detected_set),
                "expected_features_missed": [
                    f for f in expected_features if f not in detected_set
                ],
                "verification_time_ms": round(elapsed, 2),
                "errors": result.errors,
            }

        except Exception as e:
            summary["errors"] += 1
            results[name] = {
                "verdict": "error",
                "expected_buggy": bench["is_buggy"],
                "correct": False,
                "category": bench["category"],
                "description": bench["description"],
                "error": str(e),
            }

    summary["features_expected"] = total_features_expected
    summary["features_detected"] = total_features_detected
    if total_features_expected > 0:
        summary["feature_detection_recall"] = round(
            total_features_detected / total_features_expected, 3
        )

    output = {
        "experiment": "modern_pytorch_pattern_detection",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summary,
        "benchmarks": results,
    }
    return output


def main():
    output = run_benchmarks()

    out_path = os.path.join(
        IMPL_ROOT, "experiments", "modern_pytorch_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results written to {out_path}")
    print(f"\nSummary:")
    s = output["summary"]
    print(f"  Total benchmarks: {s['total']}")
    print(f"  Correct verdicts: {s['correct_verdicts']}/{s['total']}")
    print(f"  Feature detection recall: {s['feature_detection_recall']}")
    print(f"  Verified safe:   {s['verified_safe']}")
    print(f"  Verified unsafe: {s['verified_unsafe']}")
    print(f"  Errors:          {s['errors']}")

    print("\nPer-benchmark results:")
    for name, r in output["benchmarks"].items():
        marker = "✓" if r["correct"] else "✗"
        feat_missed = r.get("expected_features_missed", [])
        feat_str = f" (missed features: {feat_missed})" if feat_missed else ""
        print(f"  {marker} {name}: {r['verdict']} "
              f"(expected {'buggy' if r['expected_buggy'] else 'safe'})"
              f"{feat_str}")
        for w in r.get("dynamic_feature_warnings", []):
            print(f"      ⚠ {w[:80]}...")


if __name__ == "__main__":
    main()
