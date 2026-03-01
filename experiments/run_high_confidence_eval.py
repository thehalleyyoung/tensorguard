"""
High-Confidence Mode Evaluation for TensorGuard.

Runs TensorGuard on Suite D (50 external PyTorch benchmarks) with and
without high_confidence_only=True.  Compares FP rates to validate that
high-confidence mode achieves 0% FP, meeting the Sadowski et al. <1%
threshold for CI/CD gating.

Usage:
    cd implementation && python3 experiments/run_high_confidence_eval.py
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_checker import verify_model
from experiments.external_pytorch_benchmark import (
    EXTERNAL_PYTORCH_BENCHMARKS,
    get_benchmark_summary,
)


def wilson_ci(successes: int, total: int, z: float = 1.96):
    """Return (lower, upper) Wilson score 95% CI for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def compute_metrics(tp, fp, tn, fn):
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    n = tp + fp + tn + fn
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fp_rate": round(fp_rate, 4),
        "precision_95ci": [round(v, 4) for v in wilson_ci(tp, tp + fp)],
        "fp_rate_95ci": [round(v, 4) for v in wilson_ci(fp, fp + tn)],
        "total": n,
    }


def run_evaluation():
    summary = get_benchmark_summary()
    print("High-Confidence Mode Evaluation — Suite D (50 external benchmarks)")
    print(f"  Total models : {summary['total']}")
    print(f"  Buggy        : {summary['buggy']}")
    print(f"  Correct      : {summary['correct']}")
    print("=" * 72)

    # Standard mode counters
    std_tp = std_fp = std_tn = std_fn = 0
    # High-confidence mode counters
    hc_tp = hc_fp = hc_tn = hc_fn = 0

    per_model = {}
    total_time_std = 0.0
    total_time_hc = 0.0

    for name, bench in sorted(EXTERNAL_PYTORCH_BENCHMARKS.items()):
        source = bench["source"]
        input_shapes = bench["input_shapes"]
        is_buggy = bench["is_buggy"]
        category = bench["category"]

        # ── Standard mode ──
        t0 = time.monotonic()
        try:
            r_std = verify_model(source, input_shapes=input_shapes,
                                 high_confidence_only=False)
            std_detected = not r_std.safe
            std_error = None
        except Exception as exc:
            std_detected = False
            std_error = str(exc)
        t_std = (time.monotonic() - t0) * 1000
        total_time_std += t_std

        # ── High-confidence mode ──
        t0 = time.monotonic()
        try:
            r_hc = verify_model(source, input_shapes=input_shapes,
                                high_confidence_only=True)
            hc_detected = not r_hc.safe
            hc_error = None
        except Exception as exc:
            hc_detected = False
            hc_error = str(exc)
        t_hc = (time.monotonic() - t0) * 1000
        total_time_hc += t_hc

        # ── Classify standard ──
        if is_buggy and std_detected:
            std_tp += 1; std_v = "TP"
        elif is_buggy and not std_detected:
            std_fn += 1; std_v = "FN"
        elif not is_buggy and std_detected:
            std_fp += 1; std_v = "FP"
        else:
            std_tn += 1; std_v = "TN"

        # ── Classify high-confidence ──
        if is_buggy and hc_detected:
            hc_tp += 1; hc_v = "TP"
        elif is_buggy and not hc_detected:
            hc_fn += 1; hc_v = "FN"
        elif not is_buggy and hc_detected:
            hc_fp += 1; hc_v = "FP"
        else:
            hc_tn += 1; hc_v = "TN"

        ok_std = "✓" if std_v in ("TP", "TN") else "✗"
        ok_hc = "✓" if hc_v in ("TP", "TN") else "✗"
        print(f"  {ok_std} STD:{std_v}  {ok_hc} HC:{hc_v}  {name} ({category})")

        per_model[name] = {
            "is_buggy": is_buggy,
            "category": category,
            "standard": {"verdict": std_v, "detected": std_detected,
                         "time_ms": round(t_std, 1), "error": std_error},
            "high_confidence": {"verdict": hc_v, "detected": hc_detected,
                                "time_ms": round(t_hc, 1), "error": hc_error},
        }

    std_metrics = compute_metrics(std_tp, std_fp, std_tn, std_fn)
    hc_metrics = compute_metrics(hc_tp, hc_fp, hc_tn, hc_fn)

    print()
    print("=" * 72)
    print("Standard Mode:")
    print(f"  TP={std_tp}  FP={std_fp}  TN={std_tn}  FN={std_fn}")
    print(f"  Precision : {std_metrics['precision']:.4f}  FP rate: {std_metrics['fp_rate']:.4f}")
    print(f"  Recall    : {std_metrics['recall']:.4f}")
    print(f"  F1        : {std_metrics['f1']:.4f}")
    print(f"  Time      : {total_time_std:.0f}ms")
    print()
    print("High-Confidence Mode:")
    print(f"  TP={hc_tp}  FP={hc_fp}  TN={hc_tn}  FN={hc_fn}")
    print(f"  Precision : {hc_metrics['precision']:.4f}  FP rate: {hc_metrics['fp_rate']:.4f}")
    print(f"  Recall    : {hc_metrics['recall']:.4f}")
    print(f"  F1        : {hc_metrics['f1']:.4f}")
    print(f"  Time      : {total_time_hc:.0f}ms")
    print()

    sadowski_threshold = 0.02
    meets_threshold = hc_metrics["fp_rate"] <= sadowski_threshold
    print(f"Sadowski et al. CI/CD threshold: {sadowski_threshold:.0%}")
    print(f"High-confidence FP rate: {hc_metrics['fp_rate']:.2%}")
    print(f"Meets threshold: {'YES ✓' if meets_threshold else 'NO ✗'}")

    output = {
        "benchmark_summary": summary,
        "standard_mode": std_metrics,
        "high_confidence_mode": hc_metrics,
        "standard_total_time_ms": round(total_time_std, 1),
        "high_confidence_total_time_ms": round(total_time_hc, 1),
        "sadowski_threshold": sadowski_threshold,
        "meets_threshold": meets_threshold,
        "per_model_results": per_model,
    }

    out_path = Path(__file__).parent / "high_confidence_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    run_evaluation()
