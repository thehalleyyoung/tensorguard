"""
External Benchmark Evaluation for TensorGuard.

Runs TensorGuard (verify_model) and a simple PyTea-style baseline on
50 real-world PyTorch nn.Module benchmarks.  Computes TP/FP/TN/FN,
precision, recall, F1, and Wilson score 95% confidence intervals.

Usage:
    cd implementation && python experiments/run_external_benchmark.py
"""

import ast
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_checker import verify_model
from experiments.external_pytorch_benchmark import (
    EXTERNAL_PYTORCH_BENCHMARKS,
    get_benchmark_summary,
)


# ── Wilson score confidence interval ────────────────────────────────────────

def wilson_ci(successes: int, total: int, z: float = 1.96):
    """Return (lower, upper) Wilson score 95% CI for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ── PyTea-style baseline ───────────────────────────────────────────────────
# Simulates what PyTea does: check each nn layer in isolation for obvious
# parameter errors (e.g., groups not dividing channels) but does NOT
# track shapes across layers.

def pytea_baseline_check(source: str, input_shapes: dict) -> bool:
    """Return True if the baseline detects a bug (very shallow check).

    Only checks:
      - nn.Conv*d groups divides in_channels and out_channels
      - nn.MultiheadAttention embed_dim divisible by num_heads
    Does NOT propagate shapes between layers.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id

        # Check Conv groups
        if name in ("Conv1d", "Conv2d", "Conv3d"):
            kw = {k.arg: k.value for k in node.keywords}
            groups_node = kw.get("groups")
            if groups_node is not None:
                try:
                    groups = ast.literal_eval(groups_node)
                except Exception:
                    continue
                # get in_channels (first positional arg)
                if len(node.args) >= 1:
                    try:
                        in_ch = ast.literal_eval(node.args[0])
                        if in_ch % groups != 0:
                            return True
                    except Exception:
                        pass
                if len(node.args) >= 2:
                    try:
                        out_ch = ast.literal_eval(node.args[1])
                        if out_ch % groups != 0:
                            return True
                    except Exception:
                        pass

        # Check MultiheadAttention
        if name == "MultiheadAttention":
            if len(node.args) >= 2:
                try:
                    embed_dim = ast.literal_eval(node.args[0])
                    num_heads = ast.literal_eval(node.args[1])
                    if embed_dim % num_heads != 0:
                        return True
                except Exception:
                    pass

    return False


# ── Main evaluation ────────────────────────────────────────────────────────

def run_evaluation():
    summary = get_benchmark_summary()
    print(f"External PyTorch Benchmark Suite")
    print(f"  Total models : {summary['total']}")
    print(f"  Buggy        : {summary['buggy']}")
    print(f"  Correct      : {summary['correct']}")
    print(f"  Categories   : {', '.join(summary['categories'])}")
    print("=" * 72)

    # TensorGuard results
    tg_tp = tg_fp = tg_tn = tg_fn = 0
    # Baseline results
    bl_tp = bl_fp = bl_tn = bl_fn = 0

    per_model = {}
    total_time_ms = 0.0

    for name, bench in sorted(EXTERNAL_PYTORCH_BENCHMARKS.items()):
        source = bench["source"]
        input_shapes = bench["input_shapes"]
        is_buggy = bench["is_buggy"]
        category = bench["category"]

        # ── TensorGuard ──
        t0 = time.monotonic()
        try:
            result = verify_model(source, input_shapes=input_shapes)
            tg_detected = not result.safe
            tg_error = None
        except Exception as exc:
            tg_detected = False
            tg_error = str(exc)
        tg_time = (time.monotonic() - t0) * 1000
        total_time_ms += tg_time

        # ── PyTea baseline ──
        bl_detected = pytea_baseline_check(source, input_shapes)

        # ── TensorGuard classification ──
        if is_buggy and tg_detected:
            tg_verdict = "TP"; tg_tp += 1
        elif is_buggy and not tg_detected:
            tg_verdict = "FN"; tg_fn += 1
        elif not is_buggy and tg_detected:
            tg_verdict = "FP"; tg_fp += 1
        else:
            tg_verdict = "TN"; tg_tn += 1

        # ── Baseline classification ──
        if is_buggy and bl_detected:
            bl_verdict = "TP"; bl_tp += 1
        elif is_buggy and not bl_detected:
            bl_verdict = "FN"; bl_fn += 1
        elif not is_buggy and bl_detected:
            bl_verdict = "FP"; bl_fp += 1
        else:
            bl_verdict = "TN"; bl_tn += 1

        ok = "✓" if tg_verdict in ("TP", "TN") else "✗"
        err_str = f" [ERROR: {tg_error}]" if tg_error else ""
        print(f"  {ok} TG:{tg_verdict}  BL:{bl_verdict}  {name} ({category}) "
              f"[{tg_time:.0f}ms]{err_str}")

        per_model[name] = {
            "is_buggy": is_buggy,
            "category": category,
            "description": bench["description"],
            "tensorguard": {
                "verdict": tg_verdict,
                "detected_bug": tg_detected,
                "time_ms": round(tg_time, 1),
                "error": tg_error,
            },
            "baseline": {
                "verdict": bl_verdict,
                "detected_bug": bl_detected,
            },
        }

    # ── Metrics ──────────────────────────────────────────────────────────
    def metrics(tp, fp, tn, fn):
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
        n = tp + fp + tn + fn
        prec_ci = wilson_ci(tp, tp + fp) if (tp + fp) else (0.0, 0.0)
        rec_ci = wilson_ci(tp, tp + fn) if (tp + fn) else (0.0, 0.0)
        acc_ci = wilson_ci(tp + tn, n)
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "precision_95ci": [round(prec_ci[0], 4), round(prec_ci[1], 4)],
            "recall_95ci": [round(rec_ci[0], 4), round(rec_ci[1], 4)],
            "accuracy_95ci": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
        }

    tg_metrics = metrics(tg_tp, tg_fp, tg_tn, tg_fn)
    bl_metrics = metrics(bl_tp, bl_fp, bl_tn, bl_fn)

    print()
    print("=" * 72)
    print("TensorGuard Results:")
    print(f"  TP={tg_tp}  FP={tg_fp}  TN={tg_tn}  FN={tg_fn}")
    print(f"  Precision : {tg_metrics['precision']:.4f}  "
          f"95%CI {tg_metrics['precision_95ci']}")
    print(f"  Recall    : {tg_metrics['recall']:.4f}  "
          f"95%CI {tg_metrics['recall_95ci']}")
    print(f"  F1        : {tg_metrics['f1']:.4f}")
    print(f"  Accuracy  : {tg_metrics['accuracy']:.4f}  "
          f"95%CI {tg_metrics['accuracy_95ci']}")
    print(f"  Total time: {total_time_ms:.0f}ms")
    print()
    print("PyTea-style Baseline Results:")
    print(f"  TP={bl_tp}  FP={bl_fp}  TN={bl_tn}  FN={bl_fn}")
    print(f"  Precision : {bl_metrics['precision']:.4f}  "
          f"95%CI {bl_metrics['precision_95ci']}")
    print(f"  Recall    : {bl_metrics['recall']:.4f}  "
          f"95%CI {bl_metrics['recall_95ci']}")
    print(f"  F1        : {bl_metrics['f1']:.4f}")
    print(f"  Accuracy  : {bl_metrics['accuracy']:.4f}  "
          f"95%CI {bl_metrics['accuracy_95ci']}")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "benchmark_summary": summary,
        "tensorguard": tg_metrics,
        "baseline_pytea_style": bl_metrics,
        "total_time_ms": round(total_time_ms, 1),
        "per_model_results": per_model,
    }

    out_path = Path(__file__).parent / "external_benchmark_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    run_evaluation()
