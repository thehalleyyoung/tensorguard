#!/usr/bin/env python3
"""Bootstrap confidence interval analysis for per-category F1 scores."""

import json
import random
import os
import sys

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "external_benchmark_results.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "bootstrap_ci_results.json")

N_BOOTSTRAP = 10000
CI_LOWER = 2.5
CI_UPPER = 97.5


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def compute_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_accuracy(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def bootstrap_metric(samples, metric_fn, n_bootstrap=N_BOOTSTRAP):
    """Resample with replacement and compute metric each time."""
    n = len(samples)
    scores = []
    for _ in range(n_bootstrap):
        resample = [samples[random.randint(0, n - 1)] for _ in range(n)]
        scores.append(metric_fn(resample))
    scores.sort()
    lo = scores[int(n_bootstrap * CI_LOWER / 100)]
    hi = scores[int(n_bootstrap * CI_UPPER / 100) - 1]
    return lo, hi


def f1_from_verdicts(verdicts):
    tp = sum(1 for v in verdicts if v == "TP")
    fp = sum(1 for v in verdicts if v == "FP")
    fn = sum(1 for v in verdicts if v == "FN")
    return compute_f1(tp, fp, fn)


def accuracy_from_verdicts(verdicts):
    tp = sum(1 for v in verdicts if v == "TP")
    tn = sum(1 for v in verdicts if v == "TN")
    total = len(verdicts)
    return (tp + tn) / total if total > 0 else 0.0


def precision_from_verdicts(verdicts):
    tp = sum(1 for v in verdicts if v == "TP")
    fp = sum(1 for v in verdicts if v == "FP")
    return compute_precision(tp, fp)


def recall_from_verdicts(verdicts):
    tp = sum(1 for v in verdicts if v == "TP")
    fn = sum(1 for v in verdicts if v == "FN")
    return compute_recall(tp, fn)


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    per_model = data["per_model_results"]

    # Group verdicts by category
    categories = {}
    all_verdicts = []
    for model_name, info in per_model.items():
        cat = info["category"]
        verdict = info["tensorguard"]["verdict"]
        categories.setdefault(cat, []).append(verdict)
        all_verdicts.append(verdict)

    # Per-category analysis
    per_category = {}
    for cat in sorted(categories.keys()):
        verdicts = categories[cat]
        n = len(verdicts)
        tp = sum(1 for v in verdicts if v == "TP")
        fp = sum(1 for v in verdicts if v == "FP")
        tn = sum(1 for v in verdicts if v == "TN")
        fn = sum(1 for v in verdicts if v == "FN")

        f1 = compute_f1(tp, fp, fn)
        acc = compute_accuracy(tp, tn, fp, fn)

        f1_lo, f1_hi = bootstrap_metric(verdicts, f1_from_verdicts)
        acc_lo, acc_hi = bootstrap_metric(verdicts, accuracy_from_verdicts)

        per_category[cat] = {
            "n": n,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "f1": round(f1, 4),
            "f1_95ci_lower": round(f1_lo, 4),
            "f1_95ci_upper": round(f1_hi, 4),
            "accuracy": round(acc, 4),
            "accuracy_95ci_lower": round(acc_lo, 4),
            "accuracy_95ci_upper": round(acc_hi, 4),
        }

    # Aggregate analysis
    agg_tp = sum(1 for v in all_verdicts if v == "TP")
    agg_fp = sum(1 for v in all_verdicts if v == "FP")
    agg_tn = sum(1 for v in all_verdicts if v == "TN")
    agg_fn = sum(1 for v in all_verdicts if v == "FN")

    agg_f1 = compute_f1(agg_tp, agg_fp, agg_fn)
    agg_prec = compute_precision(agg_tp, agg_fp)
    agg_rec = compute_recall(agg_tp, agg_fn)
    agg_acc = compute_accuracy(agg_tp, agg_tn, agg_fp, agg_fn)

    f1_lo, f1_hi = bootstrap_metric(all_verdicts, f1_from_verdicts)
    prec_lo, prec_hi = bootstrap_metric(all_verdicts, precision_from_verdicts)
    rec_lo, rec_hi = bootstrap_metric(all_verdicts, recall_from_verdicts)
    acc_lo, acc_hi = bootstrap_metric(all_verdicts, accuracy_from_verdicts)

    aggregate = {
        "n": len(all_verdicts),
        "f1": round(agg_f1, 4),
        "f1_95ci": [round(f1_lo, 4), round(f1_hi, 4)],
        "precision_95ci": [round(prec_lo, 4), round(prec_hi, 4)],
        "recall_95ci": [round(rec_lo, 4), round(rec_hi, 4)],
        "accuracy_95ci": [round(acc_lo, 4), round(acc_hi, 4)],
    }

    result = {
        "per_category": per_category,
        "aggregate": aggregate,
        "methodology": (
            "Bootstrap confidence intervals computed via the percentile method "
            "with 10,000 resamples (with replacement) per metric. For each "
            "category, the verdict labels (TP/FP/TN/FN) are resampled and F1 "
            "and accuracy are recomputed on each bootstrap sample. The 2.5th "
            "and 97.5th percentiles of the bootstrap distribution form the 95% "
            "CI. Random seed fixed at 42 for reproducibility."
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {OUTPUT_PATH}")
    print(f"\nAggregate F1: {agg_f1:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]")
    print(f"Aggregate Accuracy: {agg_acc:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]")
    print(f"\nPer-category results ({len(per_category)} categories):")
    for cat, info in sorted(per_category.items()):
        print(f"  {cat:15s}  n={info['n']}  F1={info['f1']:.4f} [{info['f1_95ci_lower']:.4f}, {info['f1_95ci_upper']:.4f}]  Acc={info['accuracy']:.4f} [{info['accuracy_95ci_lower']:.4f}, {info['accuracy_95ci_upper']:.4f}]")


if __name__ == "__main__":
    main()
