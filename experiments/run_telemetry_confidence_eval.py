#!/usr/bin/env python3
"""
Telemetry-based confidence evaluation experiment.

Demonstrates that solver telemetry features produce positive Brier resolution
(RES > 0), unlike the discrete 5-tier confidence mapping which yields RES=0.000.

Runs CEGAR verification on the benchmark suite, collects telemetry features,
trains a logistic regression via leave-one-out cross-validation, and reports
the Brier decomposition comparison.

Outputs: experiments/results/telemetry_confidence_results.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, IMPL_ROOT)

from src.calibration_analysis import CONFIDENCE_MAP, Prediction
from src.telemetry_confidence import (
    TelemetryConfidenceScorer,
    TelemetryFeatures,
    extract_telemetry_features,
    _brier_decomposition,
    _auc_roc,
)

RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "telemetry_confidence_results.json",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner — collect telemetry from CEGAR verification
# ═══════════════════════════════════════════════════════════════════════════════

def _get_test_cases() -> List[Dict[str, Any]]:
    """Import TEST_CASES from the CEGAR ablation v5 benchmark suite."""
    from experiments.run_cegar_ablation_v5 import TEST_CASES
    return TEST_CASES


def run_verification_with_telemetry(
    test_cases: List[Dict[str, Any]],
) -> Tuple[List[TelemetryFeatures], List[int], List[float], List[Dict[str, Any]]]:
    """Run CEGAR verification and collect telemetry features.

    Returns
    -------
    features_list : list of TelemetryFeatures
    outcomes : list of int  (1 = correct prediction, 0 = wrong)
    discrete_probs : list of float  (old discrete confidence)
    per_benchmark : list of dict  (per-benchmark details)
    """
    from src.shape_cegar import run_shape_cegar, CEGARStatus

    features_list: List[TelemetryFeatures] = []
    outcomes: List[int] = []
    discrete_probs: List[float] = []
    per_benchmark: List[Dict[str, Any]] = []

    for tc in test_cases:
        name = tc["name"]
        has_bug = tc["has_bug"]

        try:
            result = run_shape_cegar(
                tc["code"],
                input_shapes=tc["input_shapes"],
                max_iterations=10,
                enable_quality_filter=True,
            )
            detected = result.has_real_bugs

            # Count seed predicates (predicates added in iteration 0)
            seed_count = 0
            if result.iteration_log:
                preds_added = getattr(result.iteration_log[0], "predicates_added", [])
                seed_count = len(preds_added) if isinstance(preds_added, list) else int(preds_added)

            feats = extract_telemetry_features(result, seed_predicate_count=seed_count)

            # Discrete confidence (old system)
            status_name = result.final_status.name
            if status_name in ("SAFE", "REAL_BUG_FOUND"):
                conf_name = "FORMAL"
            elif status_name == "MAX_ITER":
                conf_name = "MEDIUM"
            else:
                conf_name = "LOW"
            discrete_prob = CONFIDENCE_MAP[conf_name]

            # Outcome: 1 if prediction matches ground truth
            prediction_correct = int(detected == has_bug)

        except Exception as exc:
            feats = TelemetryFeatures()
            detected = False
            conf_name = "LOW"
            discrete_prob = CONFIDENCE_MAP["LOW"]
            prediction_correct = int(not has_bug)  # not detecting = correct only if no bug

        features_list.append(feats)
        outcomes.append(prediction_correct)
        discrete_probs.append(discrete_prob)

        per_benchmark.append({
            "name": name,
            "has_bug": has_bug,
            "detected": detected,
            "confidence_level": conf_name,
            "discrete_prob": discrete_prob,
            "outcome_correct": prediction_correct,
            "telemetry_features": feats.to_dict(),
        })

    return features_list, outcomes, discrete_probs, per_benchmark


# ═══════════════════════════════════════════════════════════════════════════════
# Leave-one-out cross-validation
# ═══════════════════════════════════════════════════════════════════════════════

def loo_cross_validation(
    features_list: List[TelemetryFeatures],
    outcomes: List[int],
) -> List[float]:
    """Leave-one-out cross-validated telemetry confidence scores."""
    n = len(features_list)
    loo_probs: List[float] = []

    for i in range(n):
        train_f = features_list[:i] + features_list[i + 1:]
        train_y = outcomes[:i] + outcomes[i + 1:]
        scorer = TelemetryConfidenceScorer()
        scorer.fit(train_f, train_y, lr=0.1, epochs=500, reg_lambda=0.01)
        loo_probs.append(scorer.predict_confidence(features_list[i]))

    return loo_probs


def kfold_cross_validation(
    features_list: List[TelemetryFeatures],
    outcomes: List[int],
    k: int = 5,
) -> List[float]:
    """K-fold cross-validated telemetry confidence scores."""
    n = len(features_list)
    fold_size = max(1, n // k)
    cv_probs = [0.5] * n

    for fold in range(k):
        val_start = fold * fold_size
        val_end = min(val_start + fold_size, n)
        if fold == k - 1:
            val_end = n

        train_f = features_list[:val_start] + features_list[val_end:]
        train_y = outcomes[:val_start] + outcomes[val_end:]

        scorer = TelemetryConfidenceScorer()
        scorer.fit(train_f, train_y, lr=0.1, epochs=500, reg_lambda=0.01)

        for i in range(val_start, val_end):
            cv_probs[i] = scorer.predict_confidence(features_list[i])

    return cv_probs


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment() -> Dict[str, Any]:
    """Run the full telemetry confidence evaluation."""
    t0 = time.monotonic()
    print("=" * 70)
    print("Telemetry-Based Confidence Evaluation")
    print("=" * 70)

    # Load benchmarks
    test_cases = _get_test_cases()
    print(f"\nLoaded {len(test_cases)} benchmarks from CEGAR ablation v5 suite")

    # Run verification and collect telemetry
    print("\nRunning CEGAR verification with telemetry collection...")
    features_list, outcomes, discrete_probs, per_benchmark = (
        run_verification_with_telemetry(test_cases)
    )
    n = len(features_list)
    print(f"  Collected telemetry for {n} benchmarks")
    print(f"  Correct predictions: {sum(outcomes)}/{n}")

    # ── Old discrete-confidence Brier decomposition ──────────────────────
    y = [float(o) for o in outcomes]
    old_brier = sum((p - yi) ** 2 for p, yi in zip(discrete_probs, y)) / n
    old_rel, old_res, old_unc = _brier_decomposition(y, discrete_probs)

    print(f"\n--- Discrete Confidence (old system) ---")
    print(f"  Brier score:  {old_brier:.6f}")
    print(f"  REL (reliability): {old_rel:.6f}")
    print(f"  RES (resolution):  {old_res:.6f}")
    print(f"  UNC (uncertainty): {old_unc:.6f}")

    # ── Telemetry-based LOO cross-validation ─────────────────────────────
    print(f"\nRunning leave-one-out cross-validation ({n} folds)...")
    loo_probs = loo_cross_validation(features_list, outcomes)

    new_brier = sum((p - yi) ** 2 for p, yi in zip(loo_probs, y)) / n
    new_rel, new_res, new_unc = _brier_decomposition(y, loo_probs)
    new_auc = _auc_roc(y, loo_probs)

    print(f"\n--- Telemetry Confidence (LOO cross-validated) ---")
    print(f"  Brier score:  {new_brier:.6f}")
    print(f"  REL (reliability): {new_rel:.6f}")
    print(f"  RES (resolution):  {new_res:.6f}  {'✓ POSITIVE' if new_res > 0 else '✗ ZERO'}")
    print(f"  UNC (uncertainty): {new_unc:.6f}")
    print(f"  AUC-ROC:           {new_auc:.4f}")

    # ── K-fold cross-validation (k=5) for comparison ─────────────────────
    if n >= 5:
        kfold_probs = kfold_cross_validation(features_list, outcomes, k=5)
        kf_brier = sum((p - yi) ** 2 for p, yi in zip(kfold_probs, y)) / n
        kf_rel, kf_res, kf_unc = _brier_decomposition(y, kfold_probs)
        kf_auc = _auc_roc(y, kfold_probs)
    else:
        kfold_probs = loo_probs
        kf_brier = new_brier
        kf_rel, kf_res, kf_unc = new_rel, new_res, new_unc
        kf_auc = new_auc

    # ── Train full model and report feature importance ───────────────────
    full_scorer = TelemetryConfidenceScorer()
    full_scorer.fit(features_list, outcomes, lr=0.1, epochs=1000, reg_lambda=0.01)
    eval_result = full_scorer.evaluate(features_list, outcomes)

    feature_names = TelemetryFeatures.feature_names()
    feature_importance = {}
    if full_scorer._fitted and full_scorer.weights:
        for name, w in zip(feature_names, full_scorer.weights):
            feature_importance[name] = round(abs(w), 6)

    # Sort by importance
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: -x[1]
    )

    print(f"\n--- Feature Importance (|weight|) ---")
    for fname, imp in sorted_features[:7]:
        print(f"  {fname:>25s}: {imp:.4f}")

    # ── Summary statistics ───────────────────────────────────────────────
    loo_distinct = len(set(round(p, 4) for p in loo_probs))
    disc_distinct = len(set(round(p, 4) for p in discrete_probs))

    print(f"\n--- Comparison Summary ---")
    print(f"  Discrete confidence distinct values: {disc_distinct}")
    print(f"  Telemetry confidence distinct values: {loo_distinct}")
    print(f"  Old RES: {old_res:.6f}  →  New RES: {new_res:.6f}")
    print(f"  Resolution improvement: {new_res - old_res:+.6f}")
    print(f"  Resolution positive (key metric): {'YES ✓' if new_res > 0 else 'NO ✗'}")

    elapsed = time.monotonic() - t0

    # ── Assemble results ─────────────────────────────────────────────────
    results = {
        "experiment": "telemetry_confidence_evaluation",
        "description": (
            "Compares discrete 5-tier confidence (RES=0) against logistic "
            "regression on solver telemetry features. Cross-validated to "
            "avoid overfitting. Key metric: Brier resolution > 0."
        ),
        "n_benchmarks": n,
        "n_correct_predictions": sum(outcomes),
        "discrete_confidence": {
            "brier_score": round(old_brier, 6),
            "REL": round(old_rel, 6),
            "RES": round(old_res, 6),
            "UNC": round(old_unc, 6),
            "n_distinct_values": disc_distinct,
        },
        "telemetry_confidence_loo": {
            "brier_score": round(new_brier, 6),
            "REL": round(new_rel, 6),
            "RES": round(new_res, 6),
            "UNC": round(new_unc, 6),
            "auc_roc": round(new_auc, 6),
            "n_distinct_values": loo_distinct,
        },
        "telemetry_confidence_kfold": {
            "brier_score": round(kf_brier, 6),
            "REL": round(kf_rel, 6),
            "RES": round(kf_res, 6),
            "UNC": round(kf_unc, 6),
            "auc_roc": round(kf_auc, 6),
            "k": 5 if n >= 5 else n,
        },
        "full_model_eval": eval_result,
        "feature_importance": dict(sorted_features),
        "resolution_improvement": round(new_res - old_res, 6),
        "resolution_positive": new_res > 0,
        "key_result": (
            f"Telemetry confidence achieves RES={new_res:.6f} vs discrete "
            f"RES={old_res:.6f}. Resolution is "
            f"{'POSITIVE (non-vacuous)' if new_res > 0 else 'still zero'}."
        ),
        "per_benchmark": per_benchmark,
        "elapsed_sec": round(elapsed, 2),
    }

    # Save
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total time: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    run_experiment()
