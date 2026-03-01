#!/usr/bin/env python3
"""
Run calibration analysis on existing neuro-symbolic pipeline results.

Loads benchmark results, computes all calibration metrics, and saves
a calibration report to experiments/results/calibration_results.json.
"""

from __future__ import annotations

import json
import os
import random
import sys

# Allow running from the implementation/ directory
_impl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _impl_dir not in sys.path:
    sys.path.insert(0, _impl_dir)

from src.calibration_analysis import (
    CONFIDENCE_MAP,
    Prediction,
    compute_calibration_report,
    load_predictions_from_results,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _generate_synthetic_predictions(n: int = 200, seed: int = 42) -> list[Prediction]:
    """Create synthetic pipeline-style predictions when no real data exists."""
    rng = random.Random(seed)
    confidence_levels = list(CONFIDENCE_MAP.items())
    preds: list[Prediction] = []

    for i in range(n):
        has_bug = rng.random() < 0.5
        conf_name, conf_val = rng.choice(confidence_levels)

        # Simulate reasonable pipeline behaviour: higher confidence → more
        # likely to be correct, but with some noise.
        if conf_val >= 0.85:
            correct = rng.random() < 0.88
        elif conf_val >= 0.60:
            correct = rng.random() < 0.70
        else:
            correct = rng.random() < 0.50

        if correct:
            pred_class = 1 if has_bug else 0
        else:
            pred_class = 0 if has_bug else 1

        true_class = 1 if has_bug else 0
        preds.append(Prediction(
            confidence=conf_val,
            predicted_class=pred_class,
            true_class=true_class,
            label_name=f"synthetic_{i}",
        ))
    return preds


def main() -> None:
    # Try loading real results first
    predictions = load_predictions_from_results(RESULTS_DIR)
    source = "pipeline_results"

    if not predictions:
        # Also search the parent experiments/ directory
        predictions = load_predictions_from_results(EXPERIMENTS_DIR)

    if not predictions:
        print("No pipeline result files found. Generating synthetic predictions.")
        predictions = _generate_synthetic_predictions()
        source = "synthetic"
    else:
        print(f"Loaded {len(predictions)} predictions from experiment results.")

    report = compute_calibration_report(predictions, n_bins=10)

    output = {
        "experiment": "calibration_analysis",
        "source": source,
        "description": (
            "Calibration analysis of the neuro-symbolic pipeline. "
            "Evaluates alignment between pipeline confidence and actual accuracy."
        ),
        **report.to_dict(),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nCalibration Report ({report.n_predictions} predictions)")
    print(f"  Brier score:     {report.brier_score:.4f}")
    print(f"  ECE:             {report.ece:.4f}")
    print(f"  MCE:             {report.mce:.4f}")
    print(f"  Calibration:     {report.calibration_component:.4f}")
    print(f"  Resolution:      {report.sharpness_component:.4f}")
    print(f"  Uncertainty:     {report.uncertainty_component:.4f}")
    print(f"  Mean confidence: {report.mean_confidence:.4f}")
    print(f"  Mean accuracy:   {report.mean_accuracy:.4f}")
    print(f"  Overconf ratio:  {report.overconfidence_ratio:.4f}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
