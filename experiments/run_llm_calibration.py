#!/usr/bin/env python3
"""
LLM Calibration Experiment for the Neuro-Symbolic Pipeline.

Computes Expected Calibration Error (ECE) for the LLM component of the
hybrid TensorGuard pipeline. Measures how well-calibrated the LLM's
self-reported confidence scores are relative to actual accuracy.

Generates:
  - ECE (Expected Calibration Error)
  - Reliability diagram data (binned accuracy vs confidence)
  - Per-bin statistics

The experiment uses the NeurosymPipeline's LLM stage (or a simulation
if no API key is available) against the expanded benchmark suite.

Outputs: implementation/experiments/results/llm_calibration.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "llm_calibration.json"

NUM_BINS = 10


@dataclass
class CalibrationSample:
    """A single calibration sample."""
    name: str
    ground_truth_has_bug: bool
    llm_predicts_bug: Optional[bool]
    llm_confidence: float
    llm_correct: bool


@dataclass
class CalibrationBin:
    """A single bin in the reliability diagram."""
    bin_lower: float
    bin_upper: float
    avg_confidence: float
    avg_accuracy: float
    count: int
    gap: float  # |avg_accuracy - avg_confidence|


def compute_ece(samples: List[CalibrationSample], num_bins: int = NUM_BINS) -> Tuple[float, List[CalibrationBin]]:
    """Compute Expected Calibration Error and reliability diagram data.

    ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|

    where B_b is the set of samples in bin b.
    """
    bins: List[List[CalibrationSample]] = [[] for _ in range(num_bins)]

    for s in samples:
        if s.llm_predicts_bug is None:
            continue
        bin_idx = min(int(s.llm_confidence * num_bins), num_bins - 1)
        bins[bin_idx].append(s)

    n = sum(len(b) for b in bins)
    if n == 0:
        return 0.0, []

    ece = 0.0
    diagram: List[CalibrationBin] = []

    for i, bin_samples in enumerate(bins):
        lower = i / num_bins
        upper = (i + 1) / num_bins

        if not bin_samples:
            diagram.append(CalibrationBin(
                bin_lower=round(lower, 2),
                bin_upper=round(upper, 2),
                avg_confidence=0.0,
                avg_accuracy=0.0,
                count=0,
                gap=0.0,
            ))
            continue

        avg_conf = sum(s.llm_confidence for s in bin_samples) / len(bin_samples)
        avg_acc = sum(1 for s in bin_samples if s.llm_correct) / len(bin_samples)
        gap = abs(avg_acc - avg_conf)

        ece += (len(bin_samples) / n) * gap

        diagram.append(CalibrationBin(
            bin_lower=round(lower, 2),
            bin_upper=round(upper, 2),
            avg_confidence=round(avg_conf, 4),
            avg_accuracy=round(avg_acc, 4),
            count=len(bin_samples),
            gap=round(gap, 4),
        ))

    return round(ece, 4), diagram


def _load_benchmarks() -> List[Dict[str, Any]]:
    """Load benchmarks from expanded_benchmark_suite."""
    try:
        from experiments.expanded_benchmark_suite import EXPANDED_BENCHMARKS
        return EXPANDED_BENCHMARKS
    except ImportError:
        pass
    try:
        sys.path.insert(0, str(EXPERIMENTS_DIR))
        from expanded_benchmark_suite import EXPANDED_BENCHMARKS
        return EXPANDED_BENCHMARKS
    except ImportError:
        return []


def _try_llm_analysis(code: str) -> Tuple[Optional[bool], float]:
    """Attempt LLM analysis using NeurosymPipeline's LLM stage.

    Returns (predicts_bug, confidence). Falls back to a heuristic
    simulation if no API key or openai package is available.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from src.neurosym_pipeline import NeurosymPipeline
            pipeline = NeurosymPipeline(openai_api_key=api_key)
            if pipeline._client is not None:
                from src.neurosym_pipeline import _query_llm
                analysis = _query_llm(pipeline._client, code, pipeline.model)
                return analysis.predicts_bug, analysis.confidence
        except Exception:
            pass

    # Fallback: heuristic simulation for calibration methodology demo
    # Uses simple code-pattern matching as a proxy for LLM behaviour
    import hashlib
    import re

    # Deterministic pseudo-random based on code hash
    code_hash = int(hashlib.sha256(code.encode()).hexdigest()[:8], 16)
    base_conf = 0.5 + (code_hash % 40) / 100.0  # 0.50–0.89

    # Simple heuristic: look for common bug indicators
    bug_signals = 0
    if re.search(r'nn\.Linear\(\d+,\s*\d+\).*nn\.Linear\(\d+,\s*\d+\)', code, re.DOTALL):
        # Check if Linear chain dimensions match
        linears = re.findall(r'nn\.Linear\((\d+),\s*(\d+)\)', code)
        for i in range(len(linears) - 1):
            if linears[i][1] != linears[i + 1][0]:
                bug_signals += 2
    if '.cuda()' in code and '.cpu()' in code:
        bug_signals += 1
    if 'self.training' in code:
        bug_signals += 0  # Not necessarily a bug

    predicts_bug = bug_signals >= 2
    confidence = min(base_conf + bug_signals * 0.05, 0.99)
    return predicts_bug, round(confidence, 2)


def main():
    benchmarks = _load_benchmarks()
    if not benchmarks:
        print("ERROR: Could not load benchmarks. Ensure expanded_benchmark_suite.py exists.")
        sys.exit(1)

    print(f"Loaded {len(benchmarks)} benchmarks for LLM calibration")
    using_api = bool(os.environ.get("OPENAI_API_KEY"))
    print(f"Mode: {'Live LLM API' if using_api else 'Heuristic simulation (set OPENAI_API_KEY for live)'}")

    samples: List[CalibrationSample] = []

    for i, bench in enumerate(benchmarks):
        name = bench["name"]
        has_bug = bench.get("has_bug", False)
        code = bench["code"]

        predicts_bug, confidence = _try_llm_analysis(code)

        if predicts_bug is not None:
            correct = (predicts_bug == has_bug)
        else:
            correct = False

        sample = CalibrationSample(
            name=name,
            ground_truth_has_bug=has_bug,
            llm_predicts_bug=predicts_bug,
            llm_confidence=confidence,
            llm_correct=correct,
        )
        samples.append(sample)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(benchmarks)}] processed")

    # Compute calibration metrics
    ece, reliability_diagram = compute_ece(samples)

    # Overall accuracy
    valid_samples = [s for s in samples if s.llm_predicts_bug is not None]
    accuracy = sum(1 for s in valid_samples if s.llm_correct) / len(valid_samples) if valid_samples else 0.0

    # Brier score: mean((confidence - correct)^2)
    brier = 0.0
    for s in valid_samples:
        target = 1.0 if s.ground_truth_has_bug else 0.0
        pred_prob = s.llm_confidence if s.llm_predicts_bug else (1.0 - s.llm_confidence)
        brier += (pred_prob - target) ** 2
    brier = brier / len(valid_samples) if valid_samples else 0.0

    summary = {
        "experiment": "llm_calibration",
        "description": (
            "Expected Calibration Error (ECE) for the LLM component of the "
            "neuro-symbolic pipeline. Measures alignment between LLM self-reported "
            "confidence and actual prediction accuracy."
        ),
        "mode": "live_api" if using_api else "heuristic_simulation",
        "num_benchmarks": len(benchmarks),
        "num_valid_predictions": len(valid_samples),
        "metrics": {
            "ece": ece,
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier, 4),
        },
        "reliability_diagram": [asdict(b) for b in reliability_diagram],
        "interpretation": {
            "ece": (
                "ECE < 0.05 indicates well-calibrated; 0.05-0.15 is moderate; "
                "> 0.15 indicates poor calibration requiring recalibration."
            ),
            "brier_score": (
                "Brier score ranges 0-1; lower is better. Combines calibration "
                "and discrimination quality."
            ),
        },
        "per_sample": [asdict(s) for s in samples],
    }

    print(f"\n{'='*60}")
    print("LLM Calibration Results")
    print(f"{'='*60}")
    print(f"Samples: {len(valid_samples)}/{len(benchmarks)}")
    print(f"Overall accuracy: {accuracy:.3f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"\nReliability Diagram:")
    print(f"  {'Bin':>12} {'Count':>6} {'Avg Conf':>10} {'Avg Acc':>10} {'Gap':>8}")
    for b in reliability_diagram:
        label = f"[{b.bin_lower:.1f}, {b.bin_upper:.1f})"
        print(f"  {label:>12} {b.count:>6} {b.avg_confidence:>10.3f} {b.avg_accuracy:>10.3f} {b.gap:>8.3f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
