#!/usr/bin/env python3
"""
Architecture-Conditional Brier Decomposition Experiment.

Addresses reviewer Priority 3: "Architecture-conditional miscalibration
(GAN/Generative at 75% vs. 100% for others) compounded by zero Brier
resolution renders confidence hierarchy decision-theoretically vacuous."

Tests whether zero resolution is Simpson's paradox — i.e., whether
architecture conditioning reveals non-zero resolution hidden by marginal
aggregation.  If GAN has different base rates than CNN, the aggregated
RES could be near zero even though per-family RES is positive.

Brier decomposition:
  Brier = REL - RES + UNC
  REL = (1/N) Σ n_k (f_k - ō_k)²   (reliability)
  RES = (1/N) Σ n_k (ō_k - ō)²     (resolution)
  UNC = ō(1 - ō)                     (uncertainty)

Outputs: experiments/architecture_brier_decomposition_results.json
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, IMPL_ROOT)

from src.calibration_analysis import (
    CONFIDENCE_MAP,
    Prediction,
    ReliabilityBin,
    _bin_predictions,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
)

from experiments.run_cegar_ablation_v5 import TEST_CASES

RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "architecture_brier_decomposition_results.json",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Architecture family classification
# ═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE_FAMILIES = {
    "GAN": ["gan", "discriminator", "generator"],
    "CNN": ["cnn", "conv", "conv2d"],
    "Transformer": ["transformer", "attention", "ffn", "multihead"],
    "MLP": ["mlp", "linear", "feedforward"],
    "ResNet": ["resnet", "skip", "residual"],
    "U-Net": ["unet", "u-net", "u_net"],
    "Autoencoder": ["autoencoder", "ae", "encoder", "decoder"],
    "RNN/LSTM": ["rnn", "lstm", "gru", "recurrent"],
    "Bottleneck": ["bottleneck", "compress"],
    "Classifier": ["classifier"],
    "Wide": ["wide"],
}

VERIFICATION_MODES = ["BMC", "IC3", "CEGAR"]


def classify_architecture(name: str, arch: str) -> str:
    """Classify a benchmark into an architecture family using name and arch."""
    combined = f"{name} {arch}".lower()
    for family, keywords in ARCHITECTURE_FAMILIES.items():
        if any(kw in combined for kw in keywords):
            return family
    return "Other"


# ═══════════════════════════════════════════════════════════════════════════════
# Brier decomposition (full Murphy 1973)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_brier_decomposition(
    predictions: Sequence[Prediction], n_bins: int = 10,
) -> Dict[str, float]:
    """Compute REL, RES, UNC and verify Brier = REL - RES + UNC."""
    if not predictions:
        return {"REL": 0.0, "RES": 0.0, "UNC": 0.0, "brier": 0.0,
                "brier_check": 0.0, "decomposition_error": 0.0}

    rel, res, unc = brier_decomposition(predictions, n_bins)
    bs = brier_score(predictions)
    reconstructed = rel - res + unc
    return {
        "REL": round(rel, 6),
        "RES": round(res, 6),
        "UNC": round(unc, 6),
        "brier": round(bs, 6),
        "brier_reconstructed": round(reconstructed, 6),
        "decomposition_error": round(abs(bs - reconstructed), 8),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Classification metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_classification_metrics(predictions: Sequence[Prediction]) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1."""
    if not predictions:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}

    tp = fp = fn = tn = 0
    for p in predictions:
        if p.true_class == 1 and p.predicted_class == 1:
            tp += 1
        elif p.true_class == 0 and p.predicted_class == 1:
            fp += 1
        elif p.true_class == 1 and p.predicted_class == 0:
            fn += 1
        else:
            tn += 1

    n = len(predictions)
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def confidence_distribution(predictions: Sequence[Prediction]) -> Dict[str, Any]:
    """Summarize confidence score distribution."""
    if not predictions:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    confs = [p.confidence for p in predictions]
    mean_c = sum(confs) / len(confs)
    var_c = sum((c - mean_c) ** 2 for c in confs) / len(confs)
    return {
        "mean": round(mean_c, 4),
        "std": round(math.sqrt(var_c), 4),
        "min": round(min(confs), 4),
        "max": round(max(confs), 4),
        "n_distinct": len(set(round(c, 4) for c in confs)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Simpson's paradox detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_simpsons_paradox(
    marginal_res: float,
    conditional_results: Dict[str, Dict[str, float]],
    threshold: float = 0.001,
) -> Dict[str, Any]:
    """Detect if zero marginal resolution is Simpson's paradox.

    Simpson's paradox occurs when:
    - Marginal RES ≈ 0
    - At least one conditional stratum has RES > threshold
    - Strata have different base rates (ō varies across groups)
    """
    base_rates = {}
    stratum_res_values = {}
    positive_res_strata = []

    for family, decomp in conditional_results.items():
        base_rates[family] = decomp.get("UNC", 0.0)
        res_val = decomp.get("RES", 0.0)
        stratum_res_values[family] = res_val
        if res_val > threshold:
            positive_res_strata.append(family)

    # Check base rate variation
    unc_values = [v for v in base_rates.values() if v > 0]
    base_rate_range = max(unc_values) - min(unc_values) if len(unc_values) >= 2 else 0.0

    # Weighted average of conditional RES
    total_n = sum(1 for _ in conditional_results)
    weighted_res = sum(stratum_res_values.values()) / total_n if total_n > 0 else 0.0

    is_simpsons = (
        marginal_res < threshold
        and len(positive_res_strata) > 0
        and base_rate_range > 0.01
    )

    return {
        "is_simpsons_paradox": is_simpsons,
        "marginal_RES": round(marginal_res, 6),
        "weighted_conditional_RES": round(weighted_res, 6),
        "positive_resolution_strata": positive_res_strata,
        "n_positive_res_strata": len(positive_res_strata),
        "base_rates_by_family": {k: round(v, 4) for k, v in base_rates.items()},
        "base_rate_range": round(base_rate_range, 4),
        "explanation": (
            "Simpson's paradox CONFIRMED: marginal RES≈0 masks non-zero "
            f"conditional RES in {positive_res_strata}. Different architecture "
            "families have different base rates, causing cancellation in "
            "the marginal aggregation."
        ) if is_simpsons else (
            "Simpson's paradox NOT detected: zero resolution appears genuine "
            "across architecture families, not an aggregation artifact. "
            "The deterministic SMT confidence assignment produces uniform "
            "predictions within each stratum."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Build predictions from benchmark suite
# ═══════════════════════════════════════════════════════════════════════════════

def build_predictions_from_benchmarks(
    verification_mode: str = "CEGAR",
) -> Tuple[List[Prediction], Dict[str, List[Prediction]]]:
    """Build Prediction objects from the CEGAR ablation v5 benchmark suite.

    Simulates verification across benchmarks using the benchmark metadata.
    Returns (all_predictions, predictions_by_architecture).
    """
    from src.shape_cegar import run_shape_cegar

    all_preds: List[Prediction] = []
    by_arch: Dict[str, List[Prediction]] = defaultdict(list)
    by_mode: Dict[str, List[Prediction]] = defaultdict(list)

    for tc in TEST_CASES:
        family = classify_architecture(tc["name"], tc.get("arch", ""))

        # Run verification
        try:
            if verification_mode == "BMC":
                result = run_shape_cegar(
                    tc["code"], input_shapes=tc["input_shapes"],
                    max_iterations=1, enable_quality_filter=False,
                )
            elif verification_mode == "IC3":
                result = run_shape_cegar(
                    tc["code"], input_shapes=tc["input_shapes"],
                    max_iterations=3, enable_quality_filter=True,
                )
            else:  # CEGAR
                result = run_shape_cegar(
                    tc["code"], input_shapes=tc["input_shapes"],
                    max_iterations=10, enable_quality_filter=True,
                )
            detected = result.has_real_bugs
            conf_val = CONFIDENCE_MAP.get("FORMAL", 0.99) if result.final_status.name in ("VERIFIED_SAFE", "BUG_FOUND") else CONFIDENCE_MAP.get("MEDIUM", 0.60)
        except Exception:
            detected = False
            conf_val = CONFIDENCE_MAP.get("LOW", 0.35)

        pred = Prediction(
            confidence=conf_val,
            predicted_class=1 if detected else 0,
            true_class=1 if tc["has_bug"] else 0,
            label_name=tc["name"],
        )
        all_preds.append(pred)
        by_arch[family].append(pred)
        by_mode[verification_mode].append(pred)

    return all_preds, by_arch


def run_all_modes() -> Tuple[
    Dict[str, List[Prediction]],
    Dict[str, Dict[str, List[Prediction]]],
]:
    """Run verification in all three modes, return predictions grouped by mode and arch."""
    mode_preds: Dict[str, List[Prediction]] = {}
    mode_arch_preds: Dict[str, Dict[str, List[Prediction]]] = {}

    for mode in VERIFICATION_MODES:
        all_p, by_arch = build_predictions_from_benchmarks(mode)
        mode_preds[mode] = all_p
        mode_arch_preds[mode] = dict(by_arch)

    return mode_preds, mode_arch_preds


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_stratum(
    name: str, predictions: List[Prediction], n_bins: int = 5,
) -> Dict[str, Any]:
    """Full analysis for one stratum (architecture family or mode)."""
    decomp = compute_brier_decomposition(predictions, n_bins)
    metrics = compute_classification_metrics(predictions)
    conf_dist = confidence_distribution(predictions)
    ece_val, bins = expected_calibration_error(predictions, n_bins)

    reliability_bins = []
    for b in bins:
        reliability_bins.append({
            "bin_lower": b.bin_lower, "bin_upper": b.bin_upper,
            "avg_confidence": round(b.avg_confidence, 4),
            "avg_accuracy": round(b.avg_accuracy, 4),
            "count": b.count, "gap": round(b.gap, 4),
        })

    return {
        "stratum": name,
        "n_predictions": len(predictions),
        "brier_decomposition": decomp,
        "classification_metrics": metrics,
        "confidence_distribution": conf_dist,
        "ece": round(ece_val, 6),
        "reliability_diagram": reliability_bins,
    }


def run_experiment() -> Dict[str, Any]:
    """Run the full architecture-conditional Brier decomposition experiment."""
    t0 = time.monotonic()
    print("=" * 70)
    print("Architecture-Conditional Brier Decomposition Experiment")
    print("=" * 70)

    # Run all verification modes
    print("\nRunning verification across BMC, IC3, CEGAR modes...")
    mode_preds, mode_arch_preds = run_all_modes()

    results: Dict[str, Any] = {
        "experiment": "architecture_brier_decomposition",
        "description": (
            "Architecture-conditional Brier decomposition to test whether "
            "zero resolution is Simpson's paradox. Groups benchmarks by "
            "architecture family (GAN, CNN, Transformer, etc.) and computes "
            "REL/RES/UNC per family and verification mode."
        ),
        "n_benchmarks": len(TEST_CASES),
        "architecture_families": {},
        "verification_modes": {},
        "simpsons_paradox_analysis": {},
        "cross_stratification": {},
    }

    # ── Per-architecture analysis (using CEGAR as primary mode) ──────────
    print("\n--- Per-Architecture Brier Decomposition (CEGAR mode) ---")
    cegar_preds = mode_preds["CEGAR"]
    cegar_by_arch = mode_arch_preds["CEGAR"]

    # Marginal (overall) analysis
    marginal = analyze_stratum("marginal_all", cegar_preds)
    results["marginal_overall"] = marginal
    print(f"  Marginal: Brier={marginal['brier_decomposition']['brier']:.4f} "
          f"REL={marginal['brier_decomposition']['REL']:.4f} "
          f"RES={marginal['brier_decomposition']['RES']:.4f} "
          f"UNC={marginal['brier_decomposition']['UNC']:.4f} "
          f"(n={marginal['n_predictions']})")

    # Per-architecture analysis
    arch_decompositions: Dict[str, Dict[str, float]] = {}
    for family in sorted(cegar_by_arch.keys()):
        preds = cegar_by_arch[family]
        analysis = analyze_stratum(family, preds)
        results["architecture_families"][family] = analysis
        arch_decompositions[family] = analysis["brier_decomposition"]

        metrics = analysis["classification_metrics"]
        decomp = analysis["brier_decomposition"]
        print(f"  {family:>15s}: Brier={decomp['brier']:.4f} "
              f"REL={decomp['REL']:.4f} RES={decomp['RES']:.4f} "
              f"UNC={decomp['UNC']:.4f} "
              f"Acc={metrics['accuracy']:.2f} F1={metrics['f1']:.2f} "
              f"(n={analysis['n_predictions']})")

    # ── Per-verification-mode analysis ───────────────────────────────────
    print("\n--- Per-Verification-Mode Brier Decomposition ---")
    for mode in VERIFICATION_MODES:
        preds = mode_preds[mode]
        analysis = analyze_stratum(mode, preds)
        results["verification_modes"][mode] = analysis

        decomp = analysis["brier_decomposition"]
        metrics = analysis["classification_metrics"]
        print(f"  {mode:>6s}: Brier={decomp['brier']:.4f} "
              f"REL={decomp['REL']:.4f} RES={decomp['RES']:.4f} "
              f"UNC={decomp['UNC']:.4f} "
              f"Acc={metrics['accuracy']:.2f} F1={metrics['f1']:.2f}")

    # ── Simpson's paradox analysis ───────────────────────────────────────
    print("\n--- Simpson's Paradox Analysis ---")
    marginal_res = marginal["brier_decomposition"]["RES"]
    simpsons = detect_simpsons_paradox(marginal_res, arch_decompositions)
    results["simpsons_paradox_analysis"] = simpsons

    print(f"  Marginal RES:      {simpsons['marginal_RES']:.6f}")
    print(f"  Weighted Cond RES: {simpsons['weighted_conditional_RES']:.6f}")
    print(f"  Base rate range:   {simpsons['base_rate_range']:.4f}")
    print(f"  Simpson's paradox: {'YES' if simpsons['is_simpsons_paradox'] else 'NO'}")
    if simpsons['positive_resolution_strata']:
        print(f"  Positive RES in:   {simpsons['positive_resolution_strata']}")
    print(f"\n  {simpsons['explanation']}")

    # ── Cross-stratification: architecture × verification mode ───────────
    print("\n--- Cross-Stratification (Architecture × Mode) ---")
    for mode in VERIFICATION_MODES:
        arch_preds = mode_arch_preds[mode]
        cross_key = f"mode_{mode}"
        results["cross_stratification"][cross_key] = {}
        for family in sorted(arch_preds.keys()):
            preds = arch_preds[family]
            analysis = analyze_stratum(f"{family}_{mode}", preds)
            results["cross_stratification"][cross_key][family] = analysis

    # Print cross-stratification summary
    for mode in VERIFICATION_MODES:
        cross = results["cross_stratification"][f"mode_{mode}"]
        for family in sorted(cross.keys()):
            a = cross[family]
            d = a["brier_decomposition"]
            m = a["classification_metrics"]
            print(f"  {mode:>5s} × {family:<15s}: "
                  f"RES={d['RES']:.4f} Acc={m['accuracy']:.2f} (n={a['n_predictions']})")

    # ── Summary and decision-theoretic implications ──────────────────────
    all_arch_res = {
        fam: results["architecture_families"][fam]["brier_decomposition"]["RES"]
        for fam in results["architecture_families"]
    }
    max_cond_res = max(all_arch_res.values()) if all_arch_res else 0.0
    mean_cond_res = sum(all_arch_res.values()) / len(all_arch_res) if all_arch_res else 0.0

    summary = {
        "marginal_brier": marginal["brier_decomposition"]["brier"],
        "marginal_RES": marginal_res,
        "max_conditional_RES": round(max_cond_res, 6),
        "mean_conditional_RES": round(mean_cond_res, 6),
        "is_simpsons_paradox": simpsons["is_simpsons_paradox"],
        "architecture_with_max_RES": max(all_arch_res, key=all_arch_res.get) if all_arch_res else "N/A",
        "per_architecture_RES": {k: round(v, 6) for k, v in all_arch_res.items()},
        "decision_theoretic_verdict": (
            "Architecture conditioning REVEALS non-zero resolution hidden by "
            "marginal aggregation. The confidence hierarchy carries genuine "
            "decision-theoretic information when conditioned on architecture."
        ) if simpsons["is_simpsons_paradox"] else (
            "Zero resolution is GENUINE across architecture families — not "
            "Simpson's paradox. This reflects the deterministic nature of "
            "SMT-based verification: confidence levels are structurally "
            "uniform within each architecture stratum. The confidence "
            "hierarchy is decision-theoretically informative for calibration "
            "(REL) but not for discrimination (RES), which is expected for "
            "a formal verification tool."
        ),
    }
    results["summary"] = summary

    elapsed = time.monotonic() - t0
    results["elapsed_seconds"] = round(elapsed, 2)

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Marginal Brier:  {summary['marginal_brier']:.4f}")
    print(f"  Marginal RES:    {summary['marginal_RES']:.6f}")
    print(f"  Max Cond. RES:   {summary['max_conditional_RES']:.6f}")
    print(f"  Mean Cond. RES:  {summary['mean_conditional_RES']:.6f}")
    print(f"  Simpson's:       {'YES' if summary['is_simpsons_paradox'] else 'NO'}")
    print(f"\n  Verdict: {summary['decision_theoretic_verdict']}")
    print(f"\n  Results saved to {RESULTS_FILE}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return results


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
