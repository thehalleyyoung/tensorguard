#!/usr/bin/env python3
"""Integrated statistical analysis applied to main evaluation headline results.

Addresses reviewer criticism: statistical rigor infrastructure (Brier decomposition,
PPV curves, B-H correction) was built but never applied to headline numbers.

This script loads ALL five main evaluation result files, extracts per-prediction
data, and applies:
  1. Brier decomposition (Murphy, 1973) to diagnose MCE=0.400 source
  2. Prevalence-conditioned PPV/NPV at π ∈ {0.01, 0.02, 0.05, 0.10, 0.20, 0.50}
  3. Benjamini-Hochberg FDR correction across all cross-suite p-values

Saves integrated results to integrated_statistical_results.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_rigor import (
    BrierDecomposition,
    brier_decomposition,
    ppv_npv_curve,
    compute_ppv,
    compute_npv,
    benjamini_hochberg,
    bonferroni,
    holm_bonferroni,
    familywise_error_probability,
    generate_report,
)
from src.calibration_analysis import (
    Prediction,
    compute_calibration_report,
    CONFIDENCE_MAP,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent

RESULT_FILES = {
    "cegar_ablation": "cegar_ablation_v5_results.json",
    "deep_composition": "deep_composition_benchmark_results.json",
    "external_bug": "external_bug_reproduction_results.json",
    "fx_torchvision": "fx_torchvision_benchmark_results.json",
    "neurosym_pipeline": "neurosym_pipeline_results.json",
}

TARGET_PREVALENCES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _conf_name_to_score(name: str) -> float:
    return CONFIDENCE_MAP.get(name.upper(), 0.5)


# ─── Per-suite data extraction ───────────────────────────────────────────────

def _extract_neurosym(data: Dict) -> Tuple[List[int], List[float], Dict]:
    """Extract y_true, y_prob and metrics from neurosym pipeline results."""
    y_true, y_prob = [], []
    for bm in data.get("benchmarks", []):
        has_bug = bm.get("has_bug")
        llm_predicts = bm.get("llm_predicts_bug")
        if has_bug is None or llm_predicts is None:
            continue
        conf_name = bm.get("pipeline_confidence", "")
        if conf_name and conf_name.upper() in CONFIDENCE_MAP:
            conf = _conf_name_to_score(conf_name)
        else:
            conf = bm.get("llm_confidence", 0.5)
        y_true.append(1 if has_bug else 0)
        y_prob.append(conf)
    metrics = data.get("pipeline_metrics", {})
    return y_true, y_prob, metrics


def _extract_cegar(data: Dict) -> Tuple[List[int], List[float], Dict]:
    """Extract from CEGAR ablation (use cegar_filtered config)."""
    y_true, y_prob = [], []
    cfg = data.get("configs", {}).get("cegar_filtered", {})
    for bm in cfg.get("per_benchmark", []):
        has_bug = bm.get("has_bug")
        detected = bm.get("detected_bug")
        if has_bug is None:
            continue
        y_true.append(1 if has_bug else 0)
        # CEGAR is deterministic: detected → high confidence, else low
        y_prob.append(0.95 if detected else 0.05)
    metrics = cfg.get("metrics", {})
    return y_true, y_prob, metrics


def _extract_external_bug(data: Dict) -> Tuple[List[int], List[float], Dict]:
    """Extract from external bug reproduction results."""
    y_true, y_prob = [], []
    for r in data.get("individual_results", []):
        is_buggy = r.get("is_buggy")
        detected = r.get("detected")
        if is_buggy is None:
            continue
        y_true.append(1 if is_buggy else 0)
        y_prob.append(0.90 if detected else 0.10)
    metrics = {
        k: data[k] for k in ("precision", "recall", "f1", "accuracy")
        if k in data
    }
    metrics.update({
        "tp": data.get("true_positives", 0),
        "fp": data.get("false_positives", 0),
        "fn": data.get("false_negatives", 0),
        "tn": data.get("true_negatives", 0),
    })
    return y_true, y_prob, metrics


def _extract_fx_torchvision(data: Dict) -> Tuple[List[int], List[float], Dict]:
    """Extract from FX/torchvision benchmark."""
    y_true, y_prob = [], []
    for r in data.get("results", []):
        expected_safe = r.get("expected_safe")
        actual_safe = r.get("actual_safe")
        if expected_safe is None or actual_safe is None:
            continue
        # Ground truth: bug present = not safe
        y_true.append(0 if expected_safe else 1)
        y_prob.append(0.05 if actual_safe else 0.95)
    correct = data.get("correct_verdict", 0)
    total = data.get("total", 1)
    metrics = {"accuracy": data.get("accuracy", correct / max(total, 1))}
    return y_true, y_prob, metrics


def _extract_deep_composition(data: Dict) -> Tuple[List[int], List[float], Dict]:
    """Extract from deep composition benchmark (TensorGuard arm)."""
    y_true, y_prob = [], []
    for r in data.get("tensorguard", {}).get("results", []):
        expected_safe = r.get("expected_safe")
        correct = r.get("correct")
        if expected_safe is None or correct is None:
            continue
        has_bug = not expected_safe
        y_true.append(1 if has_bug else 0)
        # TG: correct detection → aligned confidence
        if has_bug:
            y_prob.append(0.95 if correct else 0.10)
        else:
            y_prob.append(0.05 if correct else 0.90)
    tg = data.get("tensorguard", {})
    metrics = {"accuracy": tg.get("accuracy", 0.0), "correct": tg.get("correct", 0), "total": tg.get("total", 0)}
    return y_true, y_prob, metrics


EXTRACTORS = {
    "neurosym_pipeline": _extract_neurosym,
    "cegar_ablation": _extract_cegar,
    "external_bug": _extract_external_bug,
    "fx_torchvision": _extract_fx_torchvision,
    "deep_composition": _extract_deep_composition,
}


# ─── Sensitivity / Specificity from confusion matrix ─────────────────────────

def _sens_spec_from_yt_yp(y_true: List[int], y_prob: List[float],
                           threshold: float = 0.5) -> Tuple[float, float]:
    """Compute sensitivity and specificity from y_true/y_prob."""
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_prob):
        pred = 1 if yp >= threshold else 0
        if yt == 1 and pred == 1:
            tp += 1
        elif yt == 0 and pred == 1:
            fp += 1
        elif yt == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens, spec


# ─── Fisher exact p-value (pure Python) ─────────────────────────────────────

def _log_factorial(n: int) -> float:
    return sum(math.log(i) for i in range(1, n + 1))


def _fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact test p-value for 2×2 table [[a,b],[c,d]].

    Tests H_a: odds ratio > 1.  Uses the hypergeometric distribution.
    """
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1, c2 = a + c, b + d

    def _log_hyper(x: int) -> float:
        return (_log_factorial(r1) + _log_factorial(r2) +
                _log_factorial(c1) + _log_factorial(c2) -
                _log_factorial(n) - _log_factorial(x) -
                _log_factorial(r1 - x) - _log_factorial(c1 - x) -
                _log_factorial(r2 - c1 + x))

    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p_val = 0.0
    p_obs = math.exp(_log_hyper(a))
    for x in range(lo, hi + 1):
        px = math.exp(_log_hyper(x))
        if px <= p_obs + 1e-12:
            p_val += px
    return min(p_val, 1.0)


def _mcnemar_p(b: int, c: int) -> float:
    """McNemar mid-p for paired binary data (discordant counts b, c)."""
    if b + c == 0:
        return 1.0
    n = b + c
    x = min(b, c)
    # Exact binomial tail
    p = 0.0
    for k in range(x + 1):
        lp = (_log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)
              - n * math.log(2))
        p += math.exp(lp)
    return min(2.0 * p, 1.0)  # two-sided


# ─── Pairwise comparisons for p-value collection ────────────────────────────

def collect_comparison_pvalues(
    suite_data: Dict[str, Tuple[List[int], List[float], Dict]],
    all_data: Dict[str, Dict],
) -> List[Tuple[str, float]]:
    """Collect p-values from meaningful cross-suite comparisons."""
    comparisons: List[Tuple[str, float]] = []

    # 1. CEGAR filtered vs single-pass (within cegar_ablation)
    cegar_raw = all_data.get("cegar_ablation")
    if cegar_raw and "configs" in cegar_raw:
        sp = cegar_raw["configs"].get("single_pass", {}).get("metrics", {})
        cf = cegar_raw["configs"].get("cegar_filtered", {}).get("metrics", {})
        sp_tp, sp_fp = sp.get("tp", 0), sp.get("fp", 0)
        sp_fn, sp_tn = sp.get("fn", 0), sp.get("tn", 0)
        cf_tp, cf_fp = cf.get("tp", 0), cf.get("fp", 0)
        cf_fn, cf_tn = cf.get("fn", 0), cf.get("tn", 0)
        # Compare detection rates: McNemar-style using discordant pairs
        # CEGAR catches cf_tp - sp_tp more; single-pass catches sp_tp - cf_tp more
        disc_cegar = max(cf_tp - sp_tp, 0)
        disc_sp = max(sp_tp - cf_tp, 0)
        p = _mcnemar_p(disc_cegar, disc_sp)
        comparisons.append(("CEGAR_filtered_vs_single_pass", p))

    # 2. TensorGuard vs LLM on deep composition
    dc_raw = all_data.get("deep_composition")
    if dc_raw:
        tg_correct = dc_raw.get("tensorguard", {}).get("correct", 0)
        tg_total = dc_raw.get("tensorguard", {}).get("total", 1)
        llm_correct = dc_raw.get("llm", {}).get("correct", 0)
        llm_total = dc_raw.get("llm", {}).get("total", 1)
        tg_wrong = tg_total - tg_correct
        llm_wrong = llm_total - llm_correct
        p = _fisher_exact_p(tg_correct, tg_wrong, llm_correct, llm_wrong)
        comparisons.append(("TG_vs_LLM_deep_composition", p))

    # 3. Pipeline vs LLM-only on neurosym
    ns_raw = all_data.get("neurosym_pipeline")
    if ns_raw:
        pipe_m = ns_raw.get("pipeline_metrics", {})
        llm_m = ns_raw.get("llm_only_metrics", {})
        p_tp, p_fp = pipe_m.get("TP", 0), pipe_m.get("FP", 0)
        p_fn, p_tn = pipe_m.get("FN", 0), pipe_m.get("TN", 0)
        l_tp, l_fp = llm_m.get("TP", 0), llm_m.get("FP", 0)
        l_fn, l_tn = llm_m.get("FN", 0), llm_m.get("TN", 0)
        # Fisher exact on detection tables
        p_val = _fisher_exact_p(p_tp, p_fn, l_tp, l_fn)
        comparisons.append(("pipeline_vs_LLM_neurosym", p_val))

        # 4. Pipeline vs TG-only
        tg_m = ns_raw.get("tensorguard_only_metrics", {})
        t_tp, t_fn = tg_m.get("TP", 0), tg_m.get("FN", 0)
        p_val = _fisher_exact_p(p_tp, p_fn, t_tp, t_fn)
        comparisons.append(("pipeline_vs_TG_only_neurosym", p_val))

    # 5. Cross-suite precision comparisons (external vs neurosym)
    ext = suite_data.get("external_bug")
    ns = suite_data.get("neurosym_pipeline")
    if ext and ns:
        ext_yt, ext_yp, ext_m = ext
        ns_yt, ns_yp, ns_m = ns
        ext_prec = ext_m.get("precision", 0)
        ns_prec = ns_m.get("precision", 0)
        # Use Fisher on TP/FP counts
        ext_tp = ext_m.get("tp", 0)
        ext_fp = ext_m.get("fp", 0)
        ns_tp = ns_m.get("TP", ns_m.get("tp", 0))
        ns_fp = ns_m.get("FP", ns_m.get("fp", 0))
        if ext_tp + ext_fp > 0 and ns_tp + ns_fp > 0:
            p_val = _fisher_exact_p(ext_tp, ext_fp, ns_tp, ns_fp)
            comparisons.append(("external_vs_neurosym_precision", p_val))

    # 6. FX torchvision perfect accuracy vs external bug accuracy
    fx = suite_data.get("fx_torchvision")
    if fx and ext:
        fx_yt, fx_yp, fx_m = fx
        fx_correct = sum(1 for yt, yp in zip(fx_yt, fx_yp)
                         if (yt == 1) == (yp >= 0.5))
        fx_wrong = len(fx_yt) - fx_correct
        ext_correct = sum(1 for yt, yp in zip(ext_yt, ext_yp)
                          if (yt == 1) == (yp >= 0.5))
        ext_wrong = len(ext_yt) - ext_correct
        p_val = _fisher_exact_p(fx_correct, fx_wrong, ext_correct, ext_wrong)
        comparisons.append(("fx_vs_external_accuracy", p_val))

    return comparisons


# ─── Main analysis ───────────────────────────────────────────────────────────

def run_analysis() -> Dict[str, Any]:
    """Run the full integrated statistical analysis."""
    print("=" * 70)
    print("INTEGRATED STATISTICAL ANALYSIS")
    print("Applying Brier decomposition, PPV curves, B-H correction")
    print("to main evaluation headline results")
    print("=" * 70)

    # Load all result files
    all_raw: Dict[str, Dict] = {}
    for suite_key, fname in RESULT_FILES.items():
        path = EXPERIMENTS_DIR / fname
        data = _load_json(path)
        if data is not None:
            all_raw[suite_key] = data
            print(f"  ✓ Loaded {fname}")
        else:
            print(f"  ✗ Missing {fname}")

    if not all_raw:
        print("ERROR: No result files found. Cannot proceed.")
        return {}

    # Extract per-suite y_true / y_prob / metrics
    suite_data: Dict[str, Tuple[List[int], List[float], Dict]] = {}
    for suite_key, data in all_raw.items():
        extractor = EXTRACTORS.get(suite_key)
        if extractor is None:
            continue
        y_true, y_prob, metrics = extractor(data)
        if y_true:
            suite_data[suite_key] = (y_true, y_prob, metrics)
            print(f"  → {suite_key}: {len(y_true)} predictions extracted")

    # ── 1. Brier Decomposition per suite ──────────────────────────────────
    print("\n" + "─" * 70)
    print("1. BRIER SCORE DECOMPOSITION (Murphy, 1973)")
    print("─" * 70)

    brier_results: Dict[str, Dict] = {}
    combined_yt: List[int] = []
    combined_yp: List[float] = []

    for suite_key, (yt, yp, _) in suite_data.items():
        bd = brier_decomposition(yt, yp, n_bins=10)
        brier_results[suite_key] = {
            "brier_score": round(bd.brier_score, 6),
            "reliability_REL": round(bd.reliability, 6),
            "resolution_RES": round(bd.resolution, 6),
            "uncertainty_UNC": round(bd.uncertainty, 6),
            "identity_check": round(bd.reliability - bd.resolution + bd.uncertainty, 6),
            "n_predictions": len(yt),
            "n_bins": bd.n_bins,
            "bin_counts": bd.bin_counts,
            "bin_accuracies": [round(x, 4) for x in bd.bin_accuracies],
            "bin_mean_probs": [round(x, 4) for x in bd.bin_mean_probs],
        }
        combined_yt.extend(yt)
        combined_yp.extend(yp)
        print(f"\n  {suite_key} (n={len(yt)}):")
        print(f"    Brier = {bd.brier_score:.4f}")
        print(f"    REL (calibration)  = {bd.reliability:.4f}")
        print(f"    RES (discrimination) = {bd.resolution:.4f}")
        print(f"    UNC (irreducible)  = {bd.uncertainty:.4f}")
        print(f"    Identity: REL - RES + UNC = {bd.reliability - bd.resolution + bd.uncertainty:.4f}")

    # Combined across all suites
    if combined_yt:
        bd_all = brier_decomposition(combined_yt, combined_yp, n_bins=10)
        brier_results["_combined"] = {
            "brier_score": round(bd_all.brier_score, 6),
            "reliability_REL": round(bd_all.reliability, 6),
            "resolution_RES": round(bd_all.resolution, 6),
            "uncertainty_UNC": round(bd_all.uncertainty, 6),
            "n_predictions": len(combined_yt),
        }
        print(f"\n  COMBINED (n={len(combined_yt)}):")
        print(f"    Brier = {bd_all.brier_score:.4f}")
        print(f"    REL = {bd_all.reliability:.4f}, RES = {bd_all.resolution:.4f}, UNC = {bd_all.uncertainty:.4f}")

    # ── MCE diagnosis ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("MCE = 0.400 DIAGNOSIS")
    print("─" * 70)

    # Compute calibration report on combined data for MCE analysis
    predictions = []
    for suite_key, (yt, yp, _) in suite_data.items():
        for y, p in zip(yt, yp):
            predictions.append(Prediction(
                confidence=p,
                predicted_class=1 if p >= 0.5 else 0,
                true_class=y,
                label_name=suite_key,
            ))

    cal_report = compute_calibration_report(predictions, n_bins=10)
    mce_diagnosis: Dict[str, Any] = {
        "observed_MCE": round(cal_report.mce, 4),
        "observed_ECE": round(cal_report.ece, 4),
        "brier_score": round(cal_report.brier_score, 4),
        "calibration_component": round(cal_report.calibration_component, 6),
        "sharpness_component": round(cal_report.sharpness_component, 6),
        "uncertainty_component": round(cal_report.uncertainty_component, 6),
        "overconfidence_ratio": round(cal_report.overconfidence_ratio, 4),
        "mean_confidence": round(cal_report.mean_confidence, 4),
        "mean_accuracy": round(cal_report.mean_accuracy, 4),
    }

    if combined_yt:
        rel = bd_all.reliability
        res = bd_all.resolution
        unc = bd_all.uncertainty
        # Diagnosis: is the problem calibration (high REL) or discrimination (low RES)?
        if rel > res:
            diagnosis = ("CALIBRATION_FAILURE: REL ({:.4f}) > RES ({:.4f}). "
                         "The pipeline's probability estimates are poorly calibrated. "
                         "MCE stems primarily from miscalibrated confidence scores, "
                         "not from inability to discriminate bugs from non-bugs.").format(rel, res)
        elif res < unc * 0.5:
            diagnosis = ("DISCRIMINATION_FAILURE: RES ({:.4f}) < UNC/2 ({:.4f}). "
                         "The pipeline cannot sufficiently discriminate between "
                         "bug and non-bug cases.").format(res, unc * 0.5)
        else:
            diagnosis = ("MIXED: Both calibration (REL={:.4f}) and discrimination "
                         "(RES={:.4f}) contribute. REL/RES ratio = {:.2f}. "
                         "Recalibration (e.g., Platt scaling) would help if REL "
                         "is the dominant term.").format(rel, res, rel / max(res, 1e-9))

        mce_diagnosis["diagnosis"] = diagnosis
        mce_diagnosis["rel_over_res_ratio"] = round(rel / max(res, 1e-9), 4)
        print(f"\n  {diagnosis}")

    # ── 2. PPV/NPV at target prevalences ──────────────────────────────────
    print("\n" + "─" * 70)
    print("2. PREVALENCE-CONDITIONED PPV/NPV CURVES")
    print("─" * 70)

    ppv_results: Dict[str, Dict] = {}
    for suite_key, (yt, yp, metrics) in suite_data.items():
        sens, spec = _sens_spec_from_yt_yp(yt, yp)
        ppv_at_prev: Dict[str, Dict[str, float]] = {}
        for pi in TARGET_PREVALENCES:
            ppv_val = compute_ppv(sens, spec, pi)
            npv_val = compute_npv(sens, spec, pi)
            ppv_at_prev[f"pi={pi:.2f}"] = {
                "PPV": round(ppv_val, 4),
                "NPV": round(npv_val, 4),
            }

        curve = ppv_npv_curve(sens, spec, prevalence_range=(0.01, 0.50), n_steps=50)
        ppv_results[suite_key] = {
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "breakeven_prevalence": round(curve.breakeven_prevalence, 4) if curve.breakeven_prevalence else None,
            "ppv_npv_at_target_prevalences": ppv_at_prev,
        }
        print(f"\n  {suite_key}: sens={sens:.3f}, spec={spec:.3f}")
        print(f"    Breakeven prevalence (PPV≥0.5): {curve.breakeven_prevalence}")
        for pi_label, vals in ppv_at_prev.items():
            print(f"    {pi_label}: PPV={vals['PPV']:.4f}, NPV={vals['NPV']:.4f}")

    # ── 3. Benjamini-Hochberg FDR Correction ──────────────────────────────
    print("\n" + "─" * 70)
    print("3. BENJAMINI-HOCHBERG FDR CORRECTION")
    print("─" * 70)

    comparisons = collect_comparison_pvalues(suite_data, all_raw)
    if comparisons:
        labels, raw_pvals = zip(*comparisons)
        labels = list(labels)
        raw_pvals = list(raw_pvals)

        bh = benjamini_hochberg(raw_pvals, alpha=0.05)
        bonf = bonferroni(raw_pvals, alpha=0.05)
        holm = holm_bonferroni(raw_pvals, alpha=0.05)
        fwer = familywise_error_probability(len(raw_pvals), alpha=0.05)

        print(f"\n  {len(comparisons)} comparisons, FWER (uncorrected) = {fwer:.4f}")
        print(f"  B-H rejected: {bh.n_rejected}/{bh.n_tests}")
        print(f"  Bonferroni rejected: {bonf.n_rejected}/{bonf.n_tests}")
        print(f"  Holm rejected: {holm.n_rejected}/{holm.n_tests}")

        comparison_table: List[Dict[str, Any]] = []
        for i, label in enumerate(labels):
            entry = {
                "comparison": label,
                "raw_p": round(raw_pvals[i], 6),
                "bh_adjusted_p": round(bh.adjusted_p_values[i], 6),
                "bh_rejected": bh.rejected[i],
                "bonferroni_adjusted_p": round(bonf.adjusted_p_values[i], 6),
                "bonferroni_rejected": bonf.rejected[i],
                "holm_adjusted_p": round(holm.adjusted_p_values[i], 6),
                "holm_rejected": holm.rejected[i],
            }
            comparison_table.append(entry)
            sig = "***" if bh.rejected[i] else "n.s."
            print(f"    {label}: raw={raw_pvals[i]:.4f} → BH={bh.adjusted_p_values[i]:.4f} {sig}")

        mc_results = {
            "n_tests": len(comparisons),
            "fdr_level": 0.05,
            "fwer_uncorrected": round(fwer, 4),
            "bh_n_rejected": bh.n_rejected,
            "bonferroni_n_rejected": bonf.n_rejected,
            "holm_n_rejected": holm.n_rejected,
            "comparisons": comparison_table,
        }
    else:
        mc_results = {"n_tests": 0, "note": "No comparisons available"}
        print("  No comparisons available.")

    # ── 4. Generate StatisticalReport using generate_report() ─────────────
    print("\n" + "─" * 70)
    print("4. COMPREHENSIVE STATISTICAL REPORT (via generate_report)")
    print("─" * 70)

    report_input = {
        "y_true": combined_yt,
        "y_prob": combined_yp,
        "p_values": [c[1] for c in comparisons] if comparisons else [],
        "n_bins": 10,
        "alpha": 0.05,
        "analysis": "integrated_headline_results",
        "n_suites": len(suite_data),
        "suites": list(suite_data.keys()),
    }

    if combined_yt:
        sens_all, spec_all = _sens_spec_from_yt_yp(combined_yt, combined_yp)
        report_input["sensitivity"] = sens_all
        report_input["specificity"] = spec_all

    report = generate_report(report_input)
    report_json = json.loads(report.to_json())
    print(f"  Report generated with {len(combined_yt)} total predictions across {len(suite_data)} suites")
    if report.brier:
        print(f"  Combined Brier: {report.brier.brier_score:.4f}")
    if report.ppv_npv:
        print(f"  Combined sens={report.ppv_npv.sensitivity:.3f}, spec={report.ppv_npv.specificity:.3f}")
    if report.multiple_comparison:
        print(f"  B-H: {report.multiple_comparison.n_rejected}/{report.multiple_comparison.n_tests} rejected")

    # ── Assemble final output ─────────────────────────────────────────────
    output = {
        "description": (
            "Integrated statistical analysis applying Brier decomposition, "
            "prevalence-conditioned PPV/NPV, and Benjamini-Hochberg FDR correction "
            "to the five main evaluation headline results. Addresses reviewer "
            "criticism that statistical rigor infrastructure was built but not applied."
        ),
        "suites_analyzed": list(suite_data.keys()),
        "total_predictions": len(combined_yt),
        "brier_decomposition_per_suite": brier_results,
        "mce_diagnosis": mce_diagnosis,
        "ppv_npv_curves": ppv_results,
        "multiple_comparison_correction": mc_results,
        "comprehensive_report": report_json,
        "latex_tables": report.to_latex_tables(),
    }

    return output


def main():
    output = run_analysis()
    if not output:
        sys.exit(1)

    out_path = EXPERIMENTS_DIR / "integrated_statistical_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved to {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
