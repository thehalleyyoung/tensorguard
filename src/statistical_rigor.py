"""Comprehensive statistical methodology for TensorGuard evaluation.

Addresses three gaps:
1. Brier score decomposition (Murphy, 1973) into calibration, refinement, uncertainty
2. Prevalence-conditioned PPV/NPV curves for deployment decisions
3. Benjamini-Hochberg FDR correction for multiple comparisons
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ─── Brier Score Decomposition (Murphy 1973) ─────────────────────────────────

@dataclass
class BrierDecomposition:
    """Full Murphy (1973) decomposition of the Brier score.

    Identity: brier_score = reliability - resolution + uncertainty
    """
    brier_score: float
    reliability: float      # calibration component – lower is better
    resolution: float       # refinement component – higher is better
    uncertainty: float       # base-rate uncertainty – irreducible
    n_bins: int
    bin_counts: List[int]
    bin_accuracies: List[float]
    bin_mean_probs: List[float]

    def reliability_diagram_data(self) -> Dict[str, List[float]]:
        """Return data suitable for plotting a reliability diagram."""
        n = self.n_bins
        bin_edges = [i / n for i in range(n + 1)]
        return {
            "bin_edges": bin_edges,
            "bin_accuracies": list(self.bin_accuracies),
            "bin_mean_probs": list(self.bin_mean_probs),
            "bin_counts": list(self.bin_counts),
        }


def brier_decomposition(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> BrierDecomposition:
    """Decompose Brier score into reliability, resolution, and uncertainty.

    Uses equal-width binning following Murphy (1973):
      Brier = reliability − resolution + uncertainty
    where
      reliability = (1/N) Σ_k n_k (ō_k − f̄_k)²   (calibration)
      resolution  = (1/N) Σ_k n_k (ō_k − ō)²       (refinement)
      uncertainty = ō(1 − ō)                         (base rate)

    Parameters
    ----------
    y_true : sequence of int
        Binary ground-truth labels (0 or 1).
    y_prob : sequence of float
        Predicted probabilities for the positive class.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    BrierDecomposition
    """
    n = len(y_true)
    if n == 0:
        return BrierDecomposition(
            brier_score=0.0, reliability=0.0, resolution=0.0,
            uncertainty=0.0, n_bins=n_bins,
            bin_counts=[], bin_accuracies=[], bin_mean_probs=[],
        )

    if len(y_prob) != n:
        raise ValueError("y_true and y_prob must have the same length")

    # Overall Brier score
    bs = sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / n

    # Overall mean outcome (base rate)
    o_bar = sum(y_true) / n

    # Bin predictions
    bin_sums_prob: List[float] = [0.0] * n_bins
    bin_sums_outcome: List[float] = [0.0] * n_bins
    bin_counts: List[int] = [0] * n_bins

    for y, p in zip(y_true, y_prob):
        idx = min(int(p * n_bins), n_bins - 1)
        bin_counts[idx] += 1
        bin_sums_prob[idx] += p
        bin_sums_outcome[idx] += y

    bin_accuracies: List[float] = []
    bin_mean_probs: List[float] = []

    for k in range(n_bins):
        if bin_counts[k] > 0:
            bin_accuracies.append(bin_sums_outcome[k] / bin_counts[k])
            bin_mean_probs.append(bin_sums_prob[k] / bin_counts[k])
        else:
            bin_accuracies.append(0.0)
            bin_mean_probs.append(0.0)

    # Compute components
    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        if bin_counts[k] == 0:
            continue
        reliability += bin_counts[k] * (bin_accuracies[k] - bin_mean_probs[k]) ** 2
        resolution += bin_counts[k] * (bin_accuracies[k] - o_bar) ** 2

    reliability /= n
    resolution /= n
    uncertainty = o_bar * (1.0 - o_bar)

    return BrierDecomposition(
        brier_score=bs,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        n_bins=n_bins,
        bin_counts=bin_counts,
        bin_accuracies=bin_accuracies,
        bin_mean_probs=bin_mean_probs,
    )


# ─── Prevalence-Conditioned PPV / NPV ────────────────────────────────────────

def compute_ppv(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Positive predictive value via Bayes' theorem.

    PPV = (sens × prev) / (sens × prev + (1 − spec) × (1 − prev))
    """
    numerator = sensitivity * prevalence
    denominator = numerator + (1.0 - specificity) * (1.0 - prevalence)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_npv(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Negative predictive value via Bayes' theorem.

    NPV = (spec × (1 − prev)) / (spec × (1 − prev) + (1 − sens) × prev)
    """
    numerator = specificity * (1.0 - prevalence)
    denominator = numerator + (1.0 - sensitivity) * prevalence
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


@dataclass
class PPVNPVCurve:
    """PPV/NPV as functions of prevalence."""
    prevalences: List[float]
    ppv_values: List[float]
    npv_values: List[float]
    breakeven_prevalence: Optional[float]  # prevalence where PPV ≥ threshold
    sensitivity: float
    specificity: float


def ppv_npv_curve(
    sensitivity: float,
    specificity: float,
    prevalence_range: Tuple[float, float] = (0.01, 0.50),
    n_steps: int = 50,
    ppv_threshold: float = 0.5,
) -> PPVNPVCurve:
    """Generate PPV/NPV curves over a prevalence range.

    Parameters
    ----------
    sensitivity : float
        True positive rate.
    specificity : float
        True negative rate.
    prevalence_range : tuple
        (min_prevalence, max_prevalence).
    n_steps : int
        Number of points to evaluate.
    ppv_threshold : float
        The threshold for computing breakeven prevalence (PPV ≥ threshold).

    Returns
    -------
    PPVNPVCurve
    """
    lo, hi = prevalence_range
    if n_steps < 2:
        n_steps = 2
    step = (hi - lo) / (n_steps - 1)

    prevalences: List[float] = []
    ppv_values: List[float] = []
    npv_values: List[float] = []
    breakeven: Optional[float] = None

    for i in range(n_steps):
        prev = lo + i * step
        prevalences.append(prev)
        ppv = compute_ppv(sensitivity, specificity, prev)
        npv = compute_npv(sensitivity, specificity, prev)
        ppv_values.append(ppv)
        npv_values.append(npv)
        if breakeven is None and ppv >= ppv_threshold:
            breakeven = prev

    return PPVNPVCurve(
        prevalences=prevalences,
        ppv_values=ppv_values,
        npv_values=npv_values,
        breakeven_prevalence=breakeven,
        sensitivity=sensitivity,
        specificity=specificity,
    )


# ─── Multiple Comparison Correction ──────────────────────────────────────────

@dataclass
class BHResult:
    """Result of Benjamini-Hochberg FDR correction."""
    adjusted_p_values: List[float]
    rejected: List[bool]
    n_rejected: int
    n_tests: int
    fdr_level: float


@dataclass
class BonferroniResult:
    """Result of Bonferroni correction."""
    adjusted_p_values: List[float]
    rejected: List[bool]
    n_rejected: int
    n_tests: int
    alpha: float


@dataclass
class HolmResult:
    """Result of Holm-Bonferroni step-down correction."""
    adjusted_p_values: List[float]
    rejected: List[bool]
    n_rejected: int
    n_tests: int
    alpha: float


def benjamini_hochberg(
    p_values: List[float], alpha: float = 0.05
) -> BHResult:
    """Benjamini-Hochberg procedure for FDR control.

    Parameters
    ----------
    p_values : list of float
        Raw p-values from statistical tests.
    alpha : float
        Desired false discovery rate.

    Returns
    -------
    BHResult
    """
    m = len(p_values)
    if m == 0:
        return BHResult([], [], 0, 0, alpha)

    # Sort p-values, keep original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Compute adjusted p-values (step-up)
    adjusted = [0.0] * m
    # Start from the largest p-value
    prev_adj = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1  # 1-based rank
        adj = p * m / rank
        adj = min(adj, prev_adj)  # enforce monotonicity
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev_adj = adj

    rejected = [adj <= alpha for adj in adjusted]
    n_rejected = sum(rejected)

    return BHResult(
        adjusted_p_values=adjusted,
        rejected=rejected,
        n_rejected=n_rejected,
        n_tests=m,
        fdr_level=alpha,
    )


def bonferroni(
    p_values: List[float], alpha: float = 0.05
) -> BonferroniResult:
    """Bonferroni correction for FWER control.

    Adjusted p_i = min(p_i × m, 1.0)
    """
    m = len(p_values)
    if m == 0:
        return BonferroniResult([], [], 0, 0, alpha)

    adjusted = [min(p * m, 1.0) for p in p_values]
    rejected = [adj <= alpha for adj in adjusted]
    return BonferroniResult(
        adjusted_p_values=adjusted,
        rejected=rejected,
        n_rejected=sum(rejected),
        n_tests=m,
        alpha=alpha,
    )


def holm_bonferroni(
    p_values: List[float], alpha: float = 0.05
) -> HolmResult:
    """Holm-Bonferroni step-down procedure for FWER control.

    More powerful than Bonferroni while still controlling FWER.
    """
    m = len(p_values)
    if m == 0:
        return HolmResult([], [], 0, 0, alpha)

    # Sort p-values keeping original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    prev_adj = 0.0
    for rank_idx, (orig_idx, p) in enumerate(indexed):
        # Holm adjusted: p * (m - rank_idx)
        adj = p * (m - rank_idx)
        adj = max(adj, prev_adj)  # enforce monotonicity (non-decreasing)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev_adj = adj

    rejected = [adj <= alpha for adj in adjusted]
    return HolmResult(
        adjusted_p_values=adjusted,
        rejected=rejected,
        n_rejected=sum(rejected),
        n_tests=m,
        alpha=alpha,
    )


def familywise_error_probability(n_tests: int, alpha: float = 0.05) -> float:
    """Probability of ≥ 1 false positive under independence.

    FWER = 1 − (1 − α)^n
    """
    if n_tests <= 0:
        return 0.0
    return 1.0 - (1.0 - alpha) ** n_tests


# ─── Paired classifier comparison (McNemar + bootstrap) ──────────────────────

@dataclass
class McNemarResult:
    """Result of an exact (binomial) McNemar test for two paired classifiers.

    ``b`` is the number of items on which classifier A is correct and B is
    wrong; ``c`` the number on which A is wrong and B is correct.  Concordant
    pairs (both right / both wrong) carry no information about the *difference*
    and are excluded, which is exactly what McNemar's test conditions on.
    """
    n_discordant: int          # b + c
    b: int                     # A correct, B wrong
    c: int                     # A wrong, B correct
    statistic: int             # min(b, c) (the exact-test statistic)
    p_value: float             # two-sided exact binomial p-value
    odds_ratio: Optional[float]  # b / c (None if c == 0)


def mcnemar_exact_test(b: int, c: int) -> McNemarResult:
    """Exact two-sided McNemar test on the discordant counts ``b`` and ``c``.

    Under H0 (the two classifiers are equally likely to be the one that is
    right on a discordant pair) ``b ~ Binomial(b + c, 1/2)``.  The two-sided
    exact p-value is ``min(1, 2 · P[X ≤ min(b, c)])`` with ``X ~ Bin(n, 1/2)``,
    ``n = b + c``.  When there are no discordant pairs the p-value is 1.0.

    No SciPy dependency: the binomial tail is summed exactly with
    :func:`math.comb`.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return McNemarResult(0, b, c, 0, 1.0, None)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    odds = (b / c) if c > 0 else None
    return McNemarResult(n, b, c, k, p, odds)


def mcnemar_from_correctness(
    correct_a: Sequence[bool], correct_b: Sequence[bool]
) -> McNemarResult:
    """McNemar test from two aligned per-item correctness vectors."""
    if len(correct_a) != len(correct_b):
        raise ValueError("correctness vectors must be the same length")
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if (not a) and bb)
    return mcnemar_exact_test(b, c)


@dataclass
class PairedBootstrapResult:
    """Percentile bootstrap CI for a paired difference of a metric."""
    point_estimate: float      # metric(A) - metric(B) on the observed sample
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    fraction_above_zero: float  # share of resamples with diff > 0


def _accuracy(correct: Sequence[bool], idx: Sequence[int]) -> float:
    if not idx:
        return 0.0
    return sum(1 for i in idx if correct[i]) / len(idx)


def paired_bootstrap_accuracy_diff(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Percentile bootstrap CI for ``accuracy(A) − accuracy(B)`` on paired data.

    Items are resampled *jointly* (the same bootstrap index set is used for
    both classifiers) so the pairing — and hence the correlation between the
    two methods' errors — is preserved.
    """
    import random

    if len(correct_a) != len(correct_b):
        raise ValueError("correctness vectors must be the same length")
    n = len(correct_a)
    if n == 0:
        return PairedBootstrapResult(0.0, 0.0, 0.0, confidence, 0, 0.0)

    all_idx = list(range(n))
    point = _accuracy(correct_a, all_idx) - _accuracy(correct_b, all_idx)

    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(n_resamples):
        sample = [rng.randrange(n) for _ in range(n)]
        diffs.append(_accuracy(correct_a, sample) - _accuracy(correct_b, sample))
    diffs.sort()
    lo_q = (1.0 - confidence) / 2.0
    hi_q = 1.0 - lo_q
    lo = diffs[max(0, int(lo_q * (n_resamples - 1)))]
    hi = diffs[min(n_resamples - 1, int(hi_q * (n_resamples - 1)))]
    frac_above = sum(1 for d in diffs if d > 0.0) / n_resamples
    return PairedBootstrapResult(point, lo, hi, confidence, n_resamples, frac_above)


# ─── Integrated Statistical Report ───────────────────────────────────────────

@dataclass
class StatisticalReport:
    """Combined report integrating all three statistical analyses."""
    brier: Optional[BrierDecomposition] = None
    ppv_npv: Optional[PPVNPVCurve] = None
    multiple_comparison: Optional[BHResult] = None
    bonferroni_result: Optional[BonferroniResult] = None
    holm_result: Optional[HolmResult] = None
    fwer_uncorrected: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def to_latex_tables(self) -> str:
        """Generate LaTeX tables for paper integration."""
        parts: List[str] = []

        # Brier decomposition table
        if self.brier is not None:
            b = self.brier
            parts.append(
                "\\begin{table}[h]\n"
                "\\centering\n"
                "\\caption{Brier Score Decomposition (Murphy, 1973)}\n"
                "\\label{tab:brier-decomposition}\n"
                "\\begin{tabular}{lr}\n"
                "\\toprule\n"
                "Component & Value \\\\\n"
                "\\midrule\n"
                f"Brier Score & {b.brier_score:.4f} \\\\\n"
                f"Reliability (calibration) & {b.reliability:.4f} \\\\\n"
                f"Resolution (refinement) & {b.resolution:.4f} \\\\\n"
                f"Uncertainty & {b.uncertainty:.4f} \\\\\n"
                "\\bottomrule\n"
                "\\end{tabular}\n"
                "\\end{table}"
            )

        # PPV/NPV summary table
        if self.ppv_npv is not None:
            pn = self.ppv_npv
            be_str = (
                f"{pn.breakeven_prevalence:.4f}"
                if pn.breakeven_prevalence is not None
                else "N/A"
            )
            parts.append(
                "\\begin{table}[h]\n"
                "\\centering\n"
                "\\caption{Prevalence-Conditioned Predictive Values}\n"
                "\\label{tab:ppv-npv}\n"
                "\\begin{tabular}{lr}\n"
                "\\toprule\n"
                "Parameter & Value \\\\\n"
                "\\midrule\n"
                f"Sensitivity & {pn.sensitivity:.4f} \\\\\n"
                f"Specificity & {pn.specificity:.4f} \\\\\n"
                f"Breakeven prevalence (PPV$\\geq$0.5) & {be_str} \\\\\n"
                f"PPV at $\\pi=0.05$ & {compute_ppv(pn.sensitivity, pn.specificity, 0.05):.4f} \\\\\n"
                f"PPV at $\\pi=0.20$ & {compute_ppv(pn.sensitivity, pn.specificity, 0.20):.4f} \\\\\n"
                f"PPV at $\\pi=0.50$ & {compute_ppv(pn.sensitivity, pn.specificity, 0.50):.4f} \\\\\n"
                "\\bottomrule\n"
                "\\end{tabular}\n"
                "\\end{table}"
            )

        # Multiple comparison table
        if self.multiple_comparison is not None:
            mc = self.multiple_comparison
            parts.append(
                "\\begin{table}[h]\n"
                "\\centering\n"
                "\\caption{Multiple Comparison Correction Results}\n"
                "\\label{tab:multiple-comparison}\n"
                "\\begin{tabular}{lr}\n"
                "\\toprule\n"
                "Metric & Value \\\\\n"
                "\\midrule\n"
                f"Number of tests & {mc.n_tests} \\\\\n"
                f"FDR level ($\\alpha$) & {mc.fdr_level:.3f} \\\\\n"
                f"Rejected (B-H) & {mc.n_rejected} \\\\\n"
            )
            if self.bonferroni_result is not None:
                parts[-1] += (
                    f"Rejected (Bonferroni) & {self.bonferroni_result.n_rejected} \\\\\n"
                )
            if self.holm_result is not None:
                parts[-1] += (
                    f"Rejected (Holm) & {self.holm_result.n_rejected} \\\\\n"
                )
            if self.fwer_uncorrected is not None:
                parts[-1] += (
                    f"FWER (uncorrected) & {self.fwer_uncorrected:.4f} \\\\\n"
                )
            parts[-1] += (
                "\\bottomrule\n"
                "\\end{tabular}\n"
                "\\end{table}"
            )

        return "\n\n".join(parts)

    def to_json(self) -> str:
        """Serialize report to JSON."""
        d: Dict = {"metadata": self.metadata}

        if self.brier is not None:
            b = self.brier
            d["brier_decomposition"] = {
                "brier_score": b.brier_score,
                "reliability": b.reliability,
                "resolution": b.resolution,
                "uncertainty": b.uncertainty,
                "n_bins": b.n_bins,
                "bin_counts": b.bin_counts,
                "bin_accuracies": b.bin_accuracies,
                "bin_mean_probs": b.bin_mean_probs,
            }

        if self.ppv_npv is not None:
            pn = self.ppv_npv
            d["ppv_npv"] = {
                "sensitivity": pn.sensitivity,
                "specificity": pn.specificity,
                "breakeven_prevalence": pn.breakeven_prevalence,
                "prevalences": pn.prevalences,
                "ppv_values": pn.ppv_values,
                "npv_values": pn.npv_values,
            }

        if self.multiple_comparison is not None:
            mc = self.multiple_comparison
            d["multiple_comparison"] = {
                "method": "benjamini_hochberg",
                "n_tests": mc.n_tests,
                "fdr_level": mc.fdr_level,
                "n_rejected": mc.n_rejected,
                "adjusted_p_values": mc.adjusted_p_values,
                "rejected": mc.rejected,
            }

        if self.bonferroni_result is not None:
            br = self.bonferroni_result
            d["bonferroni"] = {
                "n_tests": br.n_tests,
                "alpha": br.alpha,
                "n_rejected": br.n_rejected,
                "adjusted_p_values": br.adjusted_p_values,
            }

        if self.holm_result is not None:
            hr = self.holm_result
            d["holm_bonferroni"] = {
                "n_tests": hr.n_tests,
                "alpha": hr.alpha,
                "n_rejected": hr.n_rejected,
                "adjusted_p_values": hr.adjusted_p_values,
            }

        if self.fwer_uncorrected is not None:
            d["fwer_uncorrected"] = self.fwer_uncorrected

        return json.dumps(d, indent=2)


def generate_report(results_dict: Dict) -> StatisticalReport:
    """Generate a comprehensive statistical report from evaluation results.

    Parameters
    ----------
    results_dict : dict
        Should contain any of:
        - "y_true" / "y_prob": for Brier decomposition
        - "sensitivity" / "specificity": for PPV/NPV curves
        - "p_values": for multiple comparison correction
        - "n_bins": optional bin count (default 10)
        - "alpha": optional significance level (default 0.05)

    Returns
    -------
    StatisticalReport
    """
    report = StatisticalReport()
    report.metadata = {
        k: v for k, v in results_dict.items()
        if k not in ("y_true", "y_prob", "p_values")
    }

    n_bins = results_dict.get("n_bins", 10)
    alpha = results_dict.get("alpha", 0.05)

    # 1. Brier decomposition
    y_true = results_dict.get("y_true")
    y_prob = results_dict.get("y_prob")
    if y_true is not None and y_prob is not None:
        report.brier = brier_decomposition(y_true, y_prob, n_bins=n_bins)

    # 2. PPV/NPV curves
    sens = results_dict.get("sensitivity")
    spec = results_dict.get("specificity")
    if sens is not None and spec is not None:
        report.ppv_npv = ppv_npv_curve(sens, spec)

    # 3. Multiple comparison correction
    pvals = results_dict.get("p_values")
    if pvals is not None and len(pvals) > 0:
        report.multiple_comparison = benjamini_hochberg(pvals, alpha=alpha)
        report.bonferroni_result = bonferroni(pvals, alpha=alpha)
        report.holm_result = holm_bonferroni(pvals, alpha=alpha)
        report.fwer_uncorrected = familywise_error_probability(
            len(pvals), alpha=alpha
        )

    return report


# ─── DerSimonian-Laird Meta-Analysis (Corrected) ─────────────────────────────
#
# The original DerSimonian-Laird pooled estimate is inappropriate for
# k=4 suites with bounded [0,1] metrics (F1 scores).  This corrected
# version:
#   1. Reports honest ranges instead of a single pooled estimate
#   2. Adds logit-transformation for bounded metrics
#   3. Supports meta-regression with a difficulty covariate
#
# Reference: DerSimonian & Laird, "Meta-analysis in clinical trials",
#            Controlled Clinical Trials 7(3), 1986.


@dataclass
class SuiteResult:
    """Per-suite performance result for meta-analysis.

    Attributes
    ----------
    name : str
        Suite name (e.g., ``"easy"``, ``"medium"``, ``"hard"``).
    f1 : float
        F1 score (bounded in [0, 1]).
    n : int
        Number of test cases in this suite.
    difficulty : float
        Difficulty covariate (0 = easiest, 1 = hardest).
    """

    name: str
    f1: float
    n: int
    difficulty: float = 0.0


@dataclass
class HonestRangeReport:
    """Honest range reporting for k=4 suites.

    Instead of a misleading DerSimonian-Laird pooled estimate, reports
    the actual range of F1 scores with caveats about difficulty-dependent
    performance.

    Attributes
    ----------
    f1_min : float
        Minimum F1 across suites.
    f1_max : float
        Maximum F1 across suites.
    f1_values : list of float
        F1 per suite (ordered by difficulty).
    suite_names : list of str
        Suite names (ordered by difficulty).
    weighted_mean : float
        Sample-size-weighted mean (reported with caveat).
    caveat : str
        Required caveat for the weighted mean.
    performance_trend : str
        Description of how performance varies with difficulty.
    """

    f1_min: float
    f1_max: float
    f1_values: List[float]
    suite_names: List[str]
    weighted_mean: float
    caveat: str
    performance_trend: str


@dataclass
class LogitTransformResult:
    """Logit-transformed meta-analysis for bounded [0,1] metrics.

    The logit transformation logit(p) = log(p/(1-p)) maps [0,1] to
    (-∞, +∞), making normal-theory meta-analysis appropriate.

    Attributes
    ----------
    logit_values : list of float
        Logit-transformed F1 values.
    logit_variances : list of float
        Approximate variances of logit-transformed values:
        Var(logit(p)) ≈ 1/(n*p*(1-p))
    pooled_logit : float
        Inverse-variance weighted mean on logit scale.
    pooled_f1 : float
        Back-transformed pooled estimate: expit(pooled_logit).
    ci_lower_f1 : float
        Lower 95% CI on F1 scale.
    ci_upper_f1 : float
        Upper 95% CI on F1 scale.
    """

    logit_values: List[float]
    logit_variances: List[float]
    pooled_logit: float
    pooled_f1: float
    ci_lower_f1: float
    ci_upper_f1: float


@dataclass
class MetaRegressionResult:
    """Meta-regression with difficulty covariate.

    Fits: logit(F1_i) = β_0 + β_1 × difficulty_i + ε_i

    Attributes
    ----------
    intercept : float
        β_0 (logit-scale intercept).
    slope : float
        β_1 (logit-scale slope for difficulty).
    slope_significant : bool
        Whether the slope is statistically significant (|t| > 2).
    r_squared : float
        Proportion of heterogeneity explained by difficulty.
    predicted_f1 : list of float
        Predicted F1 at each suite's difficulty level.
    identifiable : bool
        Whether the regression is identifiable (≥2 distinct difficulty values).
    """

    intercept: float = 0.0
    slope: float = 0.0
    slope_significant: bool = False
    r_squared: float = 0.0
    predicted_f1: List[float] = field(default_factory=list)
    identifiable: bool = False


@dataclass
class CorrectedMetaAnalysis:
    """Corrected meta-analysis replacing DerSimonian-Laird.

    Attributes
    ----------
    honest_range : HonestRangeReport
        Honest range reporting (primary result).
    logit_transform : LogitTransformResult or None
        Logit-transformed analysis (if all F1 in (0,1)).
    meta_regression : MetaRegressionResult or None
        Meta-regression with difficulty (if identifiable).
    original_dl_estimate : float
        The original DerSimonian-Laird pooled estimate (for reference,
        with caveat that it is inappropriate for k=4 heterogeneous suites).
    dl_caveat : str
        Caveat on the DerSimonian-Laird estimate.
    """

    honest_range: Optional[HonestRangeReport] = None
    logit_transform: Optional[LogitTransformResult] = None
    meta_regression: Optional[MetaRegressionResult] = None
    original_dl_estimate: float = 0.0
    dl_caveat: str = (
        "The DerSimonian-Laird pooled estimate is retained for reference "
        "only. With k=4 suites showing systematic difficulty-dependent "
        "heterogeneity, the pooled estimate is misleading. Use the honest "
        "range report as the primary summary."
    )


def _logit(p: float) -> float:
    """Logit transform: log(p / (1 - p))."""
    p = max(min(p, 1.0 - 1e-10), 1e-10)
    return math.log(p / (1.0 - p))


def _expit(x: float) -> float:
    """Inverse logit (expit): 1 / (1 + exp(-x))."""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def honest_range_report(suites: List[SuiteResult]) -> HonestRangeReport:
    """Generate honest range reporting for k suites.

    Instead of a single pooled estimate, reports the range of F1 scores
    with explicit caveat about difficulty-dependent performance.

    Parameters
    ----------
    suites : list of SuiteResult
        Per-suite results, need not be sorted.

    Returns
    -------
    HonestRangeReport
    """
    sorted_suites = sorted(suites, key=lambda s: s.difficulty)
    f1_values = [s.f1 for s in sorted_suites]
    names = [s.name for s in sorted_suites]
    total_n = sum(s.n for s in sorted_suites)

    weighted_mean = (
        sum(s.f1 * s.n for s in sorted_suites) / total_n
        if total_n > 0
        else 0.0
    )

    # Determine performance trend
    if len(f1_values) >= 2:
        if f1_values[-1] < f1_values[0]:
            trend = (
                f"F1 in range [{min(f1_values):.3f}, {max(f1_values):.3f}] "
                f"with performance degrading on harder benchmarks"
            )
        elif f1_values[-1] > f1_values[0]:
            trend = (
                f"F1 in range [{min(f1_values):.3f}, {max(f1_values):.3f}] "
                f"with performance improving on harder benchmarks (unusual)"
            )
        else:
            trend = (
                f"F1 in range [{min(f1_values):.3f}, {max(f1_values):.3f}] "
                f"with stable performance across difficulty levels"
            )
    else:
        trend = f"F1 = {f1_values[0]:.3f} (single suite)"

    return HonestRangeReport(
        f1_min=min(f1_values) if f1_values else 0.0,
        f1_max=max(f1_values) if f1_values else 0.0,
        f1_values=f1_values,
        suite_names=names,
        weighted_mean=weighted_mean,
        caveat=(
            "Weighted mean is reported with caveat: suites show systematic "
            "heterogeneity correlated with difficulty. The honest range "
            "is the appropriate primary summary."
        ),
        performance_trend=trend,
    )


def logit_transform_meta_analysis(
    suites: List[SuiteResult],
) -> Optional[LogitTransformResult]:
    """Logit-transformed meta-analysis for bounded [0,1] metrics.

    Applies logit transformation to make normal-theory methods appropriate,
    then computes inverse-variance weighted pooled estimate.

    Parameters
    ----------
    suites : list of SuiteResult

    Returns
    -------
    LogitTransformResult or None
        None if any F1 is exactly 0 or 1 (logit undefined).
    """
    if not suites:
        return None

    # Check that all F1 are in (0, 1)
    if any(s.f1 <= 0.0 or s.f1 >= 1.0 for s in suites):
        return None

    logit_values = [_logit(s.f1) for s in suites]
    # Approximate variance: Var(logit(p)) ≈ 1 / (n * p * (1-p))
    logit_vars = [
        1.0 / (s.n * s.f1 * (1.0 - s.f1)) for s in suites
    ]

    # Inverse-variance weighted mean
    weights = [1.0 / v for v in logit_vars]
    total_weight = sum(weights)
    pooled_logit = sum(w * lv for w, lv in zip(weights, logit_values)) / total_weight

    # 95% CI on logit scale
    se_pooled = math.sqrt(1.0 / total_weight)
    ci_lower_logit = pooled_logit - 1.96 * se_pooled
    ci_upper_logit = pooled_logit + 1.96 * se_pooled

    return LogitTransformResult(
        logit_values=logit_values,
        logit_variances=logit_vars,
        pooled_logit=pooled_logit,
        pooled_f1=_expit(pooled_logit),
        ci_lower_f1=_expit(ci_lower_logit),
        ci_upper_f1=_expit(ci_upper_logit),
    )


def meta_regression_difficulty(
    suites: List[SuiteResult],
) -> MetaRegressionResult:
    """Meta-regression: logit(F1) = β_0 + β_1 × difficulty.

    Simple weighted least squares on the logit scale.

    Parameters
    ----------
    suites : list of SuiteResult

    Returns
    -------
    MetaRegressionResult
    """
    if len(suites) < 2:
        return MetaRegressionResult(identifiable=False)

    difficulties = [s.difficulty for s in suites]
    if len(set(difficulties)) < 2:
        return MetaRegressionResult(identifiable=False)

    # Logit-transform F1 values (clamp to avoid infinities)
    ys = [_logit(s.f1) for s in suites]
    xs = difficulties

    # Weights: n * p * (1-p) (inverse of logit variance)
    ws = [s.n * max(s.f1, 0.01) * max(1.0 - s.f1, 0.01) for s in suites]

    # Weighted OLS
    sw = sum(ws)
    sx = sum(w * x for w, x in zip(ws, xs))
    sy = sum(w * y for w, y in zip(ws, ys))
    sxx = sum(w * x * x for w, x, in zip(ws, xs))
    sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))

    denom = sw * sxx - sx * sx
    if abs(denom) < 1e-12:
        return MetaRegressionResult(identifiable=False)

    b1 = (sw * sxy - sx * sy) / denom
    b0 = (sy - b1 * sx) / sw

    # Predicted F1
    predicted_logit = [b0 + b1 * x for x in xs]
    predicted_f1 = [_expit(pl) for pl in predicted_logit]

    # R-squared (weighted)
    ss_tot = sum(w * (y - sy / sw) ** 2 for w, y in zip(ws, ys))
    ss_res = sum(w * (y - yh) ** 2 for w, y, yh in zip(ws, ys, predicted_logit))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Approximate significance (|slope / SE| > 2)
    if ss_res > 0 and len(suites) > 2:
        mse = ss_res / (sum(ws) - 2)
        se_b1_sq = mse * sw / denom if denom != 0 else float('inf')
        se_b1 = math.sqrt(max(se_b1_sq, 0))
        significant = abs(b1 / se_b1) > 2.0 if se_b1 > 0 else False
    else:
        significant = len(suites) == 2 and abs(b1) > 0.01

    return MetaRegressionResult(
        intercept=b0,
        slope=b1,
        slope_significant=significant,
        r_squared=max(0.0, min(1.0, r2)),
        predicted_f1=predicted_f1,
        identifiable=True,
    )


def corrected_meta_analysis(
    suites: List[SuiteResult],
) -> CorrectedMetaAnalysis:
    """Run corrected meta-analysis replacing DerSimonian-Laird.

    Produces:
    1. Honest range report (primary)
    2. Logit-transformed pooled estimate (if applicable)
    3. Meta-regression with difficulty covariate (if identifiable)
    4. Original DL estimate (retained with caveat)

    Parameters
    ----------
    suites : list of SuiteResult

    Returns
    -------
    CorrectedMetaAnalysis
    """
    hr = honest_range_report(suites)
    lt = logit_transform_meta_analysis(suites)
    mr = meta_regression_difficulty(suites)

    # Compute original DL estimate for reference
    total_n = sum(s.n for s in suites)
    dl = sum(s.f1 * s.n for s in suites) / total_n if total_n > 0 else 0.0

    return CorrectedMetaAnalysis(
        honest_range=hr,
        logit_transform=lt,
        meta_regression=mr if mr.identifiable else None,
        original_dl_estimate=dl,
    )
