"""
Expected-cost deployment analysis for TensorGuard.

Implements expected cost analysis under asymmetric loss ratios
L(FN)/L(FP) ∈ {10, 100, 1000}, incorporating CompositionSemantics
action dimensions (MONOLITHIC_SAFE, PER_SUBGRAPH_SAFE, UNKNOWN).

The key insight: false negatives (missed shape bugs that reach
production) are far more costly than false positives (spurious
warnings that waste developer time).  The optimal verification
strategy depends on the loss ratio regime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# CompositionSemantics (mirrors dynamo_extractor but standalone)
# ═══════════════════════════════════════════════════════════════════════════════

class CompositionSemantics(Enum):
    """Verification scope semantics."""
    MONOLITHIC_SAFE = auto()
    PER_SUBGRAPH_SAFE = auto()
    UNKNOWN = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# Verification result abstraction
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationOutcome:
    """Simplified verification outcome for cost analysis."""
    predicted_safe: bool
    actual_safe: Optional[bool] = None  # ground truth if available
    confidence: float = 0.5
    composition_semantics: str = "MONOLITHIC_SAFE"
    backend: str = "unknown"
    num_graph_breaks: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Loss model
# ═══════════════════════════════════════════════════════════════════════════════

# False positive cost: developer investigates a spurious warning.
DEFAULT_FP_COST = 1.0


def _fn_cost(loss_ratio: float) -> float:
    """Cost of a false negative given the loss ratio L(FN)/L(FP)."""
    return DEFAULT_FP_COST * loss_ratio


# ═══════════════════════════════════════════════════════════════════════════════
# Composition semantics cost modifiers
# ═══════════════════════════════════════════════════════════════════════════════

# PER_SUBGRAPH_SAFE has a higher effective FN rate because cross-break
# dependencies may be missed.  UNKNOWN is worse still.
SEMANTICS_FN_MULTIPLIER = {
    "MONOLITHIC_SAFE": 1.0,
    "PER_SUBGRAPH_SAFE": 1.15,  # ~15% higher FN risk
    "UNKNOWN": 1.5,             # ~50% higher FN risk
}

SEMANTICS_FP_MULTIPLIER = {
    "MONOLITHIC_SAFE": 1.0,
    "PER_SUBGRAPH_SAFE": 1.05,
    "UNKNOWN": 1.1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Cost computation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a verification strategy."""
    total_expected_cost: float
    fp_cost: float
    fn_cost: float
    tp_cost: float  # true positive: correctly flagged bug (developer fixes)
    tn_cost: float  # true negative: correctly verified safe (zero cost)
    num_fp: int
    num_fn: int
    num_tp: int
    num_tn: int
    loss_ratio: float
    composition_semantics: str
    strategy_recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "total_expected_cost": round(self.total_expected_cost, 4),
            "fp_cost": round(self.fp_cost, 4),
            "fn_cost": round(self.fn_cost, 4),
            "tp_cost": round(self.tp_cost, 4),
            "tn_cost": round(self.tn_cost, 4),
            "num_fp": self.num_fp,
            "num_fn": self.num_fn,
            "num_tp": self.num_tp,
            "num_tn": self.num_tn,
            "loss_ratio": self.loss_ratio,
            "composition_semantics": self.composition_semantics,
            "strategy_recommendation": self.strategy_recommendation,
        }


@dataclass
class DeploymentCostReport:
    """Full deployment cost analysis across loss ratios."""
    cost_by_ratio: Dict[float, CostBreakdown] = field(default_factory=dict)
    optimal_strategy: str = ""
    optimal_loss_ratio: float = 0.0
    composition_semantics: str = "MONOLITHIC_SAFE"
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "cost_by_ratio": {
                str(k): v.to_dict() for k, v in self.cost_by_ratio.items()
            },
            "optimal_strategy": self.optimal_strategy,
            "optimal_loss_ratio": self.optimal_loss_ratio,
            "composition_semantics": self.composition_semantics,
            "summary": self.summary,
        }


def compute_deployment_cost(
    verification_results: Sequence[VerificationOutcome],
    loss_ratio: float,
    composition_semantics: str = "MONOLITHIC_SAFE",
) -> CostBreakdown:
    """Compute expected deployment cost for a set of verification outcomes.

    Parameters
    ----------
    verification_results : sequence of VerificationOutcome
        The verification outcomes to analyze.
    loss_ratio : float
        The asymmetric loss ratio L(FN)/L(FP).
    composition_semantics : str
        One of MONOLITHIC_SAFE, PER_SUBGRAPH_SAFE, UNKNOWN.

    Returns
    -------
    CostBreakdown
    """
    if not verification_results:
        return CostBreakdown(
            total_expected_cost=0.0,
            fp_cost=0.0, fn_cost=0.0,
            tp_cost=0.0, tn_cost=0.0,
            num_fp=0, num_fn=0, num_tp=0, num_tn=0,
            loss_ratio=loss_ratio,
            composition_semantics=composition_semantics,
        )

    fn_multiplier = SEMANTICS_FN_MULTIPLIER.get(composition_semantics, 1.0)
    fp_multiplier = SEMANTICS_FP_MULTIPLIER.get(composition_semantics, 1.0)

    num_tp = num_tn = num_fp = num_fn = 0

    for v in verification_results:
        if v.actual_safe is None:
            # No ground truth: use probabilistic estimate
            if v.predicted_safe:
                # Predicted safe: risk of FN proportional to (1 - confidence)
                p_fn = 1.0 - v.confidence
                num_fn += p_fn
                num_tn += 1.0 - p_fn
            else:
                # Predicted unsafe: risk of FP proportional to (1 - confidence)
                p_fp = 1.0 - v.confidence
                num_fp += p_fp
                num_tp += 1.0 - p_fp
        else:
            if v.predicted_safe and v.actual_safe:
                num_tn += 1
            elif v.predicted_safe and not v.actual_safe:
                num_fn += 1
            elif not v.predicted_safe and v.actual_safe:
                num_fp += 1
            else:  # not predicted_safe and not actual_safe
                num_tp += 1

    fn_unit_cost = _fn_cost(loss_ratio) * fn_multiplier
    fp_unit_cost = DEFAULT_FP_COST * fp_multiplier
    tp_unit_cost = DEFAULT_FP_COST * 0.5  # fixing a real bug still costs time

    total_fn_cost = num_fn * fn_unit_cost
    total_fp_cost = num_fp * fp_unit_cost
    total_tp_cost = num_tp * tp_unit_cost
    total_tn_cost = 0.0  # correct safe verdict: no cost

    total = total_fn_cost + total_fp_cost + total_tp_cost + total_tn_cost

    # Strategy recommendation
    recommendation = _recommend_strategy(
        loss_ratio, composition_semantics, num_fn, num_fp, len(verification_results)
    )

    return CostBreakdown(
        total_expected_cost=total,
        fp_cost=total_fp_cost,
        fn_cost=total_fn_cost,
        tp_cost=total_tp_cost,
        tn_cost=total_tn_cost,
        num_fp=int(round(num_fp)),
        num_fn=int(round(num_fn)),
        num_tp=int(round(num_tp)),
        num_tn=int(round(num_tn)),
        loss_ratio=loss_ratio,
        composition_semantics=composition_semantics,
        strategy_recommendation=recommendation,
    )


def _recommend_strategy(
    loss_ratio: float,
    composition_semantics: str,
    num_fn: float,
    num_fp: float,
    total: int,
) -> str:
    """Recommend optimal verification strategy for the given regime."""
    if loss_ratio >= 1000:
        if composition_semantics == "UNKNOWN":
            return (
                "CRITICAL: Use MONOLITHIC_SAFE verification only. "
                "At L(FN)/L(FP)=1000, any missed bug is catastrophic. "
                "Cross-break gaps in UNKNOWN semantics are unacceptable."
            )
        if composition_semantics == "PER_SUBGRAPH_SAFE":
            return (
                "HIGH PRIORITY: Upgrade to MONOLITHIC_SAFE if possible. "
                "PER_SUBGRAPH_SAFE has ~15% higher FN risk; at L(FN)/L(FP)=1000, "
                "this translates to significant expected cost increase."
            )
        return (
            "MONOLITHIC_SAFE verification is optimal. "
            "Accept higher FP rates to minimize catastrophic FN costs."
        )
    elif loss_ratio >= 100:
        if composition_semantics == "UNKNOWN":
            return (
                "Strongly prefer MONOLITHIC_SAFE or PER_SUBGRAPH_SAFE. "
                "UNKNOWN semantics add substantial risk at L(FN)/L(FP)=100."
            )
        return (
            "Balance FP/FN tradeoff. Use highest-confidence verification "
            "available. PER_SUBGRAPH_SAFE acceptable if graph breaks are few."
        )
    else:  # loss_ratio <= 10
        return (
            "FP cost dominates at low loss ratios. "
            "Prefer high-precision verification to minimize false alarms. "
            "PER_SUBGRAPH_SAFE is acceptable for most models."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-ratio analysis
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_LOSS_RATIOS = [10.0, 100.0, 1000.0]


def compute_full_deployment_analysis(
    verification_results: Sequence[VerificationOutcome],
    composition_semantics: str = "MONOLITHIC_SAFE",
    loss_ratios: Optional[Sequence[float]] = None,
) -> DeploymentCostReport:
    """Compute deployment cost analysis across multiple loss ratios.

    Parameters
    ----------
    verification_results : sequence of VerificationOutcome
        Outcomes to analyze.
    composition_semantics : str
        Composition semantics to use.
    loss_ratios : sequence of float, optional
        Loss ratios to evaluate. Defaults to [10, 100, 1000].

    Returns
    -------
    DeploymentCostReport
    """
    ratios = loss_ratios or DEFAULT_LOSS_RATIOS

    report = DeploymentCostReport(
        composition_semantics=composition_semantics,
    )

    min_cost = float("inf")
    best_ratio = ratios[0]

    for ratio in ratios:
        breakdown = compute_deployment_cost(
            verification_results, ratio, composition_semantics
        )
        report.cost_by_ratio[ratio] = breakdown

        if breakdown.total_expected_cost < min_cost:
            min_cost = breakdown.total_expected_cost
            best_ratio = ratio

    report.optimal_loss_ratio = best_ratio
    report.optimal_strategy = report.cost_by_ratio[best_ratio].strategy_recommendation

    # Summary
    costs = {r: b.total_expected_cost for r, b in report.cost_by_ratio.items()}
    report.summary = (
        f"Deployment cost analysis under {composition_semantics}: "
        f"costs by L(FN)/L(FP) ratio = {costs}. "
        f"Optimal strategy at ratio {best_ratio}."
    )

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-semantics comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_semantics_cost(
    verification_results: Sequence[VerificationOutcome],
    loss_ratio: float,
) -> Dict[str, CostBreakdown]:
    """Compare costs across all three CompositionSemantics for a given ratio."""
    return {
        sem: compute_deployment_cost(verification_results, loss_ratio, sem)
        for sem in ["MONOLITHIC_SAFE", "PER_SUBGRAPH_SAFE", "UNKNOWN"]
    }


def compute_cost_sensitivity(
    verification_results: Sequence[VerificationOutcome],
    composition_semantics: str = "MONOLITHIC_SAFE",
    loss_ratios: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute sensitivity of expected cost to loss ratio changes."""
    ratios = loss_ratios or DEFAULT_LOSS_RATIOS
    costs = []
    for r in ratios:
        bd = compute_deployment_cost(verification_results, r, composition_semantics)
        costs.append(bd.total_expected_cost)

    # Compute elasticity: % change in cost / % change in ratio
    elasticities = []
    for i in range(1, len(ratios)):
        if costs[i - 1] == 0 or ratios[i - 1] == 0:
            elasticities.append(0.0)
        else:
            pct_cost = (costs[i] - costs[i - 1]) / costs[i - 1]
            pct_ratio = (ratios[i] - ratios[i - 1]) / ratios[i - 1]
            elasticities.append(pct_cost / pct_ratio if pct_ratio != 0 else 0.0)

    return {
        "loss_ratios": list(ratios),
        "expected_costs": costs,
        "elasticities": elasticities,
        "cost_dominated_by": "FN" if costs[-1] > costs[0] * 5 else "balanced",
    }
