"""
Telemetry-based continuous confidence scorer for TensorGuard.

Replaces the discrete 5-tier confidence mapping (FORMAL/HIGH/MEDIUM/LOW/NONE)
with a logistic regression model trained on solver telemetry features.
The discrete mapping produces zero Brier resolution (RES=0.000); this module
produces calibrated probabilities with positive resolution by exploiting
continuous variation in Z3 query counts, CEGAR iterations, predicate
counts, and structural features.

Usage::

    from src.telemetry_confidence import (
        TelemetryFeatures, TelemetryConfidenceScorer,
        extract_telemetry_features, compute_telemetry_brier_resolution,
    )

    features = extract_telemetry_features(cegar_result)
    scorer = TelemetryConfidenceScorer()
    scorer.fit(features_list, outcomes)
    p = scorer.predict_confidence(features)
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ─── Feature vector ──────────────────────────────────────────────────────────

@dataclass
class TelemetryFeatures:
    """Continuous solver telemetry features extracted from a verification run."""
    z3_queries: int = 0
    z3_sat_count: int = 0
    z3_unsat_count: int = 0
    cegar_iterations: int = 0
    n_predicates_seed: int = 0
    n_predicates_final: int = 0
    n_steps: int = 0
    n_operators: int = 0
    has_broadcast: int = 0       # binary 0/1
    has_reshape: int = 0         # binary 0/1
    has_permutation: int = 0     # binary 0/1
    device_theory_active: int = 0  # binary 0/1
    phase_theory_active: int = 0   # binary 0/1
    elapsed_ms: float = 0.0

    def to_vector(self) -> List[float]:
        """Return feature values as a flat list (fixed order)."""
        return [
            float(self.z3_queries),
            float(self.z3_sat_count),
            float(self.z3_unsat_count),
            float(self.cegar_iterations),
            float(self.n_predicates_seed),
            float(self.n_predicates_final),
            float(self.n_steps),
            float(self.n_operators),
            float(self.has_broadcast),
            float(self.has_reshape),
            float(self.has_permutation),
            float(self.device_theory_active),
            float(self.phase_theory_active),
            float(self.elapsed_ms),
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "z3_queries", "z3_sat_count", "z3_unsat_count",
            "cegar_iterations", "n_predicates_seed", "n_predicates_final",
            "n_steps", "n_operators", "has_broadcast", "has_reshape",
            "has_permutation", "device_theory_active", "phase_theory_active",
            "elapsed_ms",
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── Feature extraction ──────────────────────────────────────────────────────

def extract_telemetry_features(
    cegar_result: Any,
    seed_predicate_count: int = 0,
) -> TelemetryFeatures:
    """Extract telemetry features from a ShapeCEGARResult.

    Parameters
    ----------
    cegar_result : ShapeCEGARResult
        Result from ``run_shape_cegar``.
    seed_predicate_count : int
        Number of seed predicates from guard harvesting (before CEGAR).
    """
    feats = TelemetryFeatures()

    feats.cegar_iterations = getattr(cegar_result, "iterations", 0)
    feats.n_predicates_final = len(getattr(cegar_result, "discovered_predicates", []))
    feats.n_predicates_seed = seed_predicate_count
    feats.elapsed_ms = getattr(cegar_result, "total_time_ms", 0.0)

    # Extract from the final VerificationResult / SafetyCertificate
    vr = getattr(cegar_result, "verification_result", None)
    if vr is not None:
        cert = getattr(vr, "certificate", None)
        if cert is not None:
            feats.z3_queries = getattr(cert, "z3_queries", 0)
            feats.z3_sat_count = getattr(cert, "z3_sat_count", 0)
            feats.z3_unsat_count = getattr(cert, "z3_unsat_count", 0)
            feats.n_steps = getattr(cert, "checked_steps", 0)

            theories = getattr(cert, "theories_used", [])
            feats.device_theory_active = int("T_device" in theories)
            feats.phase_theory_active = int("T_phase" in theories)

        graph = getattr(vr, "graph", None)
        if graph is not None:
            steps = getattr(graph, "steps", [])
            feats.n_steps = feats.n_steps or len(steps)
            op_names = set()
            for step in steps:
                op = getattr(step, "op", None)
                if op is not None:
                    op_name = op.name if hasattr(op, "name") else str(op)
                    op_names.add(op_name)
                    if op_name in ("ADD", "MULTIPLY", "MATMUL", "CAT"):
                        feats.has_broadcast = 1
                    if op_name in ("RESHAPE", "FLATTEN"):
                        feats.has_reshape = 1
                    if op_name in ("PERMUTE", "TRANSPOSE"):
                        feats.has_permutation = 1
            feats.n_operators = len(op_names)

    return feats


def features_from_dict(d: Dict[str, Any]) -> TelemetryFeatures:
    """Reconstruct TelemetryFeatures from a serialised dict."""
    feats = TelemetryFeatures()
    for name in TelemetryFeatures.feature_names():
        if name in d:
            setattr(feats, name, d[name])
    return feats


# ─── Logistic regression (from scratch) ──────────────────────────────────────

def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


class TelemetryConfidenceScorer:
    """Logistic regression confidence scorer trained on solver telemetry.

    Implements gradient-descent training with L2 regularisation.
    No external ML library dependencies — uses only ``math``.
    """

    def __init__(self) -> None:
        self.weights: List[float] = []
        self.bias: float = 0.0
        self._fitted = False
        self._feature_means: List[float] = []
        self._feature_stds: List[float] = []

    # ── standardisation ──────────────────────────────────────────────────

    def _standardise(self, X: List[List[float]]) -> List[List[float]]:
        """Compute and apply z-score standardisation."""
        n = len(X)
        d = len(X[0]) if X else 0
        means = [0.0] * d
        for row in X:
            for j in range(d):
                means[j] += row[j]
        means = [m / n for m in means]

        stds = [0.0] * d
        for row in X:
            for j in range(d):
                stds[j] += (row[j] - means[j]) ** 2
        stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in stds]

        self._feature_means = means
        self._feature_stds = stds
        return self._apply_standardise(X)

    def _apply_standardise(self, X: List[List[float]]) -> List[List[float]]:
        """Apply stored standardisation parameters."""
        result = []
        for row in X:
            result.append([
                (row[j] - self._feature_means[j]) / self._feature_stds[j]
                for j in range(len(row))
            ])
        return result

    # ── training ─────────────────────────────────────────────────────────

    def fit(
        self,
        features_list: List[TelemetryFeatures],
        outcomes: List[int],
        lr: float = 0.1,
        epochs: int = 500,
        reg_lambda: float = 0.01,
    ) -> None:
        """Train logistic regression via gradient descent.

        Parameters
        ----------
        features_list : list of TelemetryFeatures
            One per sample.
        outcomes : list of int
            Binary labels (1 = safe / verified, 0 = bug found / unsafe).
        lr : float
            Learning rate.
        epochs : int
            Number of gradient descent iterations.
        reg_lambda : float
            L2 regularisation strength.
        """
        if not features_list or not outcomes:
            return

        X_raw = [f.to_vector() for f in features_list]
        y = [float(o) for o in outcomes]
        n = len(X_raw)
        d = len(X_raw[0])

        X = self._standardise(X_raw)

        self.weights = [0.0] * d
        self.bias = 0.0

        for _ in range(epochs):
            grad_w = [0.0] * d
            grad_b = 0.0

            for i in range(n):
                z = _dot(self.weights, X[i]) + self.bias
                p = _sigmoid(z)
                err = p - y[i]
                for j in range(d):
                    grad_w[j] += err * X[i][j]
                grad_b += err

            for j in range(d):
                grad_w[j] = grad_w[j] / n + reg_lambda * self.weights[j]
            grad_b /= n

            for j in range(d):
                self.weights[j] -= lr * grad_w[j]
            self.bias -= lr * grad_b

        self._fitted = True

    # ── prediction ───────────────────────────────────────────────────────

    def predict_confidence(self, features: TelemetryFeatures) -> float:
        """Return a calibrated probability of correctness for *features*.

        Returns 0.5 if the model has not been fitted.
        """
        if not self._fitted:
            return 0.5
        x = features.to_vector()
        x_std = self._apply_standardise([x])[0]
        z = _dot(self.weights, x_std) + self.bias
        return _sigmoid(z)

    def predict_batch(self, features_list: List[TelemetryFeatures]) -> List[float]:
        return [self.predict_confidence(f) for f in features_list]

    # ── evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        features_list: List[TelemetryFeatures],
        outcomes: List[int],
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Evaluate the scorer on a held-out set.

        Returns
        -------
        dict with keys:
            auc_roc : float   — AUC-ROC approximation (trapezoidal)
            brier_score : float
            brier_resolution : float
            brier_reliability : float
            brier_uncertainty : float
        """
        probs = self.predict_batch(features_list)
        y = [float(o) for o in outcomes]
        n = len(y)

        # Brier score
        brier = sum((p - yi) ** 2 for p, yi in zip(probs, y)) / n if n else 0.0

        # Brier decomposition (Murphy 1973)
        rel, res, unc = _brier_decomposition(y, probs, n_bins)

        # AUC-ROC (trapezoidal approximation)
        auc = _auc_roc(y, probs)

        return {
            "auc_roc": round(auc, 6),
            "brier_score": round(brier, 6),
            "brier_resolution": round(res, 6),
            "brier_reliability": round(rel, 6),
            "brier_uncertainty": round(unc, 6),
            "n_samples": n,
        }


# ─── Brier decomposition (standalone, no dependency on calibration_analysis) ─

def _brier_decomposition(
    y: List[float],
    probs: List[float],
    n_bins: int = 10,
) -> Tuple[float, float, float]:
    """Murphy (1973) decomposition: Brier = REL - RES + UNC."""
    n = len(y)
    if n == 0:
        return 0.0, 0.0, 0.0

    o_bar = sum(y) / n

    bin_sums_prob = [0.0] * n_bins
    bin_sums_out = [0.0] * n_bins
    bin_counts = [0] * n_bins

    for yi, pi in zip(y, probs):
        idx = min(int(pi * n_bins), n_bins - 1)
        bin_counts[idx] += 1
        bin_sums_prob[idx] += pi
        bin_sums_out[idx] += yi

    rel = 0.0
    res = 0.0
    for k in range(n_bins):
        if bin_counts[k] == 0:
            continue
        ok = bin_sums_out[k] / bin_counts[k]
        fk = bin_sums_prob[k] / bin_counts[k]
        rel += bin_counts[k] * (ok - fk) ** 2
        res += bin_counts[k] * (ok - o_bar) ** 2

    rel /= n
    res /= n
    unc = o_bar * (1.0 - o_bar)
    return rel, res, unc


def _auc_roc(y: List[float], probs: List[float]) -> float:
    """Trapezoidal AUC-ROC approximation."""
    n = len(y)
    if n == 0:
        return 0.5

    # Pair (prob, label), sort descending by prob
    pairs = sorted(zip(probs, y), key=lambda t: -t[0])
    n_pos = sum(1 for yi in y if yi > 0.5)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0.0
    fp = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0
    auc = 0.0

    for prob, label in pairs:
        if label > 0.5:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += 0.5 * (tpr + prev_tpr) * (fpr - prev_fpr)
        prev_tpr = tpr
        prev_fpr = fpr

    return auc


# ─── Integration: compute Brier resolution from telemetry ────────────────────

def compute_telemetry_brier_resolution(
    results_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare discrete-confidence RES vs telemetry-confidence RES.

    If *results_json_path* is given, loads benchmark results from that file
    (expected to have a "benchmarks" key with per-benchmark entries containing
    "telemetry_features" and "has_bug" / "detected" fields).

    Returns a dict with old_RES, new_RES, and improvement info.
    """
    from src.calibration_analysis import CONFIDENCE_MAP

    if results_json_path and os.path.isfile(results_json_path):
        with open(results_json_path) as f:
            data = json.load(f)
        benchmarks = data.get("benchmarks", [])
    else:
        benchmarks = []

    features_list: List[TelemetryFeatures] = []
    outcomes: List[int] = []
    discrete_probs: List[float] = []

    for bm in benchmarks:
        tf = bm.get("telemetry_features")
        if tf is None:
            continue
        feats = features_from_dict(tf)
        features_list.append(feats)

        outcome = 1 if bm.get("detected", False) == bm.get("has_bug", False) else 0
        outcomes.append(outcome)

        conf_name = bm.get("confidence_level", "MEDIUM")
        discrete_probs.append(CONFIDENCE_MAP.get(conf_name.upper(), 0.60))

    if len(features_list) < 3:
        return {
            "error": "Not enough benchmark data to compute resolution",
            "n_benchmarks": len(features_list),
        }

    # Old discrete-confidence Brier decomposition
    y = [float(o) for o in outcomes]
    old_rel, old_res, old_unc = _brier_decomposition(y, discrete_probs)

    # New telemetry-based: LOO cross-validation
    loo_probs: List[float] = []
    for i in range(len(features_list)):
        train_f = features_list[:i] + features_list[i + 1:]
        train_y = outcomes[:i] + outcomes[i + 1:]
        scorer = TelemetryConfidenceScorer()
        scorer.fit(train_f, train_y)
        loo_probs.append(scorer.predict_confidence(features_list[i]))

    new_rel, new_res, new_unc = _brier_decomposition(y, loo_probs)
    new_brier = sum((p - yi) ** 2 for p, yi in zip(loo_probs, y)) / len(y)

    return {
        "n_benchmarks": len(features_list),
        "discrete_confidence": {
            "REL": round(old_rel, 6),
            "RES": round(old_res, 6),
            "UNC": round(old_unc, 6),
        },
        "telemetry_confidence": {
            "REL": round(new_rel, 6),
            "RES": round(new_res, 6),
            "UNC": round(new_unc, 6),
            "brier": round(new_brier, 6),
        },
        "resolution_improvement": round(new_res - old_res, 6),
        "resolution_positive": new_res > 0,
    }
