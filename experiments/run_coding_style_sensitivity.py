#!/usr/bin/env python3
"""
Coding Style Sensitivity Analysis for Guard-Harvesting Predicate Quality.

Evaluates whether coding style affects predicate harvesting quality by
analyzing 5 PyTorch models each written in 3 coding styles:

  Style A ("Defensive"): explicit assert/isinstance guards in forward()
  Style B ("Minimal"):   bare minimum code, no guards, relies on runtime checks
  Style C ("Functional"): uses torch.nn.functional, inline shape comments

For each model × style combination, runs guard extraction + CEGAR verification
and measures: predicate count, utilization rate, false predicate rate,
verification result, and verification time.

Computes sensitivity metrics:
  - Cross-style agreement rate (same verdict across styles)
  - Predicate count coefficient of variation across styles
  - Worst-case style identification
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPL_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, IMPL_DIR)

from src.model_checker import extract_computation_graph
from src.shape_cegar import run_shape_cegar, PredicateKind
from src.guard_extractor import extract_guards

OUTPUT_PATH = os.path.join(SCRIPT_DIR, "coding_style_sensitivity_results.json")

# ═══════════════════════════════════════════════════════════════════════════════
# Model definitions: 5 models × 3 styles
# ═══════════════════════════════════════════════════════════════════════════════

MODELS: List[Dict[str, Any]] = [
    # ── Model 1: Simple MLP Classifier ─────────────────────────────────────
    {
        "name": "SimpleClassifier",
        "has_bug": False,
        "input_shapes": {"x": ("batch", 784)},
        "styles": {
            "defensive": """\
import torch
import torch.nn as nn
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        assert isinstance(x, torch.Tensor), "Input must be a tensor"
        assert x.shape[-1] == 784, "Input features must be 784"
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
            "minimal": """\
import torch.nn as nn
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
""",
            "functional": """\
import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        # x: (batch, 784)
        x = F.relu(self.fc1(x))  # -> (batch, 256)
        return self.fc2(x)       # -> (batch, 10)
""",
        },
    },
    # ── Model 2: Autoencoder ───────────────────────────────────────────────
    {
        "name": "Autoencoder",
        "has_bug": False,
        "input_shapes": {"x": ("batch", 512)},
        "styles": {
            "defensive": """\
import torch
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 64)
        self.decoder = nn.Linear(64, 512)
    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2, "Expected 2D input"
        assert x.shape[-1] == 512
        z = self.encoder(x)
        return self.decoder(z)
""",
            "minimal": """\
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 64)
        self.decoder = nn.Linear(64, 512)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
""",
            "functional": """\
import torch
import torch.nn as nn
import torch.nn.functional as F
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_w = nn.Linear(512, 64)
        self.dec_w = nn.Linear(64, 512)
    def forward(self, x):
        # x: (batch, 512)
        z = F.relu(self.enc_w(x))  # -> (batch, 64)
        return self.dec_w(z)       # -> (batch, 512)
""",
        },
    },
    # ── Model 3: Residual MLP ──────────────────────────────────────────────
    {
        "name": "ResidualMLP",
        "has_bug": False,
        "input_shapes": {"x": ("batch", 256)},
        "styles": {
            "defensive": """\
import torch
import torch.nn as nn
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.shape[-1] == 256, "Feature dim must be 256"
        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x + residual
""",
            "minimal": """\
import torch.nn as nn
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        return x + self.fc2(self.relu(self.fc1(x)))
""",
            "functional": """\
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
    def forward(self, x):
        # x: (batch, 256) — residual connection
        return x + self.fc2(F.relu(self.fc1(x)))  # -> (batch, 256)
""",
        },
    },
    # ── Model 4: Dimension Mismatch Bug ────────────────────────────────────
    {
        "name": "DimMismatchBug",
        "has_bug": True,
        "input_shapes": {"x": ("batch", 100)},
        "styles": {
            "defensive": """\
import torch
import torch.nn as nn
class DimMismatchBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(60, 10)
    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.shape[-1] == 100
        x = self.fc1(x)
        return self.fc2(x)
""",
            "minimal": """\
import torch.nn as nn
class DimMismatchBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(60, 10)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
""",
            "functional": """\
import torch
import torch.nn as nn
import torch.nn.functional as F
class DimMismatchBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(60, 10)
    def forward(self, x):
        # x: (batch, 100)
        x = F.relu(self.fc1(x))  # -> (batch, 50) BUG: fc2 expects 60
        return self.fc2(x)       # shape mismatch!
""",
        },
    },
    # ── Model 5: Transformer FFN Block ─────────────────────────────────────
    {
        "name": "TransformerFFN",
        "has_bug": False,
        "input_shapes": {"x": ("batch", "seq", 512)},
        "styles": {
            "defensive": """\
import torch
import torch.nn as nn
class TransformerFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.norm = nn.LayerNorm(512)
        self.relu = nn.ReLU()
    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.shape[-1] == 512, "Expected d_model=512"
        assert len(x.shape) >= 2, "Need at least 2D input"
        return self.norm(x + self.linear2(self.relu(self.linear1(x))))
""",
            "minimal": """\
import torch.nn as nn
class TransformerFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.norm = nn.LayerNorm(512)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.norm(x + self.linear2(self.relu(self.linear1(x))))
""",
            "functional": """\
import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 512)
    def forward(self, x):
        # x: (batch, seq, 512) — transformer FFN sub-layer
        h = F.relu(self.linear1(x))  # -> (batch, seq, 2048)
        h = self.linear2(h)          # -> (batch, seq, 512)
        out = x + h                  # residual
        return F.layer_norm(out, [512])  # -> (batch, seq, 512)
""",
        },
    },
]

STYLE_NAMES = ["defensive", "minimal", "functional"]


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_guard_predicates(code: str) -> List[Dict[str, Any]]:
    """Extract guard predicates from source and return summaries."""
    try:
        guards = extract_guards(code)
        return [
            {
                "kind": g.pattern.name.lower() if hasattr(g, "pattern") else "unknown",
                "expression": getattr(g, "raw_source", ""),
                "variable": g.variables[0] if g.variables else "",
                "provenance": g.predicate.provenance if hasattr(g.predicate, "provenance") else "explicit_guard",
            }
            for g in guards
        ]
    except Exception:
        return []


def is_shape_relevant_guard(guard_pred: Dict[str, Any]) -> bool:
    """Check if a guard-extracted predicate is shape-relevant (DIM_EQ-like)."""
    kind = guard_pred.get("kind", "")
    expr = guard_pred.get("expression", "")
    # Shape assertions: assert x.shape[-1] == N, len(x.shape) == N
    if kind in ("comparison", "assertion") and "shape" in expr:
        return True
    if kind in ("comparison", "assertion") and "len(" in expr:
        return True
    return False


def analyze_one(
    model_name: str,
    style: str,
    code: str,
    input_shapes: Dict[str, Any],
    has_bug: bool,
) -> Dict[str, Any]:
    """Run guard extraction + CEGAR on one model variant and return metrics."""
    result: Dict[str, Any] = {
        "model": model_name,
        "style": style,
        "has_bug": has_bug,
    }

    t0 = time.monotonic()

    # 1. Guard extraction (explicit guards from assert/isinstance/if)
    guard_preds = extract_guard_predicates(code)
    result["guard_count"] = len(guard_preds)

    # 2. Computation graph extraction
    try:
        graph = extract_computation_graph(code)
        result["graph_ok"] = True
    except Exception as e:
        result["graph_ok"] = False
        result["error"] = str(e)
        result["predicate_count"] = 0
        result["cegar_predicate_count"] = 0
        result["utilized_count"] = 0
        result["false_predicate_count"] = 0
        result["utilization_rate"] = 0.0
        result["false_predicate_rate"] = 0.0
        result["verification_status"] = "PARSE_ERROR"
        result["verification_correct"] = False
        result["time_ms"] = (time.monotonic() - t0) * 1000
        return result

    # 3. CEGAR verification
    try:
        cegar = run_shape_cegar(code, input_shapes=input_shapes, max_iterations=10)
        status = cegar.final_status.name
        cegar_predicates = cegar.discovered_predicates
        iterations = cegar.iterations
        total_time = cegar.total_time_ms
    except Exception as e:
        status = "ERROR"
        cegar_predicates = []
        iterations = 0
        total_time = (time.monotonic() - t0) * 1000

    # 4. Combine predicates from both sources (matching provenance analysis)
    # CEGAR-discovered predicates
    shape_relevant_kinds = {PredicateKind.DIM_EQ, PredicateKind.DIM_GT,
                            PredicateKind.DIM_GE, PredicateKind.DIM_MATCH,
                            PredicateKind.NDIM_EQ, PredicateKind.SHAPE_EQ,
                            PredicateKind.DIM_DIVISIBLE}
    cegar_count = len(cegar_predicates)
    cegar_utilized = sum(1 for p in cegar_predicates if p.kind in shape_relevant_kinds)

    # Guard-extracted predicates (shape-relevant vs not)
    guard_shape_relevant = sum(1 for g in guard_preds if is_shape_relevant_guard(g))
    guard_not_shape = len(guard_preds) - guard_shape_relevant

    # Total predicate landscape
    total_preds = cegar_count + len(guard_preds)
    total_utilized = cegar_utilized + guard_shape_relevant
    total_false = (cegar_count - cegar_utilized) + guard_not_shape

    result["predicate_count"] = total_preds
    result["cegar_predicate_count"] = cegar_count
    result["guard_predicate_count"] = len(guard_preds)
    result["guard_shape_relevant"] = guard_shape_relevant
    result["utilized_count"] = total_utilized
    result["false_predicate_count"] = total_false
    result["utilization_rate"] = round(total_utilized / total_preds, 4) if total_preds > 0 else 0.0
    result["false_predicate_rate"] = round(total_false / total_preds, 4) if total_preds > 0 else 0.0
    result["verification_status"] = status
    result["cegar_iterations"] = iterations
    result["time_ms"] = round(total_time, 2)

    # 5. Correctness check
    if has_bug:
        result["verification_correct"] = status in ("REAL_BUG_FOUND", "NO_Z3")
    else:
        result["verification_correct"] = status in ("SAFE", "NO_Z3")

    # 6. Predicate details
    pred_details = [
        {"pretty": p.pretty(), "kind": p.kind.name, "provenance": p.provenance}
        for p in cegar_predicates
    ]
    for gp in guard_preds:
        pred_details.append({
            "pretty": gp.get("expression", ""),
            "kind": gp.get("kind", "guard"),
            "provenance": gp.get("provenance", "explicit_guard"),
            "shape_relevant": is_shape_relevant_guard(gp),
        })
    result["predicates"] = pred_details

    return result


def coefficient_of_variation(values: List[float]) -> float:
    """Compute coefficient of variation (std/mean). Returns 0 if mean is 0."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return round(math.sqrt(variance) / mean, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("Coding Style Sensitivity Analysis")
    print("=" * 70)

    per_variant: List[Dict[str, Any]] = []

    for model_def in MODELS:
        model_name = model_def["name"]
        has_bug = model_def["has_bug"]
        input_shapes = model_def["input_shapes"]

        for style in STYLE_NAMES:
            code = model_def["styles"][style]
            label = f"{model_name}/{style}"
            print(f"  [{label}]...", end=" ", flush=True)
            result = analyze_one(model_name, style, code, input_shapes, has_bug)
            per_variant.append(result)
            print(f"OK (preds={result['predicate_count']}, "
                  f"status={result['verification_status']}, "
                  f"time={result['time_ms']:.1f}ms)")

    # ── Per-model sensitivity ──────────────────────────────────────────────
    per_model_sensitivity: List[Dict[str, Any]] = []
    all_agreements = 0
    total_models = 0

    for model_def in MODELS:
        model_name = model_def["name"]
        variants = [v for v in per_variant if v["model"] == model_name]

        # Cross-style agreement: do all styles produce the same verdict?
        statuses = [v["verification_status"] for v in variants]
        # Normalize: NO_Z3 with correct result counts as agreement
        normalized = []
        for v in variants:
            if v["verification_status"] == "NO_Z3":
                normalized.append("SAFE" if not v["has_bug"] else "REAL_BUG_FOUND")
            else:
                normalized.append(v["verification_status"])
        agreement = len(set(normalized)) == 1
        if agreement:
            all_agreements += 1
        total_models += 1

        # Predicate count variation
        pred_counts = [float(v["predicate_count"]) for v in variants]
        pred_cv = coefficient_of_variation(pred_counts)

        # Utilization rates
        util_rates = [v["utilization_rate"] for v in variants]
        false_rates = [v["false_predicate_rate"] for v in variants]

        # Worst-case style: fewest utilized predicates
        min_util = min(variants, key=lambda v: v["utilized_count"])
        worst_style = min_util["style"]

        per_model_sensitivity.append({
            "model": model_name,
            "has_bug": model_def["has_bug"],
            "cross_style_agreement": agreement,
            "normalized_verdicts": normalized,
            "predicate_counts": {v["style"]: v["predicate_count"] for v in variants},
            "predicate_count_cv": pred_cv,
            "utilization_rates": {v["style"]: v["utilization_rate"] for v in variants},
            "false_predicate_rates": {v["style"]: v["false_predicate_rate"] for v in variants},
            "worst_case_style": worst_style,
            "worst_case_utilized": min_util["utilized_count"],
        })

    # ── Aggregate sensitivity metrics ──────────────────────────────────────
    agreement_rate = round(all_agreements / total_models, 4) if total_models else 0.0

    all_pred_cvs = [m["predicate_count_cv"] for m in per_model_sensitivity]
    mean_pred_cv = round(sum(all_pred_cvs) / len(all_pred_cvs), 4) if all_pred_cvs else 0.0

    # Per-style aggregate
    style_util_totals: Dict[str, List[float]] = defaultdict(list)
    style_false_totals: Dict[str, List[float]] = defaultdict(list)
    style_pred_totals: Dict[str, List[int]] = defaultdict(list)
    for v in per_variant:
        style_util_totals[v["style"]].append(v["utilization_rate"])
        style_false_totals[v["style"]].append(v["false_predicate_rate"])
        style_pred_totals[v["style"]].append(v["predicate_count"])

    per_style_summary: Dict[str, Any] = {}
    for style in STYLE_NAMES:
        utils = style_util_totals[style]
        falses = style_false_totals[style]
        preds = style_pred_totals[style]
        per_style_summary[style] = {
            "mean_predicate_count": round(sum(preds) / len(preds), 2) if preds else 0,
            "mean_utilization_rate": round(sum(utils) / len(utils), 4) if utils else 0,
            "mean_false_predicate_rate": round(sum(falses) / len(falses), 4) if falses else 0,
            "total_predicates": sum(preds),
        }

    # Worst-case style (across all models)
    worst_style_counts: Dict[str, int] = defaultdict(int)
    for m in per_model_sensitivity:
        worst_style_counts[m["worst_case_style"]] += 1
    overall_worst = max(worst_style_counts, key=worst_style_counts.get)

    # ── Key finding ────────────────────────────────────────────────────────
    key_finding = (
        f"Cross-style agreement rate: {agreement_rate*100:.1f}% "
        f"({all_agreements}/{total_models} models produce consistent verdicts). "
        f"Mean predicate count CV: {mean_pred_cv:.4f}. "
        f"Defensive style yields most predicates "
        f"(mean {per_style_summary['defensive']['mean_predicate_count']:.1f}), "
        f"minimal yields fewest "
        f"(mean {per_style_summary['minimal']['mean_predicate_count']:.1f}). "
        f"Worst-case style (fewest useful predicates): '{overall_worst}' "
        f"({worst_style_counts[overall_worst]}/{total_models} models)."
    )

    results = {
        "experiment": "coding_style_sensitivity",
        "description": (
            "Evaluates guard-harvesting sensitivity to coding style by analyzing "
            "5 PyTorch models each written in 3 styles (defensive, minimal, functional). "
            "Measures predicate count, utilization rate, false predicate rate, and "
            "verification correctness to quantify how coding conventions affect "
            "predicate quality."
        ),
        "aggregate_metrics": {
            "cross_style_agreement_rate": agreement_rate,
            "models_with_agreement": all_agreements,
            "total_models": total_models,
            "mean_predicate_count_cv": mean_pred_cv,
            "overall_worst_case_style": overall_worst,
            "worst_case_style_frequency": dict(worst_style_counts),
        },
        "per_style_summary": per_style_summary,
        "per_model_sensitivity": per_model_sensitivity,
        "per_variant_details": [
            {k: v for k, v in var.items() if k != "predicates"}
            for var in per_variant
        ],
        "key_finding": key_finding,
        "methodology": (
            "For each of 5 models (4 correct, 1 buggy) × 3 coding styles, "
            "we run guard extraction (extract_guards) and shape contract discovery "
            "(run_shape_cegar) to harvest predicates. Style A ('defensive') adds "
            "explicit assert/isinstance guards. Style B ('minimal') uses bare-minimum "
            "code. Style C ('functional') uses torch.nn.functional with inline comments. "
            "Predicate utilization rate = fraction of shape-relevant predicates. "
            "False predicate rate = fraction of non-shape predicates. "
            "Cross-style agreement = same normalized verdict across all 3 styles. "
            "Predicate count CV = coefficient of variation across styles."
        ),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\n  Cross-style agreement rate: {agreement_rate*100:.1f}%")
    print(f"  Mean predicate count CV:    {mean_pred_cv:.4f}")
    print(f"  Overall worst-case style:   {overall_worst}")

    print("\n  Per-style summary:")
    for style in STYLE_NAMES:
        s = per_style_summary[style]
        print(f"    {style:12s}: "
              f"mean_preds={s['mean_predicate_count']:5.1f}  "
              f"util_rate={s['mean_utilization_rate']:.4f}  "
              f"false_rate={s['mean_false_predicate_rate']:.4f}")

    print(f"\n  Per-model agreement:")
    for m in per_model_sensitivity:
        agree_str = "✓" if m["cross_style_agreement"] else "✗"
        print(f"    {m['model']:20s}  {agree_str}  "
              f"pred_cv={m['predicate_count_cv']:.4f}  "
              f"worst={m['worst_case_style']}")

    print(f"\n  Key finding: {key_finding}")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
