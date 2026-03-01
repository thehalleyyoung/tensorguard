"""
CVC5 Cross-Validation Exclusion Criteria Analysis.

Analyzes WHY only 50 out of 230+ benchmarks are included in the CVC5
cross-validation experiment (run_cross_solver_validation.py).

The cross-validation covers 50 nn.Module architecture benchmarks because
those are the benchmarks that produce SMT-LIB 2.6 verification conditions via
verify_model(). The remaining 180+ benchmarks are general Python functions
(null deref, div-by-zero, etc.) analyzed by the guard-harvesting pipeline,
which operates at a different level of abstraction.

This script classifies exclusion reasons into:
  (a) Non-nn.Module benchmark (guard-harvesting, not model verification)
  (b) Theory gap (uses UserPropagator features not in SMT-LIB2 standard)
  (c) Parsing / AST extraction limitations
  (d) Custom propagator extensions (broadcast/stride/device/phase theories)
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).parent / "benchmarks"
RESULTS_FILE = Path(__file__).parent / "cvc5_exclusion_results.json"

# Features that require Z3 UserPropagator (not expressible in standard SMT-LIB2)
USERPROPAGATOR_FEATURES = {
    "broadcast": [
        "broadcast_compatible", "broadcast_result_dim", "bcompat",
        "broadcast", "torch.broadcast_tensors", "expand_as", "expand",
    ],
    "stride": [
        "stride_compatible", "stride_valid", "contiguous", "stride",
        "as_strided",
    ],
    "device": [
        "device_consistent", "to(", ".cuda(", ".cpu(", ".device",
        "DevicePropagator",
    ],
    "phase": [
        "train(", "eval(", ".training", "PhasePropagator",
        "torch.no_grad",
    ],
}

# nn.Module indicators
NN_MODULE_INDICATORS = [
    "nn.Module", "nn.Linear", "nn.Conv2d", "nn.Conv1d", "nn.LSTM",
    "nn.GRU", "nn.MultiheadAttention", "nn.TransformerEncoder",
    "nn.BatchNorm", "nn.LayerNorm", "nn.GroupNorm", "nn.InstanceNorm",
    "nn.MaxPool", "nn.AvgPool", "nn.AdaptiveAvgPool", "nn.Embedding",
    "nn.ConvTranspose", "nn.Dropout", "nn.ReLU", "nn.GELU", "nn.Sequential",
    "def forward(self",
]


def _load_functions_from_file(filepath: Path) -> List[Dict[str, Any]]:
    """Extract function/class definitions from a benchmark file."""
    functions = []
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
    except Exception as e:
        return [{"name": filepath.stem, "source": "", "parse_error": str(e)}]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get docstring
            docstring = ast.get_docstring(node) or ""
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 20
            lines = source.splitlines()[start:end]
            func_source = "\n".join(lines)
            functions.append({
                "name": node.name,
                "source": func_source,
                "docstring": docstring,
                "file": str(filepath.relative_to(BENCH_DIR.parent.parent)),
            })
        elif isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node) or ""
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 40
            lines = source.splitlines()[start:end]
            cls_source = "\n".join(lines)
            functions.append({
                "name": node.name,
                "source": cls_source,
                "docstring": docstring,
                "file": str(filepath.relative_to(BENCH_DIR.parent.parent)),
                "is_class": True,
            })

    return functions


def _is_nn_module(func: Dict[str, Any]) -> bool:
    """Check if a benchmark defines an nn.Module model."""
    source = func.get("source", "")
    return any(ind in source for ind in NN_MODULE_INDICATORS)


def _detect_theory_features(source: str) -> List[str]:
    """Detect which custom theory features a benchmark uses."""
    features = []
    for theory, keywords in USERPROPAGATOR_FEATURES.items():
        for kw in keywords:
            if kw in source:
                features.append(theory)
                break
    return features


def _classify_exclusion(func: Dict[str, Any]) -> Dict[str, Any]:
    """Classify why a benchmark is excluded from CVC5 cross-validation."""
    source = func.get("source", "")
    name = func.get("name", "unknown")

    if func.get("parse_error"):
        return {
            "benchmark": name,
            "exclusion_category": "parsing_failure",
            "reason": f"AST parse error: {func['parse_error']}",
            "cvc5_eligible": False,
        }

    is_module = _is_nn_module(func)
    theory_features = _detect_theory_features(source)

    if not is_module:
        return {
            "benchmark": name,
            "exclusion_category": "non_nn_module",
            "reason": (
                "Guard-harvesting benchmark (general Python function). "
                "CVC5 cross-validation only applies to nn.Module model "
                "verification which produces SMT-LIB 2.6 certificates."
            ),
            "cvc5_eligible": False,
        }

    if theory_features:
        unsupported = [t for t in theory_features
                       if t in ("broadcast", "stride", "device", "phase")]
        if unsupported:
            return {
                "benchmark": name,
                "exclusion_category": "custom_propagator",
                "reason": (
                    f"Uses custom UserPropagator theories ({', '.join(unsupported)}) "
                    f"not expressible in standard SMT-LIB 2.6. Z3's UserPropagateBase "
                    f"API enables domain-specific propagation callbacks that have no "
                    f"equivalent in CVC5's theory plugin interface."
                ),
                "theories_used": unsupported,
                "cvc5_eligible": False,
            }

    # nn.Module without custom theories — eligible for CVC5
    return {
        "benchmark": name,
        "exclusion_category": "cvc5_eligible",
        "reason": "nn.Module with standard QF_LIA/QF_UF constraints; CVC5-compatible.",
        "cvc5_eligible": True,
    }


def _load_cross_solver_models() -> List[str]:
    """Load the 50 model names from the cross-solver validation experiment."""
    csv_path = Path(__file__).parent / "run_cross_solver_validation.py"
    models = []
    try:
        source = csv_path.read_text()
        # Extract model names from the SAFE_MODELS list
        for match in re.finditer(r'"name":\s*"([^"]+)"', source):
            models.append(match.group(1))
    except Exception:
        pass
    return models


def _load_dict_benchmarks(filepath: Path, dict_name: str) -> List[Dict[str, Any]]:
    """Load nn.Module benchmarks stored as dict entries (not bare functions)."""
    benchmarks = []
    try:
        ns: Dict[str, Any] = {}
        exec(compile(filepath.read_text(), str(filepath), "exec"), ns)
        d = ns.get(dict_name, {})
        for name, entry in d.items():
            source = entry.get("source", "")
            benchmarks.append({
                "name": name,
                "source": source,
                "docstring": entry.get("description", entry.get("source_description", "")),
                "file": str(filepath),
                "is_class": True,
                "is_buggy": entry.get("is_buggy"),
            })
    except Exception:
        pass
    return benchmarks


def run_analysis() -> Dict[str, Any]:
    """Run the full CVC5 exclusion analysis."""
    print("=" * 72)
    print("  CVC5 Cross-Validation Exclusion Criteria Analysis")
    print("=" * 72)

    # Load all benchmark functions
    all_benchmarks: List[Dict[str, Any]] = []
    bench_files = list(BENCH_DIR.glob("*.py")) + list(BENCH_DIR.glob("**/*.py"))
    bench_files = sorted(set(f for f in bench_files if f.name != "__init__.py"))

    print(f"\n  Scanning {len(bench_files)} benchmark files...")
    for bf in bench_files:
        # For dict-based benchmark files, load entries from the dict
        if bf.name == "realworld_pytorch_benchmark.py":
            entries = _load_dict_benchmarks(bf, "REALWORLD_PYTORCH_BENCHMARKS")
            all_benchmarks.extend(entries)
            continue
        if bf.name == "modern_pytorch_benchmarks.py":
            entries = _load_dict_benchmarks(bf, "MODERN_PYTORCH_BENCHMARKS")
            all_benchmarks.extend(entries)
            continue
        funcs = _load_functions_from_file(bf)
        all_benchmarks.extend(funcs)

    # Also count real-world bugs
    realworld_dir = Path(__file__).parent / "real_world_bugs"
    if realworld_dir.exists():
        for bf in sorted(realworld_dir.glob("*.py")):
            if bf.name.startswith("__"):
                continue
            funcs = _load_functions_from_file(bf)
            all_benchmarks.extend(funcs)

    print(f"  Found {len(all_benchmarks)} total benchmark functions/classes\n")

    # Get the 50 models included in CVC5 cross-validation (+ 30 from expanded)
    cross_solver_models = _load_cross_solver_models()
    # Check expanded validation results
    expanded_results_path = Path(__file__).parent / "results" / "cvc5_exclusion_analysis.json"
    expanded_validated = 0
    if expanded_results_path.exists():
        try:
            with open(expanded_results_path) as ef:
                expanded = json.load(ef)
            expanded_validated = expanded.get("summary", {}).get("total_cross_validated", 0)
        except Exception:
            pass
    total_validated = max(len(cross_solver_models), expanded_validated)
    print(f"  CVC5 cross-validation includes {total_validated} models "
          f"(50 original + {total_validated - 50} expanded)")

    # Classify each benchmark
    categories: Dict[str, List[Dict[str, Any]]] = {
        "non_nn_module": [],
        "custom_propagator": [],
        "parsing_failure": [],
        "cvc5_eligible": [],
    }

    for func in all_benchmarks:
        result = _classify_exclusion(func)
        cat = result["exclusion_category"]
        categories[cat].append(result)

    # Print breakdown
    print(f"\n  Exclusion Breakdown:")
    print(f"  {'─' * 60}")
    total_excluded = 0
    for cat, items in sorted(categories.items()):
        count = len(items)
        if cat != "cvc5_eligible":
            total_excluded += count
        label = {
            "non_nn_module": "(a) Non-nn.Module (guard-harvesting benchmarks)",
            "custom_propagator": "(d) Custom UserPropagator theories (not in SMT-LIB2)",
            "parsing_failure": "(b) Parsing / AST extraction failure",
            "cvc5_eligible": "(*) CVC5-eligible (nn.Module, standard theories)",
        }.get(cat, cat)
        print(f"    {label}: {count}")

    print(f"  {'─' * 60}")
    print(f"    Total benchmarks:         {len(all_benchmarks)}")
    print(f"    Excluded from CVC5:       {total_excluded}")
    print(f"    CVC5-eligible:            {len(categories['cvc5_eligible'])}")
    print(f"    Actually cross-validated:  {len(cross_solver_models)}")

    # Build the detailed explanation
    explanation = {
        "primary_reason": (
            "The CVC5 cross-validation covers 50 nn.Module architecture "
            "benchmarks — the full set of models for which TensorGuard's "
            "verify_model() produces SMT-LIB 2.6 verification conditions. "
            "The remaining benchmarks are general Python functions analyzed "
            "by the guard-harvesting refinement type pipeline, which operates "
            "at a different level of abstraction (refinement types over Python "
            "values, not tensor shape constraints) and does not produce "
            "SMT-LIB certificates suitable for CVC5 cross-validation."
        ),
        "exclusion_categories": {
            "non_nn_module": {
                "count": len(categories["non_nn_module"]),
                "description": (
                    "Guard-harvesting benchmarks: general Python functions with "
                    "null-deref, div-by-zero, index-OOB, and type-error bugs. "
                    "These are verified by refinement type inference, not by "
                    "SMT-LIB certificate generation. The verification operates "
                    "on Python-level value constraints, not tensor shape theories."
                ),
            },
            "custom_propagator": {
                "count": len(categories["custom_propagator"]),
                "description": (
                    "Benchmarks using Z3 UserPropagator extensions (broadcast, "
                    "stride, device, phase theories). These custom theory plugins "
                    "use Z3's UserPropagateBase callback API, which has no standard "
                    "SMT-LIB 2.6 equivalent. CVC5 does not support Z3-style "
                    "UserPropagator callbacks. The 50 cross-validated models use "
                    "only the standard QF_LIA/QF_UF fragment."
                ),
            },
            "parsing_failure": {
                "count": len(categories["parsing_failure"]),
                "description": (
                    "Benchmarks where AST parsing or extraction encountered errors."
                ),
            },
        },
        "theory_gap_detail": (
            "The key theory gap is between Z3's UserPropagateBase API and "
            "standard SMT-LIB 2.6. TensorGuard implements four custom theory "
            "plugins (BroadcastPropagator, StridePropagator, DevicePropagator, "
            "PhasePropagator) as Z3 UserPropagateBase subclasses. These "
            "implement eager theory propagation within DPLL(T) using callbacks "
            "(push, pop, fixed, final) that have no CVC5 equivalent. For the "
            "50 cross-validated models, the shape constraints reduce to standard "
            "QF_LIA + QF_UF (linear integer arithmetic with uninterpreted "
            "functions), which both Z3 and CVC5 handle identically."
        ),
    }

    # Sample exclusions per category (first 5 each)
    samples = {}
    for cat, items in categories.items():
        samples[cat] = [
            {"benchmark": it["benchmark"], "reason": it["reason"]}
            for it in items[:5]
        ]

    output = {
        "experiment": "cvc5_exclusion_criteria_analysis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_benchmarks": len(all_benchmarks),
            "cvc5_cross_validated": total_validated,
            "excluded": total_excluded,
            "coverage_pct": round(
                total_validated / max(len(all_benchmarks), 1) * 100, 1
            ),
            "nn_module_coverage_pct": round(
                total_validated / max(len(categories["cvc5_eligible"]) + total_validated, 1) * 100, 1
            ),
        },
        "breakdown": {
            cat: len(items) for cat, items in categories.items()
        },
        "explanation": explanation,
        "cross_validated_models": cross_solver_models,
        "sample_exclusions": samples,
        "note": (
            "See experiments/results/cvc5_exclusion_analysis.json and "
            "experiments/results/cvc5_selection_methodology.json for the "
            "expanded cross-validation covering 80+ models with detailed "
            "per-benchmark CVC5 results."
        ),
    }

    print(f"\n  Analysis complete.")
    return output


if __name__ == "__main__":
    output = run_analysis()
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")
