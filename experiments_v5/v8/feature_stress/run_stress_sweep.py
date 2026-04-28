"""
run_stress_sweep.py
===================

Per-feature stress-test sweep for TensorGuard.

Runs all 25 discriminative cases (5 per feature × 5 features) at each
of the six feature levels (L0–L5) and emits:
  • experiments_v5/v8/feature_stress/results.json
  • experiments_v5/v8/feature_stress/SUMMARY.md

Feature levels
--------------
L0  Base fragment only (high_confidence_only=True, no CEGAR, no secondary checks)
L1  + CEGAR contract discovery (max_cegar_iterations=3)
L2  + device-consistency check (check_devices=True)
L3  + train/eval phase check (check_phases=True)
L4  + gradient-flow check (check_gradients=True)
L5  Full (high_confidence_only=False — enables flow-sensitive low-conf bugs)

Case features
-------------
L1_cegar    : 5 attention contract-violation cases (hidden_size % num_heads != 0)
L2_device   : 5 register_buffer CPU vs CUDA mismatch cases
L3_phase    : 5 BatchNorm/Dropout phase-sensitive bug cases
L4_gradient : 5 .detach() gradient-flow broken cases (B1 type)
L5_lowconf  : 5 division-by-zero cases (flow-sensitive, low-confidence)

Run with:
    cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
    PYTHONPATH=. python3.11 experiments_v5/v8/feature_stress/run_stress_sweep.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent          # .../feature_stress/
REPO = ROOT.parent.parent.parent                # tensorguard/
sys.path.insert(0, str(REPO))

from src.api import verify_architecture  # noqa: E402

PREAMBLE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from typing import Optional, Tuple, List, Dict, Any\n"
    "from dataclasses import dataclass\n"
)

# ---------------------------------------------------------------------------
# Feature levels
# ---------------------------------------------------------------------------

LEVELS: List[Dict[str, Any]] = [
    {
        "level": "L0",
        "label": "base fragment (high-confidence only, no CEGAR, no secondary checks)",
        "kwargs": {
            "high_confidence_only": True,
            "max_cegar_iterations": 0,
            "check_devices": False,
            "check_phases": False,
            "check_gradients": False,
        },
    },
    {
        "level": "L1",
        "label": "+ CEGAR contract discovery (max_cegar_iterations=3)",
        "kwargs": {
            "high_confidence_only": True,
            "max_cegar_iterations": 3,
            "check_devices": False,
            "check_phases": False,
            "check_gradients": False,
        },
    },
    {
        "level": "L2",
        "label": "+ device-consistency check (check_devices=True)",
        "kwargs": {
            "high_confidence_only": True,
            "max_cegar_iterations": 3,
            "check_devices": True,
            "check_phases": False,
            "check_gradients": False,
        },
    },
    {
        "level": "L3",
        "label": "+ train/eval phase check (check_phases=True)",
        "kwargs": {
            "high_confidence_only": True,
            "max_cegar_iterations": 3,
            "check_devices": True,
            "check_phases": True,
            "check_gradients": False,
        },
    },
    {
        "level": "L4",
        "label": "+ gradient-flow check (check_gradients=True)",
        "kwargs": {
            "high_confidence_only": True,
            "max_cegar_iterations": 3,
            "check_devices": True,
            "check_phases": True,
            "check_gradients": True,
        },
    },
    {
        "level": "L5",
        "label": "full (high_confidence_only=False — adds low-confidence violations)",
        "kwargs": {
            "high_confidence_only": False,
            "max_cegar_iterations": 3,
            "check_devices": True,
            "check_phases": True,
            "check_gradients": True,
        },
    },
]

# Features in order (matches L1–L5 intended discriminators)
FEATURES = ["L1_cegar", "L2_device", "L3_phase", "L4_gradient", "L5_lowconf"]

# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def load_cases() -> List[Dict[str, Any]]:
    """Load all 25 stress cases from cases/ directory."""
    cases = []
    cases_dir = ROOT / "cases"
    for feature in FEATURES:
        feature_dir = cases_dir / feature
        case_files = sorted(feature_dir.glob("case_*.py"))
        for cf in case_files:
            src = cf.read_text(encoding="utf-8")
            # Extract INPUT_SHAPES
            m = re.search(r"^INPUT_SHAPES\s*=\s*(\{[^}]*\})", src, re.MULTILINE)
            try:
                input_shapes = eval(m.group(1)) if m else {}
            except Exception:
                input_shapes = {}
            cases.append({
                "feature": feature,
                "case_file": str(cf.relative_to(REPO)),
                "case_name": cf.stem,
                "source": PREAMBLE + src,
                "input_shapes": input_shapes,
            })
    return cases


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

def run_case(
    source: str,
    input_shapes: Dict[str, tuple],
    level_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Run verify_architecture on a single case, return verdict dict."""
    captured = io.StringIO()
    t0 = time.perf_counter()
    err = None
    res = None
    try:
        with contextlib.redirect_stderr(captured):
            res = verify_architecture(
                source,
                input_shapes=input_shapes,
                filename="<stress>",
                **level_kwargs,
            )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out: Dict[str, Any] = {
        "elapsed_ms": round(elapsed_ms, 1),
        "exception": err,
    }
    if res is not None:
        out["bug_count"] = int(res.bug_count)
        out["abstained"] = bool(res.abstained)
        bugs = [{"msg": b.message[:120], "severity": b.severity, "confidence": b.confidence}
                for b in res.bugs[:5]]
        out["bugs"] = bugs
    return out


def decide(record: Dict[str, Any]) -> str:
    """Map a run_case result to Refuted / Verified / Abstain."""
    if record.get("exception"):
        return "Abstain"
    if record.get("abstained"):
        return "Abstain"
    if record.get("bug_count", 0) > 0:
        return "Refuted"
    return "Verified"


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading stress cases...")
    cases = load_cases()
    per_feat = ", ".join(f"{sum(1 for c in cases if c['feature']==f)} {f}" for f in FEATURES)
    print(f"  {len(cases)} cases loaded ({per_feat})")

    results_by_level: Dict[str, List[Dict[str, Any]]] = {}
    staircase: List[Dict[str, Any]] = []

    for level_def in LEVELS:
        lvl = level_def["level"]
        label = level_def["label"]
        kwargs = level_def["kwargs"]
        print(f"\n=== {lvl}: {label} ===")
        t0 = time.time()
        level_rows = []
        refuted_total = 0
        refuted_by_feature: Dict[str, int] = {f: 0 for f in FEATURES}

        for case in cases:
            rec = run_case(case["source"], case["input_shapes"], kwargs)
            verdict = decide(rec)
            row = {
                "feature": case["feature"],
                "case_name": case["case_name"],
                "case_file": case["case_file"],
                "verdict": verdict,
                **rec,
            }
            level_rows.append(row)
            if verdict == "Refuted":
                refuted_total += 1
                refuted_by_feature[case["feature"]] += 1

        elapsed = round(time.time() - t0, 1)
        print(f"  [{lvl}] {refuted_total}/25 refuted in {elapsed}s")
        for feat in FEATURES:
            print(f"    {feat}: {refuted_by_feature[feat]}/5")

        results_by_level[lvl] = level_rows
        staircase.append({
            "level": lvl,
            "label": label,
            "feature_kwargs": kwargs,
            "refuted": refuted_total,
            "n": 25,
            "refuted_by_feature": refuted_by_feature,
            "elapsed_s": elapsed,
        })

    # -----------------------------------------------------------------------
    # Honest notes
    # -----------------------------------------------------------------------
    notes = {
        "L1_cegar": (
            "CEGAR (max_cegar_iterations=3) does NOT discriminate in the current "
            "implementation. ShapeCEGARLoop._is_real_bug() always returns False for "
            "the stress cases because shape_env only tracks initial input shapes, not "
            "post-op computed shapes. CEGAR predicates are stored as metadata but never "
            "surfaced as Bug objects. This is a paper-grade observation: the CEGAR "
            "contract-discovery feature is architecturally present but not yet connected "
            "to the verdict pipeline."
        ),
        "L2_device": (
            "check_devices=True enables the post-hoc filter in verify_architecture that "
            "surfaces device_mismatch violations. The buffer-device pass in verify_model "
            "already detects register_buffer CPU/CUDA mismatches; the flag gates whether "
            "those violations reach the caller. All 5 cases discriminate correctly."
        ),
        "L3_phase": (
            "check_phases=True does NOT discriminate in the current implementation. "
            "verify_model's _encode_phase_safety() only registers satisfiable constraints "
            "(Or(TRAIN, EVAL)) for BatchNorm/Dropout, so no phase_violation is ever "
            "generated by the Z3 solver. This is a second paper-grade observation: the "
            "phase-awareness API exists but the phase violation semantics are incomplete."
        ),
        "L4_gradient": (
            "check_gradients=True enables the post-hoc filter that surfaces "
            "gradient_broken violations (triggered by .detach() on model-parameter "
            "tensors in _step_transition). All 5 cases discriminate correctly."
        ),
        "L5_lowconf": (
            "high_confidence_only=False enables a secondary flow-sensitive pass "
            "(src.real_analyzer via analyze()) that detects division-by-zero bugs "
            "from unguarded zero-valued constructor parameters. These are lower-confidence "
            "(heuristic, not Z3-backed) violations. All 5 cases discriminate correctly."
        ),
    }

    out = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": sys.version.split()[0],
            "n_cases": len(cases),
            "n_features": len(FEATURES),
            "n_levels": len(LEVELS),
            "feature_notes": notes,
        },
        "staircase": staircase,
        "case_details": results_by_level,
    }

    out_json = ROOT / "results.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_json}")

    # -----------------------------------------------------------------------
    # SUMMARY.md
    # -----------------------------------------------------------------------
    _write_summary(ROOT / "SUMMARY.md", staircase, notes)
    print(f"Wrote {ROOT / 'SUMMARY.md'}")

    # Print staircase
    print("\n" + "=" * 70)
    print("FEATURE STRESS STAIRCASE (25 cases, 5 per feature)")
    print("=" * 70)
    print(f"{'Level':<5} {'Label':<55} {'Refuted/25'}")
    print("-" * 70)
    for row in staircase:
        lvl = row["level"]
        lbl = row["label"][:53]
        print(f"{lvl:<5} {lbl:<55} {row['refuted']:>2}/25")
    print("=" * 70)
    print("\nPer-feature breakdown:")
    for row in staircase:
        breakdown = "  ".join(
            f"{f.split('_')[0]}({v}/5)"
            for f, v in row["refuted_by_feature"].items()
        )
        print(f"  {row['level']}: {breakdown}")

    print("\nHonest observations:")
    for lvl, note in [("L1", notes["L1_cegar"]), ("L3", notes["L3_phase"])]:
        print(f"  {lvl}: {note[:120]}...")


def _write_summary(path: Path, staircase: List[Dict], notes: Dict) -> None:
    lines = [
        "# TensorGuard Feature Stress Test — SUMMARY",
        "",
        "Per-feature discriminative stress corpus: 25 cases (5 per feature × 5 features).",
        "Each case is designed so that **without** the target feature the verdict is",
        "Verified/Abstain (miss), and **with** the feature enabled the verdict is",
        "Refuted (caught).",
        "",
        "## Staircase Table",
        "",
        "| Level | Feature added | Label | Refuted / 25 | +Δ |",
        "|-------|---------------|-------|--------------|----|",
    ]
    prev = 0
    feat_map = {
        "L0": "—",
        "L1": "CEGAR",
        "L2": "device check",
        "L3": "phase check",
        "L4": "gradient check",
        "L5": "low-confidence",
    }
    for row in staircase:
        lvl = row["level"]
        r = row["refuted"]
        delta = r - prev
        label_short = row["label"][:50]
        feat = feat_map.get(lvl, "—")
        lines.append(
            f"| {lvl} | {feat} | {label_short} | {r}/25 | +{delta} |"
        )
        prev = r
    lines += [
        "",
        "## Per-feature Breakdown",
        "",
        "| Feature | L0 | L1 | L2 | L3 | L4 | L5 |",
        "|---------|----|----|----|----|----|----|",
    ]
    features = list(staircase[0]["refuted_by_feature"].keys())
    for feat in features:
        vals = [str(row["refuted_by_feature"][feat]) for row in staircase]
        lines.append(f"| {feat} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Honest Observations",
        "",
        "### L1 (CEGAR) — Non-Discriminating",
        "",
        notes["L1_cegar"],
        "",
        "### L2 (Device Consistency) — Discriminating ✓",
        "",
        notes["L2_device"],
        "",
        "### L3 (Phase Check) — Non-Discriminating",
        "",
        notes["L3_phase"],
        "",
        "### L4 (Gradient Flow) — Discriminating ✓",
        "",
        notes["L4_gradient"],
        "",
        "### L5 (Low-Confidence) — Discriminating ✓",
        "",
        notes["L5_lowconf"],
        "",
        "## Case Files",
        "",
        "```",
        "experiments_v5/v8/feature_stress/cases/",
        "  L1_cegar/   case_01..05  (attention hidden_size % num_heads != 0)",
        "  L2_device/  case_01..05  (register_buffer CPU vs CUDA input)",
        "  L3_phase/   case_01..05  (BatchNorm/Dropout phase-sensitive bugs)",
        "  L4_gradient/ case_01..05 (.detach() gradient flow broken, B1 type)",
        "  L5_lowconf/ case_01..05  (division-by-zero from zero-valued constructor params)",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
