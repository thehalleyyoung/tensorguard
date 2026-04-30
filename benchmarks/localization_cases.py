#!/usr/bin/env python3
"""Localization case study benchmark.

For each of four bug examples (the original ``examples/shape_bug.py``
plus the three new ``examples/bug_*.py`` files), this script:

  1. Runs TensorGuard's static verifier and records the verbatim
     bug messages and source locations it emits.
  2. Attempts to instantiate the offending ``nn.Module`` and run a
     forward pass on a dummy tensor of the documented input shape;
     captures the resulting Python traceback verbatim.

The goal is to contrast the *symptom* (what PyTorch says at runtime)
with the *cause* (what TensorGuard says statically) on real models.

Output: ``benchmarks/localization_cases.json``.

Reproduce::

    python3.11 benchmarks/localization_cases.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.api import verify_module  # noqa: E402

CASES = [
    {
        "name": "shape_bug.BuggyModel",
        "file": "examples/shape_bug.py",
        "module": "BuggyModel",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "runtime_input": (1, 3, 32, 32),
        "buggy_line_hint": "self.fc = nn.Linear(256, 10)",
    },
    {
        "name": "bug_channel_mismatch.ChannelMismatchModel",
        "file": "examples/bug_channel_mismatch.py",
        "module": "ChannelMismatchModel",
        "input_shapes": {"x": ("B", 3, 16, 16)},
        "runtime_input": (1, 3, 16, 16),
        "buggy_line_hint": "self.conv2 = nn.Conv2d(32, 128, 3, padding=1)",
    },
    {
        "name": "bug_transpose_view.TransposeViewBug",
        "file": "examples/bug_transpose_view.py",
        "module": "TransposeViewBug",
        "input_shapes": {"x": ("B", 8, 64)},
        "runtime_input": (1, 8, 64),
        "buggy_line_hint": "y = y.view(y.size(0), -1)",
    },
    {
        "name": "bug_cat_spatial.CatSpatialMismatch",
        "file": "examples/bug_cat_spatial.py",
        "module": "CatSpatialMismatch",
        "input_shapes": {"x": ("B", 3, 16, 16)},
        "runtime_input": (1, 3, 16, 16),
        "buggy_line_hint": "torch.cat([a, b], dim=1)",
    },
]


def load_module_class(file: str, cls_name: str):
    spec = importlib.util.spec_from_file_location("_bug_mod", REPO_ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def runtime_traceback(file: str, cls_name: str, shape: tuple) -> str:
    try:
        Cls = load_module_class(file, cls_name)
        m = Cls()
        m.eval()
        with torch.no_grad():
            x = torch.randn(*shape)
            m(x)
        return "<no-error: forward succeeded>"
    except Exception:
        # Capture the verbatim traceback message.
        return traceback.format_exc()


def static_findings(file: str, input_shapes: dict) -> dict:
    r = verify_module(str(REPO_ROOT / file), input_shapes=input_shapes)
    bugs = []
    for b in r.bugs:
        bugs.append({
            "line": b.location.line,
            "column": b.location.column,
            "message_first_line": b.message.splitlines()[0],
            "category": b.category.value if hasattr(b.category, "value")
                        else str(b.category),
            "severity": b.severity,
        })
    return {
        "status": r.status,
        "n_bugs": len(r.bugs),
        "duration_ms": round(r.duration_ms, 2),
        "bugs": bugs,
    }


def short_runtime(tb: str) -> str:
    """Pull out the final exception line for the comparison table."""
    lines = [l for l in tb.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def run() -> dict:
    rows = []
    for c in CASES:
        static = static_findings(c["file"], c["input_shapes"])
        rt = runtime_traceback(c["file"], c["module"], c["runtime_input"])
        rows.append({
            "name": c["name"],
            "file": c["file"],
            "buggy_line_hint": c["buggy_line_hint"],
            "static": static,
            "runtime_traceback": rt,
            "runtime_one_line": short_runtime(rt),
        })
    return {"cases": rows}


def main() -> int:
    out = run()
    p = REPO_ROOT / "benchmarks" / "localization_cases.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    for c in out["cases"]:
        print(f"  {c['name']}: static={c['static']['status']} "
              f"({c['static']['n_bugs']} bugs); rt={c['runtime_one_line'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
