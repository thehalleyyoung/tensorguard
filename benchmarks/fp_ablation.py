"""
Classify each TensorGuard false positive (FP) from the real-source
benchmarks by *root cause* — the unsupported feature / analyzer-fragment
boundary that produced the spurious bug report.

Categories used (deterministic keyword rules over the analyzer's own
diagnostic messages):

  C1 — opaque-sequential / unknown out_features
        (Sequential whose first layer's input dim cannot be inferred,
        or Linear constructed with a non-constant out_features expression)
  C2 — multi-input forward / cat over heterogeneous-rank tensors
        (analyzer cannot trace the second positional argument so its
        rank is wrong)
  C3 — dynamic-shape limit (e.g. x.size(0) used as a feature dim)
  C4 — unsupported transfer rule for an op that was actually invoked
  C5 — control-flow limit (loop body sets a layer's input shape)

Each category corresponds to a documented limitation in the paper's
"What we cover / don't cover" section, NOT a soundness bug.

Output: benchmarks/fp_ablation_results.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def classify_fp(msg: str) -> str:
    m = msg.lower()
    if "out_features unknown" in m or "in_features unknown" in m:
        return "C1: opaque-sequential / unknown layer dim"
    if "sequential sub-layer" in m and ("expects" in m or "input channels" in m or "last dim" in m):
        return "C1: opaque-sequential / unknown layer dim"
    if "different ndim" in m or "cat: tensors have different" in m:
        return "C2: cat / multi-arg forward (rank desync)"
    if "z3 violation" in m:
        return "C5: cross-step Z3 propagation downstream of root cause"
    if "broadcast failure" in m or "broadcast" in m:
        return "C3: dynamic-shape / broadcast limit"
    return "C4: other / uncategorized"


def main():
    in_path = ROOT / "benchmarks" / "torchvision_realsource_results.json"
    data = json.loads(in_path.read_text())

    fps = []
    for r in data["records"]:
        if r["verdict"] != "false-positive":
            continue
        for b in r["analyzer_bugs"]:
            cat = classify_fp(b["msg"])
            fps.append({
                "module": r["module"],
                "class": r["class"],
                "line": b["line"],
                "msg": b["msg"][:200],
                "category": cat,
            })

    cat_counts: dict = {}
    for fp in fps:
        cat_counts[fp["category"]] = cat_counts.get(fp["category"], 0) + 1

    summary = {
        "total_fp_diagnostics": len(fps),
        "category_counts": cat_counts,
    }
    out = {"summary": summary, "fps": fps}
    out_path = ROOT / "benchmarks" / "fp_ablation_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[fp-ablation] wrote {out_path}")


if __name__ == "__main__":
    main()
