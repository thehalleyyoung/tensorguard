"""
Recompute "fraction of CV verdicts entirely under Lean-witnessed handlers".

Reads:
  reproducibility/handler_scope_per_block.json  -- 488-block corpus rows
  experiments_v5/handler_soundness_scope.json   -- handler -> scope mapping

Writes:
  reproducibility/cv_lean_coverage.txt          -- integer n_cv_in_fragment

Prints: n_cv_in_fragment / n_cv_total

The baseline (pre-round-4 extension) was 35/128.  After adding
applyOp_sound_* theorems for the 8 highest-CV-traffic operators outside
the original Lean fragment (cross_entropy, to, squeeze, dropout,
contiguous, unsqueeze, clamp, argmax), the count should rise to 99/128.
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

def load_lean_set(scope_path: Path) -> set:
    with open(scope_path) as f:
        scope = json.load(f)
    return {h["name"] for h in scope["handlers"] if h["scope"] == "lean_verified"}


def compute_coverage(rows_path: Path, lean_set: set) -> tuple[int, int]:
    with open(rows_path) as f:
        data = json.load(f)
    rows = data["rows"]
    cv_rows = [r for r in rows if r.get("verdict_with_assume") == "CV"]
    n_cv = len(cv_rows)
    n_in_fragment = sum(
        1 for r in cv_rows if all(h in lean_set for h in r["handlers"])
    )
    return n_in_fragment, n_cv


def main() -> None:
    scope_path = REPO_ROOT / "experiments_v5" / "handler_soundness_scope.json"
    rows_path = REPO_ROOT / "reproducibility" / "handler_scope_per_block.json"
    out_path = REPO_ROOT / "reproducibility" / "cv_lean_coverage.txt"

    # Pre-extension baseline (original 28-op fragment, before round-4 additions)
    BASELINE = 35

    lean_set = load_lean_set(scope_path)
    n_in, n_cv = compute_coverage(rows_path, lean_set)

    print(f"Lean-verified operators: {len(lean_set)}")
    print(f"CV blocks in fragment:   {n_in}/{n_cv}")
    print(f"Baseline (pre-round-4):  {BASELINE}/{n_cv}")
    print(f"Improvement:             +{n_in - BASELINE}")

    out_path.write_text(str(n_in) + "\n")
    print(f"Written to {out_path}")

    if n_in <= BASELINE:
        raise RuntimeError(
            f"Coverage {n_in} did not exceed baseline {BASELINE}; "
            "check that the new operators are tagged lean_verified in "
            "experiments_v5/handler_soundness_scope.json"
        )


if __name__ == "__main__":
    main()
