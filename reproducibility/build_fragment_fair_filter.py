"""Deterministic 60→34 fragment-fair filter with per-bug audit CSV.

Materialises the deterministic filter rule that maps the 60-bug historical
corpus to the 34-bug fragment-fair Pytea head-to-head, and emits a single
appendix-grade CSV with one row per bug containing:

    bug_id, included_in_34, exclusion_reason, tg_verdict, pytea_verdict

Inclusion rule (fragment-fair):
    A bug is included iff its primary failing operator is implemented in
    Pytea's 2022 TypeScript operator catalogue (packages/pytea/src/ts/index.ts
    at commit c536515), as recorded in BUG_MODERN_MAP in
    experiments_v5/v8/build_modern_subset.py.

Exclusion reasons (closed enumeration):
    "pytorch2x_op_no_pytea_handler"  — op is PyTorch ≥2.x (SDPA, MHA); no Pytea handler exists
    "op_not_in_pytea_catalogue"      — op exists across versions but has no Pytea TS handler

Running this script also prints the McNemar 2×2 contingency table and the
32/34 vs 25/34 headline counts.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_MODERN_SUBSET = os.path.join(_REPO, "experiments_v5", "v8", "build_modern_subset.py")
_TG_BASELINE = os.path.join(_REPO, "experiments_v5", "v5_benchmark_results.json")
_PYTEA_BASELINE = os.path.join(_REPO, "experiments_v5", "pytea_baseline_results.json")
_OUT_CSV = os.path.join(_REPO, "reproducibility", "fragment_fair_audit.csv")

# Closed enumeration of exclusion reasons.
EXCLUSION_REASON_PYTORCH2X = "pytorch2x_op_no_pytea_handler"
EXCLUSION_REASON_NOT_IN_CATALOGUE = "op_not_in_pytea_catalogue"
VALID_EXCLUSION_REASONS = frozenset({EXCLUSION_REASON_PYTORCH2X, EXCLUSION_REASON_NOT_IN_CATALOGUE})

# Bugs whose primary op is a PyTorch ≥2.x feature absent from Pytea's 2022 catalogue.
_PYTORCH2X_BUG_IDS = frozenset({"bug_001", "bug_012"})


def _load_bug_modern_map() -> dict:
    """Parse BUG_MODERN_MAP from build_modern_subset.py without running its I/O side-effects."""
    src = open(_BUILD_MODERN_SUBSET).read()
    # Execute only up to the first file-read statement (after the constant definition).
    stop_marker = "\n# Load TG and Pytea verdicts"
    safe_src = src.split(stop_marker)[0] if stop_marker in src else src
    g: dict = {"__file__": _BUILD_MODERN_SUBSET}
    exec(compile(safe_src, _BUILD_MODERN_SUBSET, "exec"), g)
    return g["BUG_MODERN_MAP"]


def _load_tg_verdicts() -> dict[str, str]:
    with open(_TG_BASELINE) as f:
        data = json.load(f)
    return {e["id"]: e["bucket"] for e in data["bug_corpus"]["per_input"]}


def _load_pytea_verdicts() -> dict[str, str]:
    with open(_PYTEA_BASELINE) as f:
        data = json.load(f)
    out: dict[str, str] = {}
    for e in data["bug_corpus"]["per_input"]:
        m = re.match(r"(bug_\d+)", e["id"])
        if m:
            out[m.group(1)] = e["verdict"]
    return out


def _exclusion_reason(bug_id: str) -> str:
    if bug_id in _PYTORCH2X_BUG_IDS:
        return EXCLUSION_REASON_PYTORCH2X
    return EXCLUSION_REASON_NOT_IN_CATALOGUE


def build_audit_rows() -> list[dict]:
    bug_modern_map = _load_bug_modern_map()
    tg_verdicts = _load_tg_verdicts()
    pytea_verdicts = _load_pytea_verdicts()

    rows = []
    for bug_id in sorted(bug_modern_map.keys()):
        modern, _primary_op, _note = bug_modern_map[bug_id]
        included = bool(modern)
        exclusion_reason = "" if included else _exclusion_reason(bug_id)
        tg_verdict = tg_verdicts.get(bug_id, "N/A")
        pytea_verdict = pytea_verdicts.get(bug_id, "N/A")
        rows.append({
            "bug_id": bug_id,
            "included_in_34": str(included),
            "exclusion_reason": exclusion_reason,
            "tg_verdict": tg_verdict,
            "pytea_verdict": pytea_verdict,
        })
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    fieldnames = ["bug_id", "included_in_34", "exclusion_reason", "tg_verdict", "pytea_verdict"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_mcnemar_table(rows: list[dict]) -> None:
    included = [r for r in rows if r["included_in_34"] == "True"]
    tg_refuted = sum(1 for r in included if r["tg_verdict"] == "Refuted")
    pytea_refuted = sum(1 for r in included if r["pytea_verdict"] == "Refuted")
    both = sum(1 for r in included if r["tg_verdict"] == "Refuted" and r["pytea_verdict"] == "Refuted")
    tg_only = sum(1 for r in included if r["tg_verdict"] == "Refuted" and r["pytea_verdict"] != "Refuted")
    pytea_only = sum(1 for r in included if r["tg_verdict"] != "Refuted" and r["pytea_verdict"] == "Refuted")
    neither = sum(1 for r in included if r["tg_verdict"] != "Refuted" and r["pytea_verdict"] != "Refuted")
    n = len(included)

    print(f"Fragment-fair subset: n={n}")
    print(f"  TG catches:    {tg_refuted}/{n}")
    print(f"  Pytea catches: {pytea_refuted}/{n}")
    print()
    print("McNemar 2×2 contingency table (off-diagonal cells drive the test):")
    print(f"  Both refute:  {both}")
    print(f"  TG only:      {tg_only}")
    print(f"  Pytea only:   {pytea_only}")
    print(f"  Neither:      {neither}")
    print()

    excluded = [r for r in rows if r["included_in_34"] == "False"]
    print(f"Excluded: {len(excluded)} bugs")
    reason_counts: dict[str, int] = {}
    for r in excluded:
        reason_counts[r["exclusion_reason"]] = reason_counts.get(r["exclusion_reason"], 0) + 1
    for reason, count in sorted(reason_counts.items()):
        print(f"  {reason}: {count}")


def main() -> None:
    rows = build_audit_rows()
    write_csv(rows, _OUT_CSV)
    print(f"Wrote {len(rows)} rows to {_OUT_CSV}")
    print()
    print_mcnemar_table(rows)
    print(f"CSV path: {_OUT_CSV}")


if __name__ == "__main__":
    main()
