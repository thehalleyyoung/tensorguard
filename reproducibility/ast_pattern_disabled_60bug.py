#!/usr/bin/env python3
"""Run the 60-bug corpus with the AST-pattern verification path disabled.

The 'AST-pattern path' in verify_architecture is the heuristic
flow-sensitive analyser (analyze_source from real_analyzer) that runs in
parallel with the main operator-dispatch / Z3-constraint path.  Setting
high_confidence_only=True disables it.  This isolates the operator-rule
contribution from the pattern-matching contribution.

Output: reproducibility/ast_pattern_disabled_60bug.json + .md
"""
from __future__ import annotations
import json, os, sys, time, io, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture

CORPUS = ROOT / "experiments_v5" / "v5_bug_corpus.jsonl"
OUT_JSON = ROOT / "reproducibility" / "ast_pattern_disabled_60bug.json"
OUT_MD = ROOT / "reproducibility" / "ast_pattern_disabled_60bug.md"


def score(src: str, use_ast_pattern: bool) -> dict:
    t0 = time.perf_counter()
    captured = io.StringIO()
    err = None
    res = None
    try:
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            res = verify_architecture(
                src,
                max_cegar_iterations=0,
                high_confidence_only=not use_ast_pattern,
            )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    elapsed = (time.perf_counter() - t0) * 1000
    if err or res is None:
        return {"bucket": "Abstain", "elapsed_ms": elapsed}
    if res.abstained:
        return {"bucket": "Abstain", "elapsed_ms": elapsed}
    if res.bug_count > 0:
        return {"bucket": "Refuted", "elapsed_ms": elapsed,
                "bugs": [b.message[:150] for b in res.bugs[:3]]}
    return {"bucket": "Verified", "elapsed_ms": elapsed}


def main():
    records = [json.loads(l) for l in CORPUS.open()]
    print(f"Scoring {len(records)} bug repros (full pipeline vs. AST-pattern-disabled)...")

    full_results = []
    disabled_results = []
    for rec in records:
        repro_path = ROOT / rec["repro_file"]
        if not repro_path.exists():
            full_results.append({"id": rec["id"], "bucket": "Abstain"})
            disabled_results.append({"id": rec["id"], "bucket": "Abstain"})
            continue
        src = repro_path.read_text()
        r_full = score(src, use_ast_pattern=True)
        r_dis = score(src, use_ast_pattern=False)
        full_results.append({"id": rec["id"], "category": rec.get("category"), **r_full})
        disabled_results.append({"id": rec["id"], "category": rec.get("category"), **r_dis})

    full_rp = sum(1 for r in full_results if r["bucket"] == "Refuted")
    dis_rp = sum(1 for r in disabled_results if r["bucket"] == "Refuted")
    # Regressions: full detected but disabled missed
    regressions = [r for r, d in zip(full_results, disabled_results)
                   if r["bucket"] == "Refuted" and d["bucket"] != "Refuted"]

    out = {
        "full_rp": full_rp,
        "disabled_rp": dis_rp,
        "n_total": len(records),
        "regressions": regressions,
        "full_results": full_results,
        "disabled_results": disabled_results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    md = f"""# AST-pattern-disabled 60-bug corpus run

## Setup
- Full pipeline: operator-dispatch + flow-sensitive AST-pattern analyser
- Disabled: operator-dispatch only (high_confidence_only=True suppresses the
  parallel flow-sensitive path)

## Results
| Mode | RP / 60 |
|------|---------|
| Full pipeline | {full_rp}/60 |
| AST-pattern path disabled | {dis_rp}/60 |

## Analysis
- Bugs caught by full pipeline but NOT by operator-dispatch alone: {len(regressions)}/60
- These are attributable to the parallel AST-pattern path.
- Operator-dispatch-only contribution: {dis_rp}/60

## Regressions (full-pipeline caught, disabled missed)
"""
    for r in regressions:
        md += f"- {r['id']} ({r.get('category','?')})\n"
    OUT_MD.write_text(md)
    print(f"\nFull pipeline: {full_rp}/60 RP")
    print(f"AST-pattern disabled: {dis_rp}/60 RP")
    print(f"Regressions (AST-pattern contribution): {len(regressions)}")

if __name__ == "__main__":
    main()
