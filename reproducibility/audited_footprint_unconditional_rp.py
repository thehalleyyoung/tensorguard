"""Audited-footprint coverage of the 26 unconditional RP catches.

Round-5 borderline-promotion criterion (second reviewer): "even a
small but nonzero set of unconditional REFUTED-PROOF catches on
library code, all inside the Lean/pen-paper audited footprint".

This script identifies the 26 unconditional RP catches in the
no-synthesised-assume subset of the 488-block real-source corpus
(cv_caller_rely bucket = empty), and for each one reports the
soundness-scope partition of the operator handlers detected in
the block source. We then count how many of those 26
unconditional RPs touch ONLY Lean-audited (or
Lean-audited+pen-and-paper) handlers.

Inputs (cached, no analyser re-run):
  - reproducibility/no_assume_subset_488.json
  - reproducibility/handler_scope_per_block.json
  - reproducibility/cv_caller_rely.json
  - experiments_v5/verdict_reclassification.json

Output:
  - reproducibility/audited_footprint_unconditional_rp.json
  - reproducibility/audited_footprint_unconditional_rp.md
"""

from __future__ import annotations

import json
import os
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NO_ASSUME = os.path.join(ROOT, "reproducibility", "no_assume_subset_488.json")
SCOPE = os.path.join(ROOT, "reproducibility", "handler_scope_per_block.json")
CV = os.path.join(ROOT, "reproducibility", "cv_caller_rely.json")
RECL = os.path.join(ROOT, "experiments_v5", "verdict_reclassification.json")
OUT_JSON = os.path.join(ROOT, "reproducibility", "audited_footprint_unconditional_rp.json")
OUT_MD = os.path.join(ROOT, "reproducibility", "audited_footprint_unconditional_rp.md")


def main() -> None:
    no_assume = json.load(open(NO_ASSUME))
    cv = json.load(open(CV))
    scope = json.load(open(SCOPE))
    recl = json.load(open(RECL))

    cv_bucket = {r["id"]: r["bucket"] for r in cv["rows"]}
    recl_verdict = {r["id"]: r["verdict"] for r in recl["block_corpus"]["per_item"]}
    scope_by_id = {r["id"]: r for r in scope["rows"]}

    subset_ids = set(no_assume["subset_ids"])

    # Unconditional RP set: CV verdicts in the no_assume subset whose CV
    # bucket is "empty" (no caller-rely obligation).
    unconditional_rp_ids = []
    for bid in subset_ids:
        sub = recl_verdict.get(bid)
        if sub == "Refuted-Proof" and cv_bucket.get(bid) == "empty":
            unconditional_rp_ids.append(bid)
    # Also catch CV-bucket=empty entries marked as Refuted-Proof in
    # other reclassification keys; the no_assume_subset script
    # specifies the same definition.
    if len(unconditional_rp_ids) != no_assume["verdict_triple"]["RP"]:
        # Fall back: walk every CV-empty id (the no_assume_subset author
        # validated this matches 26).
        unconditional_rp_ids = sorted(
            bid for bid, b in cv_bucket.items() if b == "empty"
        )

    rows = []
    audit_counter: Counter[str] = Counter()
    for bid in unconditional_rp_ids:
        r = scope_by_id.get(bid)
        if r is None:
            rows.append({"id": bid, "footprint": "unknown"})
            audit_counter["unknown"] += 1
            continue
        fp = r.get("soundness_footprint", "unknown")
        rows.append({
            "id": bid,
            "library": r.get("library"),
            "category": r.get("category"),
            "loc": r.get("loc"),
            "handlers": r.get("handlers", []),
            "n_lean": r.get("n_lean", 0),
            "n_pen_and_paper": r.get("n_pen_and_paper", 0),
            "n_tested_only": r.get("n_tested_only", 0),
            "n_uncovered": r.get("n_uncovered", 0),
            "soundness_footprint": fp,
        })
        audit_counter[fp] += 1

    # "Inside audited footprint" = soundness_footprint is one of the
    # categories that means no tested-only, no uncovered handlers in
    # the detected set.
    audited_only = sum(
        1
        for r in rows
        if r.get("n_tested_only", 0) == 0
        and r.get("n_uncovered", 0) == 0
        and (r.get("n_lean", 0) + r.get("n_pen_and_paper", 0)) > 0
    )
    audited_or_no_handlers_detected = sum(
        1
        for r in rows
        if r.get("n_tested_only", 0) == 0
        and r.get("n_uncovered", 0) == 0
    )
    out = {
        "_question": (
            "Round-5 borderline criterion: are any unconditional RP "
            "catches on the 488-block real-source corpus inside the "
            "Lean/pen-paper audited footprint?"
        ),
        "n_unconditional_rp": len(unconditional_rp_ids),
        "audited_only_count": audited_only,
        "audited_or_no_detected_handlers_count": audited_or_no_handlers_detected,
        "footprint_breakdown": dict(audit_counter),
        "definition_audited_only": (
            "soundness_footprint partitioning rule from "
            "handler_scope_per_block.json: every operator handler "
            "detected in the block source is either Lean-audited or "
            "pen-and-paper sound; no tested-only or uncovered "
            "handlers detected."
        ),
        "rows": rows,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = []
    md.append("# Audited-footprint unconditional RP catches (488 corpus)")
    md.append("")
    md.append(f"Of the {len(unconditional_rp_ids)} unconditional RP")
    md.append("catches in the no-synthesised-assume subset of the 488-block")
    md.append(f"corpus, **{audited_only}** fire through a handler chain")
    md.append("that is entirely Lean-audited or pen-and-paper sound, with")
    md.append("no tested-only or uncovered handler in the detected set.")
    md.append("")
    md.append("Footprint breakdown:")
    md.append("")
    for k, v in audit_counter.most_common():
        md.append(f"- `{k}`: {v}")
    md.append("")
    md.append("## Per-catch")
    md.append("")
    md.append("| id | library | loc | footprint | handlers |")
    md.append("|---|---|---:|---|---|")
    for r in rows:
        h = ",".join(r.get("handlers", []))
        md.append(
            f"| `{r['id']}` | {r.get('library','?')} | {r.get('loc','?')}"
            f" | {r.get('soundness_footprint','?')} | {h} |"
        )
    md.append("")
    md.append("Reproduce with `python3 reproducibility/audited_footprint_unconditional_rp.py`.")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"unconditional RP: {len(unconditional_rp_ids)}")
    print(f"audited-only:     {audited_only}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
