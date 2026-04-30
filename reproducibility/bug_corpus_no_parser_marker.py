#!/usr/bin/env python3.11
"""Round-5 reviewer Q3: 60-bug RP with the AST-pattern path AND the
``[MODEL_CHECK] No nn.Module subclass found in source`` parser-failure
marker excluded, so the residual catches are only those produced by
the rule-driven symbolic analyser running over a parsed
``nn.Module``.

Configurations re-counted on every bug:

  (A) full pipeline                                     -> reference
  (B) AST-pattern path disabled (high_confidence_only)  -> reference
  (C) AST-pattern path disabled AND parser-failure
      marker excluded from the catch count              -> THIS

Note: ``parser_marker_only`` is the new bucket -- the bug returned
non-empty bugs but every bug message starts with ``[MODEL_CHECK]
No nn.Module subclass found in source``.  Round-4 reported the
parser-marker as a Refuted-Proof; round 5 separates it out so the
``rule-driven only`` column is auditable.

Output:
  reproducibility/bug_corpus_no_parser_marker.json
  reproducibility/bug_corpus_no_parser_marker.md
"""
from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture

CORPUS = ROOT / "experiments_v5" / "v5_bug_corpus.jsonl"
OUT_JSON = ROOT / "reproducibility" / "bug_corpus_no_parser_marker.json"
OUT_MD = ROOT / "reproducibility" / "bug_corpus_no_parser_marker.md"

PARSER_MARKER = "[MODEL_CHECK] No nn.Module subclass found in source"


def _score(src: str, use_ast_pattern: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cap = io.StringIO()
    err = None
    res = None
    try:
        with contextlib.redirect_stderr(cap), contextlib.redirect_stdout(cap):
            res = verify_architecture(
                src,
                max_cegar_iterations=0,
                high_confidence_only=not use_ast_pattern,
            )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    elapsed = (time.perf_counter() - t0) * 1000
    if err or res is None:
        return {"bucket": "Abstain", "elapsed_ms": elapsed,
                "bugs": [], "err": err}
    bugs_msgs = [b.message for b in res.bugs]
    if res.abstained:
        return {"bucket": "Abstain", "elapsed_ms": elapsed,
                "bugs": bugs_msgs}
    if res.bug_count > 0:
        only_parser = all(PARSER_MARKER in m for m in bugs_msgs) and \
                      len(bugs_msgs) > 0
        return {"bucket": "Refuted", "elapsed_ms": elapsed,
                "bugs": bugs_msgs, "parser_marker_only": only_parser}
    return {"bucket": "Verified", "elapsed_ms": elapsed, "bugs": []}


def main() -> int:
    records = [json.loads(l) for l in CORPUS.open() if l.strip()]
    print(f"Scoring {len(records)} bug repros (Q3 ablation)...")

    rows = []
    rp_full = rp_ast_dis = rp_rule_driven = 0
    n_total = len(records)
    parser_only_full = parser_only_dis = 0

    for rec in records:
        repro_path = ROOT / rec["repro_file"]
        if not repro_path.exists():
            rows.append({"id": rec["id"], "category": rec.get("category"),
                         "missing": True})
            continue
        src = repro_path.read_text()
        a = _score(src, use_ast_pattern=True)
        b = _score(src, use_ast_pattern=False)
        if a["bucket"] == "Refuted":
            rp_full += 1
            if a.get("parser_marker_only"):
                parser_only_full += 1
        if b["bucket"] == "Refuted":
            rp_ast_dis += 1
            if b.get("parser_marker_only"):
                parser_only_dis += 1
            else:
                rp_rule_driven += 1
        rows.append({
            "id": rec["id"], "category": rec.get("category"),
            "full_bucket": a["bucket"],
            "full_parser_marker_only": a.get("parser_marker_only", False),
            "ast_disabled_bucket": b["bucket"],
            "ast_disabled_parser_marker_only": b.get(
                "parser_marker_only", False),
            "ast_disabled_first_real_bug": next(
                (m[:140] for m in b.get("bugs", [])
                 if PARSER_MARKER not in m), ""),
        })

    out = {
        "_question": (
            "R5-Q3: with the AST-pattern path disabled AND the "
            "[MODEL_CHECK] No nn.Module subclass found parser-failure "
            "marker excluded from the catch count, what fraction of "
            "the 60-bug corpus does the rule-driven symbolic analyser "
            "still catch?"
        ),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "n_total": n_total,
        "rp_full_pipeline": rp_full,
        "rp_full_parser_marker_only": parser_only_full,
        "rp_ast_pattern_disabled": rp_ast_dis,
        "rp_ast_disabled_parser_marker_only": parser_only_dis,
        "rp_rule_driven_only": rp_rule_driven,
        "interpretation": (
            f"On the {n_total}-bug corpus the full pipeline returns "
            f"Refuted-Proof on {rp_full}, of which "
            f"{parser_only_full} are powered exclusively by the "
            f"parser-failure marker (the bug is a script-level repro "
            f"that does not contain an ``nn.Module`` subclass at all). "
            f"With the AST-pattern path disabled the count is "
            f"{rp_ast_dis}, with {parser_only_dis} parser-marker-only. "
            f"The residual rule-driven RP count -- the analyser's "
            f"contribution under both ablations -- is "
            f"{rp_rule_driven}/{n_total} "
            f"({(rp_rule_driven/n_total*100):.1f}%).  The remaining "
            f"bugs are caught by the AST-pattern path (rounds where "
            f"that path is enabled) or by the parser-failure marker "
            f"recognising that the script-level repro is not a "
            f"reasoning subject for the calculus.  This is the most "
            f"diagnostic single number for the calculus's contribution "
            f"on the curated corpus."
        ),
        "per_bug": rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    md = [
        "# 60-bug RP with parser-failure marker excluded (round 5 Q3)",
        "",
        "## Configurations",
        "",
        "  * (A) Full pipeline (operator dispatch + AST-pattern + parser marker)",
        "  * (B) AST-pattern path disabled",
        "  * (C) AST-pattern path disabled AND parser-failure marker excluded",
        "        (rule-driven only)",
        "",
        "## Result",
        "",
        f"| Configuration | RP / {n_total} | parser-marker-only | rule-driven |",
        f"|---|---|---|---|",
        f"| (A) full pipeline | {rp_full} | {parser_only_full} | {rp_full - parser_only_full} |",
        f"| (B) AST-pattern path disabled | {rp_ast_dis} | {parser_only_dis} | {rp_rule_driven} |",
        f"| (C) AST-pattern disabled + parser-marker excluded (rule-driven only) | {rp_rule_driven} | -- | {rp_rule_driven} |",
        "",
        "## Plain reading",
        "",
        f"The rule-driven symbolic analyser catches "
        f"**{rp_rule_driven} / {n_total} = "
        f"{(rp_rule_driven/n_total*100):.1f}%** of the 60-bug corpus "
        f"once both the AST-pattern path and the parser-failure marker "
        f"are removed.  The high headline number on the curated "
        f"corpus is therefore powered substantially by the AST-pattern "
        f"path (which contributes "
        f"{rp_full - rp_ast_dis} extra catches) and by the "
        f"parser-failure marker (which contributes "
        f"{parser_only_dis} catches under (B)).  The calculus is the "
        f"correctness substrate that justifies which catches are "
        f"sound, but the recognition of a buggy fragment routinely "
        f"goes through one of the other two paths on this corpus.",
        "",
        "## Per-bug detail",
        "",
        "| id | cat | full | full_parser_only | ast_dis | ast_dis_parser_only | first_real_bug_under_(B) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("missing"):
            md.append(f"| {r['id']} | {r.get('category','?')} | MISSING | | | | |")
            continue
        md.append(
            f"| {r['id']} | {r.get('category','?')} | "
            f"{r['full_bucket']} | {r['full_parser_marker_only']} | "
            f"{r['ast_disabled_bucket']} | "
            f"{r['ast_disabled_parser_marker_only']} | "
            f"{r['ast_disabled_first_real_bug']} |")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\nFull RP: {rp_full} (parser-only {parser_only_full})")
    print(f"AST-disabled RP: {rp_ast_dis} (parser-only {parser_only_dis})")
    print(f"Rule-driven only: {rp_rule_driven}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
