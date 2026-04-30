#!/usr/bin/env python3
"""Round-2 Q5: soundness-scope partition of the 6/15 post-freeze RP fires.

Extends postfreeze_5catches_handler_scope.py to include rb_uf_010
(the dtype-class FP) and report what fraction of the 6 fires
traverse only the 35-handler in-soundness footprint vs. at least
one tested-only or uncovered handler.

Output:
    reproducibility/post_freeze_in_soundness_scope.json
    reproducibility/post_freeze_in_soundness_scope.md
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

OUT_JSON = os.path.join(ROOT, "reproducibility", "post_freeze_in_soundness_scope.json")
OUT_MD   = os.path.join(ROOT, "reproducibility", "post_freeze_in_soundness_scope.md")

# ── Data ──────────────────────────────────────────────────────────────────────
# Source: postfreeze_5catches_handler_scope.md + rb_uf_010 from
# post_freeze_n15_precision_recall.md.
#
# Each entry: (bug_id, tg_label, [handler, scope], in_soundness_only)
#
# "in_soundness_only" = True  iff every triggered handler is Lean-audited
#                              or pen-and-paper.
#             = False iff at least one triggered handler is tested-only
#                              or uncovered.
FIRES = [
    {
        "bug_id":    "rb_pf_001",
        "tg_label":  "TP",
        "bug_class": "shape",
        "handlers": [
            {"name": "linear",           "scope": "Lean-audited"},
            {"name": "mul",              "scope": "uncovered"},
        ],
        "in_soundness_only": False,  # mul is uncovered
    },
    {
        "bug_id":    "rb_pf_003",
        "tg_label":  "TP",
        "bug_class": "shape",
        "handlers": [
            {"name": "expand",    "scope": "Lean-audited"},
            {"name": "add",       "scope": "uncovered"},
            {"name": "einsum",    "scope": "pen-and-paper"},
            {"name": "unsqueeze", "scope": "tested-only"},
        ],
        "in_soundness_only": False,  # add uncovered, unsqueeze tested-only
    },
    {
        "bug_id":    "rb_pf_004",
        "tg_label":  "TP",
        "bug_class": "shape",
        "handlers": [
            {"name": "linear",  "scope": "Lean-audited"},
            {"name": "softmax", "scope": "tested-only"},
        ],
        "in_soundness_only": False,  # softmax tested-only
    },
    {
        "bug_id":    "rb_uf_008",
        "tg_label":  "TP",
        "bug_class": "shape",
        "handlers": [
            {"name": "view",    "scope": "Lean-audited"},
            {"name": "reshape", "scope": "Lean-audited"},
            {"name": "mul",     "scope": "uncovered"},
        ],
        "in_soundness_only": False,  # mul uncovered
    },
    {
        "bug_id":    "rb_uf_012",
        "tg_label":  "TP",
        "bug_class": "shape",
        "handlers": [
            {"name": "view",    "scope": "Lean-audited"},
            {"name": "permute", "scope": "Lean-audited"},
            {"name": "conv2d",  "scope": "Lean-audited"},
            {"name": "mul",     "scope": "uncovered"},
        ],
        "in_soundness_only": False,  # mul uncovered
    },
    {
        "bug_id":    "rb_uf_010",
        "tg_label":  "FP",
        "bug_class": "dtype",
        "handlers": [
            {"name": "device_mismatch", "scope": "tested-only"},
        ],
        "in_soundness_only": False,  # device_mismatch tested-only
    },
]

def run():
    n_total   = len(FIRES)
    n_in_only = sum(1 for f in FIRES if f["in_soundness_only"])
    n_mixed   = n_total - n_in_only

    out = {
        "n_fires": n_total,
        "n_in_soundness_only": n_in_only,
        "n_mixed": n_mixed,
        "fires": FIRES,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    rows = "\n".join(
        "| {} | {} | {} | {} |".format(
            fire["bug_id"],
            fire["tg_label"],
            ", ".join(f"{h['name']}({h['scope']})" for h in fire["handlers"]),
            "in-soundness only" if fire["in_soundness_only"] else "mixed (tested-only/uncovered)",
        )
        for fire in FIRES
    )

    md = f"""# Soundness-scope partition of the 6/15 post-freeze RP fires

## Headline

| Category | Count |
|---|---:|
| in-soundness only (all handlers Lean-audited or pen-and-paper) | {n_in_only} |
| mixed (at least one tested-only or uncovered handler) | {n_mixed} |
| **Total fires** | **{n_total}** |

## Per-fire breakdown

| bug_id | TG label | triggered handlers (scope) | scope category |
|---|---|---|---|
{rows}

## Interpretation

All {n_total} post-freeze RP fires traverse the mixed scope: each touches at
least one tested-only or uncovered handler in addition to any Lean-audited
ones.  None fires exclusively through the 35-handler in-soundness footprint
(28 Lean-audited + 7 pen-and-paper).  A reader comparing the headline
6/15 fire count against Theorem thm:ag-sound should note that the
compositional guarantee covers the Lean-audited operators in the trace
but not the mul/add/softmax/device-mismatch handlers that co-fire.

## Paper claim (Q5)

Round-2 Q5 asks what fraction of the 6/15 fires traverse only the
35-handler in-soundness footprint.  This artefact answers: 0/6 (0%)
fire exclusively through in-soundness handlers; all 6/6 touch at least
one tested-only or uncovered handler.  The post-freeze headline therefore
does not directly validate the formal guarantee fragment.
"""
    with open(OUT_MD, "w") as f:
        f.write(md)

    print(f"in-soundness only: {n_in_only}/{n_total}")
    print(f"mixed: {n_mixed}/{n_total}")
    print(f"Written: {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    run()
