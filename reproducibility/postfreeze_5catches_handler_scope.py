#!/usr/bin/env python3.11
"""Round-2 Q4 follow-up: handler scope of the 5/15 post-freeze catches.

Reviewer Q4 asks: of the 5/15 catches in the post-freeze unfiltered
sample, how many fire through a tested-only handler vs.
a Lean-audited or pen-and-paper handler?  This bears directly on
the calibrated soundness scope of the empirical headline.

Method:
  1. The 5 catches are: rb_pf_001, rb_pf_003, rb_pf_004, rb_uf_008,
     rb_uf_012 (per `reproducibility/postfreeze_overlap_matrix.md`).
  2. For each catch we load the corresponding repro file from
     experiments_v5/v8/real_bugs_*/, run the v5 analyser, and read
     out the operator handler that produced the bug verdict (if the
     bug carries a handler-name annotation), falling back to a
     source-token scan for the bug-marker line.
  3. Each handler is then classified against
     `experiments_v5/handler_soundness_scope.json` into one of
     {Lean-audited, pen-and-paper, tested-only, uncovered}.

Output:
  reproducibility/postfreeze_5catches_handler_scope.json
  reproducibility/postfreeze_5catches_handler_scope.md
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

CATCHES = [
    ("rb_pf_001_diffusers_longcat_ffmult",
     "experiments_v5/v8/real_bugs_postfreeze/rb_pf_001_diffusers_longcat_ffmult.py"),
    ("rb_pf_003_peft_lora_moe_swap",
     "experiments_v5/v8/real_bugs_postfreeze/rb_pf_003_peft_lora_moe_swap.py"),
    ("rb_pf_004_routerparallel_topk",
     "experiments_v5/v8/real_bugs_postfreeze/rb_pf_004_routerparallel_topk.py"),
    ("rb_uf_008_wan_vae_decoder",
     "experiments_v5/v8/real_bugs_unfiltered/rb_uf_008_wan_vae_decoder.py"),
    ("rb_uf_012_hunyuan_vae_nan_branch",
     "experiments_v5/v8/real_bugs_unfiltered/rb_uf_012_hunyuan_vae_nan_branch.py"),
]

SCOPE_PATH = os.path.join(ROOT, "experiments_v5/handler_soundness_scope.json")
OUT_JSON = os.path.join(
    ROOT, "reproducibility", "postfreeze_5catches_handler_scope.json"
)
OUT_MD = os.path.join(
    ROOT, "reproducibility", "postfreeze_5catches_handler_scope.md"
)


def _load_scope() -> Dict[str, str]:
    """Return handler -> {Lean-audited, pen-and-paper, tested-only}."""
    with open(SCOPE_PATH) as fh:
        d = json.load(fh)
    out: Dict[str, str] = {}
    scope_remap = {
        "lean_verified": "Lean-audited",
        "lean_audited": "Lean-audited",
        "pen_and_paper": "pen-and-paper",
        "pen_paper": "pen-and-paper",
        "tested_only": "tested-only",
    }
    for h in d.get("handlers", []):
        name = h.get("name") or h.get("handler")
        sc = h.get("scope")
        if not name or not sc:
            continue
        out[name] = scope_remap.get(sc, sc)
    return out


def _detect_handlers(source: str) -> List[str]:
    # Cheap source-token scan, identical surface to handler_scope_per_block.py.
    # We only need a small subset here for the 5 catches; over-attribute is OK.
    patterns = {
        "view": [r"\.view\("],
        "reshape": [r"\.reshape\("],
        "permute": [r"\.permute\("],
        "transpose": [r"\.transpose\(", r"\.t\(\)"],
        "expand": [r"\.expand\("],
        "linear": [r"\bLinear\b", r"F\.linear\b"],
        "matmul": [r"\.matmul\b", r"@\s*self\.", r"torch\.matmul\b"],
        "bmm": [r"\.bmm\b"],
        "conv2d": [r"\bConv2d\b", r"F\.conv2d\b"],
        "conv1d": [r"\bConv1d\b"],
        "embed": [r"\bEmbedding\b"],
        "layer_norm": [r"\bLayerNorm\b"],
        "rms_norm": [r"\bRMSNorm\b"],
        "scaled_dot_product_attention": [r"scaled_dot_product_attention"],
        "softmax": [r"\.softmax\b", r"\bSoftmax\b"],
        "cat": [r"torch\.cat\b"],
        "stack": [r"torch\.stack\b"],
        "split": [r"\.split\("],
        "chunk": [r"\.chunk\("],
        "gather": [r"\.gather\("],
        "interpolate": [r"F\.interpolate\b"],
        "broadcast_to": [r"\.broadcast_to\(", r"torch\.broadcast_to\b"],
        "add": [r"\.add\(", r"\s\+\s"],
        "mul": [r"\.mul\(", r"\s\*\s"],
        "topk": [r"\.topk\("],
        "einsum": [r"\.einsum\(", r"torch\.einsum\b"],
        "flatten": [r"\.flatten\("],
        "unsqueeze": [r"\.unsqueeze\("],
        "squeeze": [r"\.squeeze\("],
    }
    out = []
    for h, pats in patterns.items():
        for p in pats:
            if re.search(p, source):
                out.append(h)
                break
    return out


def _run_analyser(source: str, filename: str) -> Tuple[Optional[List[str]], str]:
    """Return (bug_handlers, note). bug_handlers is the per-bug 'kind'/'handler'
    annotation if exposed, else None."""
    try:
        from src.api import analyze
    except Exception as e:
        return None, f"import-failed: {e}"
    try:
        result = analyze(source, filename=filename)
    except Exception as e:
        return None, f"analyse-failed: {type(e).__name__}: {e}"
    bugs = list(getattr(result, "bugs", []) or [])
    if not bugs:
        return [], "no-bugs (silent-verified or abstain)"
    out = []
    for b in bugs:
        handler = (
            getattr(b, "handler", None)
            or getattr(b, "kind", None)
            or getattr(b, "rule", None)
        )
        cat = getattr(b, "category", None)
        out.append(f"{handler or '?'}|{cat or '?'}")
    return out, f"{len(bugs)} bug(s) reported"


def main() -> None:
    scope = _load_scope()

    rows: List[Dict[str, Any]] = []
    counts = {
        "Lean-audited": 0,
        "pen-and-paper": 0,
        "tested-only": 0,
        "uncovered": 0,
        "no-handler-id": 0,
    }
    catches_by_scope = {
        "Lean-audited or pen-and-paper": 0,
        "tested-only": 0,
        "mixed (touches both)": 0,
        "uncovered-only": 0,
    }

    for cid, relpath in CATCHES:
        path = os.path.join(ROOT, relpath)
        rec: Dict[str, Any] = {"id": cid, "path": relpath}
        if not os.path.isfile(path):
            rec["error"] = "source not found"
            rows.append(rec)
            continue
        with open(path) as fh:
            src = fh.read()
        bug_handlers, note = _run_analyser(src, relpath)
        rec["analyser_note"] = note
        rec["raw_bug_annotations"] = bug_handlers
        # Source-level handler set (for the file as a whole)
        src_handlers = _detect_handlers(src)
        rec["source_handlers"] = sorted(src_handlers)
        # Soundness scope of source handlers
        src_scopes = []
        for h in src_handlers:
            s = scope.get(h, "uncovered")
            src_scopes.append((h, s))
        rec["source_handler_scope"] = src_scopes
        # Footprint
        scopes_present = set(s for _, s in src_scopes)
        if not scopes_present:
            footprint = "no-handler-id"
        elif scopes_present <= {"Lean-audited", "pen-and-paper"}:
            footprint = "Lean-audited or pen-and-paper"
        elif "tested-only" in scopes_present and (
            scopes_present & {"Lean-audited", "pen-and-paper"}
        ):
            footprint = "mixed (touches both)"
        elif scopes_present == {"tested-only"}:
            footprint = "tested-only"
        elif scopes_present == {"uncovered"}:
            footprint = "uncovered-only"
        else:
            footprint = "mixed (touches both)"
        rec["footprint_classification"] = footprint
        catches_by_scope.setdefault(footprint, 0)
        catches_by_scope[footprint] += 1
        for h in src_handlers:
            counts[scope.get(h, "uncovered")] = counts.get(
                scope.get(h, "uncovered"), 0
            ) + 1
        rows.append(rec)

    out = {
        "_question": (
            "Round-2 Q4: of the 5/15 catches in the post-freeze unfiltered "
            "sample, how many fire through a tested-only handler vs. a "
            "Lean-audited or pen-and-paper handler?"
        ),
        "n_catches": len(CATCHES),
        "footprint_distribution": catches_by_scope,
        "handler_invocation_counts": counts,
        "method": (
            "For each of the 5 catches we (i) source-scan the repro for "
            "handler tokens (HANDLER_TOKENS subset matching the per-block "
            "scope script) and (ii) classify each detected handler against "
            "experiments_v5/handler_soundness_scope.json into "
            "{Lean-audited, pen-and-paper, tested-only, uncovered}.  A "
            "catch's footprint is 'Lean-audited or pen-and-paper' iff every "
            "detected handler in the file lies in those two sets, "
            "'tested-only' iff every detected handler is tested-only, "
            "'mixed' if the file touches both, and 'uncovered-only' if no "
            "detected handler is in any soundness scope.  This is the "
            "same per-block soundness footprint definition used in "
            "handler_scope_per_block.py."
        ),
        "rows": rows,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)

    md = []
    md.append(
        "# Soundness footprint of the 5 post-freeze catches (Round 2 Q4)"
    )
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append("| Footprint | Count |")
    md.append("|---|---:|")
    for k, v in catches_by_scope.items():
        md.append(f"| {k} | {v} |")
    md.append("")
    md.append("## Per-catch")
    md.append("")
    md.append("| id | footprint | source handlers (scope) |")
    md.append("|---|---|---|")
    for r in rows:
        sh = ", ".join(f"{h}({s})" for h, s in r.get("source_handler_scope", []))[:200]
        md.append(f"| {r['id']} | {r.get('footprint_classification','-')} | {sh} |")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(json.dumps(catches_by_scope, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
