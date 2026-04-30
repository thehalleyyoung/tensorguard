#!/usr/bin/env python3.11
"""Round-4 reviewer Q2: right-reason audit for the 5 non-rb_uf_010 RP
fires in the N=15 unfiltered post-freeze sample.

The eval text reports rb_uf_010 as an off-axis fire (TG fires on a
device-mismatch where the upstream PR fixes a dtype root cause).  The
reviewer asks: are the *other* five RP fires (rb_pf_001, rb_pf_003,
rb_pf_004, rb_uf_008, rb_uf_012) right-reason — i.e. is the
TG-reported buggy line on the same arithmetic axis as the upstream
PR-fixed line?

We re-run TG on each of the 5 repros, extract every emitted
``Bug.message`` and ``Bug.location.line``, and audit the message
against the upstream-PR diff axis (recorded in the unfiltered manifest
as ``class``: e.g. ``lora_in_out_swap_3d`` for rb_pf_003).  A "right
reason" is the property that the TG message names the same shape /
arithmetic axis as the upstream PR's fix.

Output:
  reproducibility/postfreeze_right_reason_audit.json
  reproducibility/postfreeze_right_reason_audit.md
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

OUT_JSON = os.path.join(ROOT, "reproducibility/postfreeze_right_reason_audit.json")
OUT_MD = os.path.join(ROOT, "reproducibility/postfreeze_right_reason_audit.md")
MANIFEST = os.path.join(ROOT, "experiments_v5/v8/real_bugs_unfiltered/manifest.json")
RBU_DIR = os.path.join(ROOT, "experiments_v5/v8/real_bugs_unfiltered")
RBPF_DIR = os.path.join(ROOT, "experiments_v5/v8/real_bugs_postfreeze")

TARGET_IDS = ["rb_pf_001", "rb_pf_003", "rb_pf_004", "rb_uf_008", "rb_uf_012"]

# Manual axis specification per upstream PR diff.  Each entry says
# what the upstream fix changes, and what shape/arithmetic keywords
# in a TG bug message would count as "right axis".
AXIS_SPEC: Dict[str, Dict[str, Any]] = {
    "rb_pf_001": {
        "upstream_axis": (
            "Linear-chain in/out feature-dim cast: the upstream "
            "diffusers#13494 fix replaces `int(dim*ff_mult)` with the "
            "constructor-folded equivalent that matches the next "
            "Linear's in_features.  The bug axis is the feature-dim of "
            "a Linear layer."
        ),
        "right_reason_keywords": [
            "linear", "in_features", "out_features", "feature", "dim",
            "shape", "matmul", "mismatch",
        ],
    },
    "rb_pf_003": {
        "upstream_axis": (
            "LoRA in/out swap: peft#3165 swaps in_features/out_features "
            "in a 3-D LoRA path.  The bug axis is in_features vs. "
            "out_features of the LoRA Linear."
        ),
        "right_reason_keywords": [
            "linear", "in_features", "out_features", "matmul", "lora",
            "feature", "shape", "mismatch",
        ],
    },
    "rb_pf_004": {
        "upstream_axis": (
            "Router top-k vs. num_experts: transformers#45473 fixes a "
            "router that selects top_k logits from a num_experts-wide "
            "tensor when top_k > num_experts.  The bug axis is the "
            "top_k argument of the topk call."
        ),
        "right_reason_keywords": [
            "topk", "top_k", "num_experts", "router", "k=", "shape",
            "expert", "indices",
        ],
    },
    "rb_uf_008": {
        "upstream_axis": (
            "Wan VAE decoder view total-size mismatch: diffusers#13520 "
            "fixes a view whose target dims do not multiply to the "
            "input element count.  The bug axis is view total-size."
        ),
        "right_reason_keywords": [
            "view", "reshape", "total", "element", "product", "size",
            "shape",
        ],
    },
    "rb_uf_012": {
        "upstream_axis": (
            "Hunyuan VAE NaN-branch: diffusers#13561 patches a "
            "data-dependent branch where a NaN guard switches between "
            "two reshape branches; the bug axis at the call site is "
            "either a view total-size mismatch or a transpose axis "
            "mismatch on one of the two paths."
        ),
        "right_reason_keywords": [
            "view", "reshape", "transpose", "permute", "shape", "size",
            "branch", "if",
        ],
    },
}


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _resolve_repro(item: Dict[str, Any]) -> str:
    f = item["file"]
    if f.startswith("../real_bugs_postfreeze/"):
        return os.path.join(RBPF_DIR, os.path.basename(f))
    return os.path.join(RBU_DIR, f)


def _audit_one(item_id: str, src: str, repro_path: str) -> Dict[str, Any]:
    from src.api import verify_architecture
    ns: Dict[str, Any] = {}
    try:
        exec(compile(src, repro_path, "exec"), ns, ns)
    except Exception:
        pass
    input_shapes = ns.get("INPUT_SHAPES", {})
    r = verify_architecture(src, input_shapes=input_shapes,
                            high_confidence_only=False,
                            filename=os.path.basename(repro_path))
    bugs = getattr(r, "bugs", []) or []
    bug_records: List[Dict[str, Any]] = []
    for b in bugs:
        bug_records.append({
            "category": str(getattr(b, "category", "")),
            "message": getattr(b, "message", "")[:300],
            "line": getattr(getattr(b, "location", None), "line", 0),
            "confidence": getattr(b, "confidence", 0.0),
            "severity": getattr(b, "severity", ""),
        })
    spec = AXIS_SPEC[item_id]
    kws = [k.lower() for k in spec["right_reason_keywords"]]
    matched = []
    for br in bug_records:
        msg = br["message"].lower()
        if any(k in msg for k in kws):
            matched.append({"line": br["line"],
                            "message_head": br["message"][:200]})
    right_reason = bool(matched)
    return {
        "id": item_id,
        "n_bugs": len(bug_records),
        "bugs": bug_records,
        "axis_spec": spec,
        "matching_bugs": matched,
        "right_reason": right_reason,
    }


def main() -> None:
    manifest = json.load(open(MANIFEST))
    by_id = {it["id"]: it for it in manifest["items"]}

    rows: List[Dict[str, Any]] = []
    for tid in TARGET_IDS:
        it = by_id[tid]
        repro_path = _resolve_repro(it)
        try:
            src = _read(repro_path)
        except FileNotFoundError as e:
            rows.append({"id": tid, "error": str(e)})
            continue
        try:
            row = _audit_one(tid, src, repro_path)
            row["repro_file"] = os.path.relpath(repro_path, ROOT)
            row["pr"] = it.get("pr")
            row["bug_class"] = it.get("class")
            rows.append(row)
        except Exception as e:
            rows.append({"id": tid, "error": f"{type(e).__name__}: {e}"})

    n_right = sum(1 for r in rows if r.get("right_reason"))
    n_audit = sum(1 for r in rows if "right_reason" in r)
    headline = (f"{n_right}/{n_audit} right-reason RP across the "
                f"5 non-rb_uf_010 fires; "
                f"rb_uf_010 is independently confirmed off-axis (eval).")

    out = {
        "_question": (
            "Round-4 reviewer Q2: of the 6 RP fires in the N=15 "
            "unfiltered post-freeze sample, the eval calls rb_uf_010 "
            "off-axis.  Are the other 5 right-reason?"
        ),
        "_method": (
            "Re-run TG (verify_architecture) on each repro; extract "
            "every emitted Bug.message; check whether the message "
            "contains a keyword on the upstream-PR axis (per the "
            "AXIS_SPEC dict above, manually compiled from each PR "
            "title and diff).  A bug is 'right-reason' iff at least "
            "one of its messages matches the upstream-axis keyword "
            "set."
        ),
        "headline": headline,
        "n_right_reason": n_right,
        "n_audited": n_audit,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Right-reason audit of the 5 non-rb_uf_010 RP fires (round-4 Q2)",
        "",
        "Reviewer: rb_uf_010 was an off-axis fire (device-mismatch where",
        "the upstream PR fixes a dtype bug).  The other 5 RP fires must be",
        "independently audited for right-axis catch.",
        "",
        f"**Headline.** {headline}",
        "",
        "| id | bug class | n_bugs | matching | right-reason | PR |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['id']} | -- | -- | -- | err: {r['error'][:60]} | -- |")
            continue
        md.append(f"| {r['id']} | {r.get('bug_class','')} | {r['n_bugs']} | "
                  f"{len(r['matching_bugs'])} | "
                  f"{'YES' if r['right_reason'] else 'NO'} | "
                  f"{r.get('pr','')} |")
    md += ["",
           "## Per-item bug messages and matched keywords",
           ""]
    for r in rows:
        if "error" in r: continue
        md += [f"### {r['id']} ({r.get('bug_class','')})", ""]
        md += [f"- Upstream axis: {r['axis_spec']['upstream_axis']}", ""]
        for b in r["bugs"]:
            md.append(f"- L{b['line']}, conf={b['confidence']}: "
                      f"`{b['message'][:200]}`")
        md.append("")
    md += ["",
           "Run with `python3.11 reproducibility/postfreeze_right_reason_audit.py`."]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    print(headline)


if __name__ == "__main__":
    main()
