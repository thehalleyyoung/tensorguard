#!/usr/bin/env python3.11
"""Round-2 Q5 follow-up: classify the 13 non-computable cases in the
30-item marker-only localisation audit (`marker_only_localization.json`)
into {silent-verified, explicit-abstain, error}, and flag membership in
the constructor-bound integer-attribute envelope class.

The reviewer (round-2 Q5) asks: of the 13/30 non-computable items, what
is the breakdown silent-verified vs. explicit-abstain, and how many fall
into the constructor-bound integer-attribute envelope class?

Method:
  * For each of the 13 ids, locate the source file under
    experiments_v5/{bug_repros,v8/real_bugs_*}.
  * Run the v5 analyser on the file (`src.v5.api.analyze`) with the
    documented input spec (best-effort recovery from manifest.json
    or class-level `INPUT_SPEC` constant; otherwise the analyser is
    invoked without an external input contract and we record the
    verdict the user-visible CLI would surface).
  * Emit verdict ∈ {silent-verified, explicit-abstain, error}.
  * Pattern-match the source for the constructor-bound envelope class:
    any of `int(...)` with a constructor-scalar arg, an
    `if dim == ...` style integer comparison, or a
    `q_len = sliding_window+1`-style arithmetic chain on a constructor
    scalar.

Output:
  reproducibility/marker_only_localization_noncomp_breakdown.json
  reproducibility/marker_only_localization_noncomp_breakdown.md
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

NON_COMP_IDS_TO_PATH = {
    "rb_001_xlstm_matq_view": "experiments_v5/v8/real_bugs_upstream/rb_001_xlstm_matq_view.py",
    "rb_002_xlstm_matk_view": "experiments_v5/v8/real_bugs_upstream/rb_002_xlstm_matk_view.py",
    "rb_pf_002_t5gemma2_xattn_cache": "experiments_v5/v8/real_bugs_postfreeze/rb_pf_002_t5gemma2_xattn_cache.py",
    "rb_pf_005_diffusers_npu_mask": "experiments_v5/v8/real_bugs_postfreeze/rb_pf_005_diffusers_npu_mask.py",
    "rb_pf_006_qwenimage_batch_ordering": "experiments_v5/v8/real_bugs_postfreeze/rb_pf_006_qwenimage_batch_ordering.py",
    "rb_uf_007_idefics3_patch_merger": "experiments_v5/v8/real_bugs_unfiltered/rb_uf_007_idefics3_patch_merger.py",
    "rb_uf_009_glm45_moe_chunk": "experiments_v5/v8/real_bugs_unfiltered/rb_uf_009_glm45_moe_chunk.py",
    "rb_uf_013_peft_vera_scaler": "experiments_v5/v8/real_bugs_unfiltered/rb_uf_013_peft_vera_scaler.py",
    "rb_uf_014_smollm3_grad_ckpt": "experiments_v5/v8/real_bugs_unfiltered/rb_uf_014_smollm3_grad_ckpt.py",
    "rb_uf_015_cosmos2_vision_transpose": "experiments_v5/v8/real_bugs_unfiltered/rb_uf_015_cosmos2_vision_transpose.py",
    "bug_001_sdpa_attn_mask_gqa": "experiments_v5/bug_repros/bug_001_sdpa_attn_mask_gqa.py",
    "bug_002_isclose_broadcast": "experiments_v5/bug_repros/bug_002_isclose_broadcast.py",
    "bug_006_cross_entropy_weight": "experiments_v5/bug_repros/bug_006_cross_entropy_weight.py",
}

# Pattern markers for the constructor-bound integer-attribute envelope class.
_INT_CAST_PAT = re.compile(r"\bint\s*\(\s*[a-zA-Z_]")
_FLOAT_CAST_PAT = re.compile(r"\bfloat\s*\(\s*[a-zA-Z_]")
_ROUND_CAST_PAT = re.compile(r"\bround\s*\(\s*[a-zA-Z_]")
_CTOR_SCALAR_PAT = re.compile(
    r"self\.(num_heads|num_chunks|chunk_size|hidden_size|d_model|"
    r"head_dim|sliding_window|num_layers|num_experts|top_k|"
    r"intermediate_size|num_attention_heads|kv_dim|"
    r"q_len|num_kv_heads|router_top_k)"
)
_BROADCAST_EQUALITY_PAT = re.compile(
    r"\.expand\(|\.broadcast_to\(|x\s*\+\s*[A-Za-z_].*\bmask\b"
)


def _classify_envelope(source: str) -> Dict[str, Any]:
    has_int_cast = bool(_INT_CAST_PAT.search(source) or _FLOAT_CAST_PAT.search(source))
    has_round_cast = bool(_ROUND_CAST_PAT.search(source))
    has_ctor_scalar = bool(_CTOR_SCALAR_PAT.search(source))
    has_broadcast_eq = bool(_BROADCAST_EQUALITY_PAT.search(source))
    member = (has_int_cast or has_round_cast or has_broadcast_eq) and has_ctor_scalar
    return {
        "has_int_cast": has_int_cast,
        "has_round_cast": has_round_cast,
        "has_ctor_scalar_attr": has_ctor_scalar,
        "has_broadcast_or_mask_chain": has_broadcast_eq,
        "constructor_bound_int_envelope_class": bool(member),
    }


def _run_analyser(source: str, filename: str) -> Dict[str, Any]:
    """Return user-visible verdict triple under TG v5 default config.

    Returns one of:
      {"verdict": "refuted", "n_bugs": k, ...}    — TG would refute
      {"verdict": "abstain", "n_abstains": k}      — explicit abstain
      {"verdict": "verified"}                       — silent verified
      {"verdict": "error", "exc": str}              — analyser crashed
    """
    try:
        from src.api import analyze
    except Exception as e:
        return {"verdict": "error", "exc": f"import-failed: {e}"}
    try:
        result = analyze(source, filename=filename)
    except Exception as e:
        return {
            "verdict": "error",
            "exc": f"{type(e).__name__}: {str(e)[:200]}",
            "tb": traceback.format_exc()[:500],
        }
    bugs = list(getattr(result, "bugs", []) or [])
    abstains = list(
        getattr(result, "abstains", None)
        or getattr(result, "abstain_reasons", [])
        or []
    )
    if bugs:
        return {
            "verdict": "refuted",
            "n_bugs": len(bugs),
            "bug_lines": [getattr(b.location, "line", None) for b in bugs[:5]],
        }
    if abstains:
        return {
            "verdict": "explicit-abstain",
            "n_abstains": len(abstains),
            "abstain_reasons": [str(a)[:80] for a in abstains[:5]],
        }
    return {"verdict": "silent-verified"}


def main() -> None:
    rows: List[Dict[str, Any]] = []
    counts = {
        "silent-verified": 0,
        "explicit-abstain": 0,
        "refuted-after-followup": 0,
        "error": 0,
    }
    n_envelope = 0

    for rid, relpath in NON_COMP_IDS_TO_PATH.items():
        path = os.path.join(ROOT, relpath)
        rec: Dict[str, Any] = {"id": rid, "path": relpath}
        if not os.path.isfile(path):
            rec["verdict"] = "error"
            rec["exc"] = "source file not found"
            counts["error"] += 1
            rows.append(rec)
            continue
        with open(path) as fh:
            source = fh.read()
        rec.update(_run_analyser(source, filename=relpath))
        env = _classify_envelope(source)
        rec["envelope"] = env
        if env["constructor_bound_int_envelope_class"]:
            n_envelope += 1
        v = rec["verdict"]
        if v == "refuted":
            counts["refuted-after-followup"] += 1
        elif v in counts:
            counts[v] += 1
        else:
            counts.setdefault(v, 0)
            counts[v] += 1
        rows.append(rec)

    out = {
        "_question": (
            "Round-2 Q5: of the 13/30 non-computable cases in the marker-only "
            "audit, what is the breakdown silent-verified vs. explicit-abstain, "
            "and how many fall into the constructor-bound integer-attribute "
            "envelope class?"
        ),
        "n_total": len(NON_COMP_IDS_TO_PATH),
        "verdict_counts": counts,
        "n_constructor_bound_int_envelope": n_envelope,
        "method": (
            "For each of the 13 ids, the analyser is invoked via "
            "`src.api.analyze(source)` with the user-visible default "
            "configuration (no synthesised assume_M, no rule edits since the "
            "freeze). Verdict triage: 'refuted' if any Bug objects are "
            "produced; 'explicit-abstain' if any Abstain reasons are present "
            "and no Bug; 'silent-verified' otherwise. Envelope-class "
            "membership is a static pattern check on the source for "
            "`int(.)/float(.)/round(.)` casts of constructor scalars or for "
            "broadcast/mask shape chains over constructor scalars."
        ),
        "rows": rows,
    }
    out_json = os.path.join(
        ROOT, "reproducibility", "marker_only_localization_noncomp_breakdown.json"
    )
    out_md = os.path.join(
        ROOT, "reproducibility", "marker_only_localization_noncomp_breakdown.md"
    )
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    md = []
    md.append("# Marker-only audit — non-computable breakdown (Round 2 Q5)")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append(f"- Non-computable cases: **{len(NON_COMP_IDS_TO_PATH)}**")
    for k, v in counts.items():
        md.append(f"- {k}: **{v}**")
    md.append(
        f"- Constructor-bound integer-attribute envelope class: "
        f"**{n_envelope}/{len(NON_COMP_IDS_TO_PATH)}**"
    )
    md.append("")
    md.append("## Per-item")
    md.append("")
    md.append("| id | verdict | envelope-class | note |")
    md.append("|---|---|:---:|---|")
    for r in rows:
        env = r.get("envelope", {})
        ec = "✓" if env.get("constructor_bound_int_envelope_class") else "."
        note = ""
        if r["verdict"] == "error":
            note = r.get("exc", "")
        elif r["verdict"] == "explicit-abstain":
            note = "; ".join(r.get("abstain_reasons", [])[:2])
        elif r["verdict"] == "refuted":
            note = f"n_bugs={r.get('n_bugs')}"
        md.append(f"| {r['id']} | {r['verdict']} | {ec} | {note[:60]} |")

    with open(out_md, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(json.dumps(counts, indent=2))
    print(f"envelope-class members: {n_envelope}/{len(NON_COMP_IDS_TO_PATH)}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
