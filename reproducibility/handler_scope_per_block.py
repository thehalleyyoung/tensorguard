#!/usr/bin/env python3.11
"""Reviewer Q1 (round 1): per-block fraction of V/CV verdicts whose
operator chain touches a tested-only handler (one of the 48/79 not
covered by Lean or pen-and-paper soundness).

Approach: regex-scan each block's Python source for tokens identifying
torch operators in the global handler taxonomy
(``experiments_v5/handler_soundness_scope.json``).  For each block we
record the multiset of handlers it invokes (best-effort source-level
detection: ``self.fc(`` matches ``linear``, ``self.conv2d(`` matches
``conv2d``, ``x.view(`` matches ``view``, etc.).  We then cross with
``experiments_v5/v8/per_block_user_visible_rp.json`` to compute the
fraction of Verified / Verified-no-assume / CV / LW / Abstain blocks
that contain *only* Lean-audited (or Lean-audited+pen-and-paper)
handlers vs. blocks that contain at least one tested-only handler.

Output: ``reproducibility/handler_scope_per_block.json`` and ``.md``.
"""
from __future__ import annotations
import json, os, re, sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCOPE = os.path.join(ROOT, "experiments_v5/handler_soundness_scope.json")
CORPUS = os.path.join(ROOT, "experiments_v5/v5_block_corpus.jsonl")
PER_BLOCK = os.path.join(ROOT, "experiments_v5/v8/per_block_user_visible_rp.json")
USER_RP = os.path.join(ROOT, "experiments_v5/v8/user_visible_rp.json")
HYBRID = os.path.join(ROOT, "experiments_v5/hybrid_mode_results.json")
RECLASS = os.path.join(ROOT, "experiments_v5/verdict_reclassification.json")
OUT_JSON = os.path.join(ROOT, "reproducibility/handler_scope_per_block.json")
OUT_MD = os.path.join(ROOT, "reproducibility/handler_scope_per_block.md")

# Token patterns: match each handler by the strings most likely to
# appear in nn.Module class source for that op.
# Conservative: prefer false-positives (over-attribute handlers to a
# block) so that "only Lean-audited" is a stricter category.
HANDLER_TOKENS = {
    # Lean-verified set
    "matmul":     [r"\.matmul\b", r"@\s*self\.", r"torch\.matmul\b"],
    "bmm":        [r"\.bmm\b", r"torch\.bmm\b"],
    "batched_matmul": [r"\.einsum\b"],   # weak proxy
    "conv1d":     [r"\bConv1d\b", r"F\.conv1d\b"],
    "conv2d":     [r"\bConv2d\b", r"F\.conv2d\b"],
    "conv3d":     [r"\bConv3d\b", r"F\.conv3d\b"],
    "conv_transpose2d": [r"\bConvTranspose2d\b", r"F\.conv_transpose2d\b"],
    "view":       [r"\.view\("],
    "reshape":    [r"\.reshape\("],
    "permute":    [r"\.permute\("],
    "transpose":  [r"\.transpose\(", r"\.t\(\)"],
    "expand":     [r"\.expand\("],
    "repeat":     [r"\.repeat\("],
    "broadcast_to": [r"\.broadcast_to\(", r"torch\.broadcast_to\b"],
    "cat":        [r"torch\.cat\b", r"\.cat\("],
    "stack":      [r"torch\.stack\b"],
    "split":      [r"\.split\(", r"torch\.split\b"],
    "chunk":      [r"\.chunk\(", r"torch\.chunk\b"],
    "unbind":     [r"\.unbind\(", r"torch\.unbind\b"],
    "gather":     [r"\.gather\(", r"torch\.gather\b"],
    "scatter":    [r"\.scatter\(", r"\.scatter_\b"],
    "index_select": [r"\.index_select\b", r"torch\.index_select\b"],
    "narrow":     [r"\.narrow\("],
    "embed":      [r"\bEmbedding\b"],
    "layer_norm": [r"\bLayerNorm\b", r"F\.layer_norm\b"],
    "rms_norm":   [r"\bRMSNorm\b"],
    "scaled_dot_product_attention": [r"scaled_dot_product_attention"],
    "linear":     [r"\bLinear\b", r"F\.linear\b"],
    # Tested-only (subset reviewer Q1 cares about)
    "batch_norm": [r"\bBatchNorm[123]d\b", r"F\.batch_norm\b"],
    "group_norm": [r"\bGroupNorm\b", r"F\.group_norm\b"],
    "instance_norm": [r"\bInstanceNorm[123]d\b", r"F\.instance_norm\b"],
    "multihead_attention": [r"\bMultiheadAttention\b"],
    "conv_transpose1d": [r"\bConvTranspose1d\b"],
    "squeeze":    [r"\.squeeze\("],
    "unsqueeze":  [r"\.unsqueeze\("],
    "flatten":    [r"\bFlatten\b", r"\.flatten\(", r"torch\.flatten\b"],
    "softmax":    [r"\.softmax\b", r"\bSoftmax\b", r"F\.softmax\b"],
    "relu":       [r"\bReLU\b", r"F\.relu\b", r"\.relu\("],
    "gelu":       [r"\bGELU\b", r"F\.gelu\b"],
    "silu":       [r"\bSiLU\b", r"F\.silu\b"],
    "tanh":       [r"\bTanh\b", r"\.tanh\("],
    "sigmoid":    [r"\bSigmoid\b", r"\.sigmoid\("],
    "dropout":    [r"\bDropout\b", r"F\.dropout\b"],
    "cross_entropy": [r"CrossEntropy", r"F\.cross_entropy\b"],
    "interpolate": [r"F\.interpolate\b"],
    "pixel_shuffle": [r"\bPixelShuffle\b"],
    "pixel_unshuffle": [r"\bPixelUnshuffle\b"],
    "topk":       [r"\.topk\(", r"torch\.topk\b"],
    "max_pool2d": [r"\bMaxPool2d\b", r"F\.max_pool2d\b"],
    "avg_pool2d": [r"\bAvgPool2d\b", r"F\.avg_pool2d\b"],
    "adaptive_avg_pool2d": [r"\bAdaptiveAvgPool2d\b"],
    "adaptive_max_pool2d": [r"\bAdaptiveMaxPool2d\b"],
    "add":        [r"\.add\(", r"\+="],
    "mul":        [r"\.mul\("],
    "div":        [r"\.div\("],
    "pow":        [r"\.pow\("],
    "sqrt":       [r"torch\.sqrt\b", r"\.sqrt\("],
    "rsqrt":      [r"torch\.rsqrt\b", r"\.rsqrt\("],
    "sum":        [r"\.sum\("],
    "mean":       [r"\.mean\("],
    "var":        [r"\.var\("],
    "std":        [r"\.std\("],
    "max":        [r"\.max\("],
    "min":        [r"\.min\("],
    "argmax":     [r"\.argmax\("],
    "argmin":     [r"\.argmin\("],
    "where":      [r"torch\.where\b"],
    "masked_fill": [r"\.masked_fill\("],
    "clamp":      [r"\.clamp\("],
    "abs":        [r"\.abs\("],
    "exp":        [r"torch\.exp\b", r"\.exp\("],
    "log":        [r"torch\.log\b", r"\.log\("],
    "log_softmax": [r"F\.log_softmax\b"],
    "to":         [r"\.to\("],
    "type":       [r"\.type\("],
    "contiguous": [r"\.contiguous\("],
    "detach":     [r"\.detach\("],
}


def detect_handlers(src: str) -> set:
    found = set()
    for h, patterns in HANDLER_TOKENS.items():
        for p in patterns:
            if re.search(p, src):
                found.add(h)
                break
    return found


def main():
    scope = json.load(open(SCOPE))
    handler_scope = {h["name"]: h["scope"] for h in scope["handlers"]}
    lean_set = {n for n, s in handler_scope.items() if s == "lean_verified"}
    pp_set = {n for n, s in handler_scope.items() if s == "pen_and_paper"}
    tested_only_set = {n for n, s in handler_scope.items() if s == "tested_only"}

    # Load 488 blocks
    blocks = {}
    with open(CORPUS) as f:
        for line in f:
            o = json.loads(line)
            blocks[o["id"]] = o

    # Load per-block verdicts (only 57 V-with-assume + 23 collapse).
    # Plus the headline distribution from user_visible_rp.json.
    pb = json.load(open(PER_BLOCK))
    # Build a verdict index for all 488 by reading
    # build_user_visible_rp.json output; per_block_user_visible_rp.json
    # only enumerates the 57 Verified blocks.  We supplement by reading
    # silent_miss_autopsy / classifications if available; otherwise we
    # treat unmatched blocks as 'unknown'.

    # Use per_block list for the V verdicts; supplement with hybrid + reclass for CV/LW
    verdicts: dict = {}
    for b in pb["per_block"]:
        verdicts[b["id"]] = {
            "verdict_with_assume": b["verdict_with_assume"],
            "verdict_no_assume": b["verdict_no_assume"],
        }
    # Round-3 reviewer Q1: also enumerate CV (128) and LW (78) verdicts.
    reclass = json.load(open(RECLASS))
    _reclass_map = {x["id"]: x["verdict"] for x in reclass["block_corpus"]["per_item"]}
    hybrid = json.load(open(HYBRID))
    for it in hybrid["per_item"]:
        bid = it["id"]
        if it["tg_verdict"] == "Refuted":
            sub = _reclass_map.get(bid)
            if sub == "CONTRACT_VIOLATION":
                verdicts[bid] = {
                    "verdict_with_assume": "CV",
                    "verdict_no_assume": "Library_Warn",
                }
            elif sub == "LIBRARY_WARN":
                verdicts[bid] = {
                    "verdict_with_assume": "Library_Warn",
                    "verdict_no_assume": "Library_Warn",
                }
        elif it["tg_verdict"] == "Abstain" and bid not in verdicts:
            verdicts[bid] = {
                "verdict_with_assume": "Abstain",
                "verdict_no_assume": "Abstain",
            }

    # Annotate each block with its handler set
    rows = []
    for bid, blk in blocks.items():
        src = blk["source"]
        hs = detect_handlers(src)
        scope_set = {h: handler_scope.get(h, "uncovered") for h in hs}
        only_lean = bool(hs) and all(
            handler_scope.get(h) in {"lean_verified", "pen_and_paper"} for h in hs
        )
        any_tested_only = any(handler_scope.get(h) == "tested_only" for h in hs)
        any_uncovered = any(handler_scope.get(h, "uncovered") == "uncovered" for h in hs)
        rows.append({
            "id": bid,
            "category": blk.get("category"),
            "library": blk.get("library"),
            "loc": blk.get("loc"),
            "verdict_with_assume": verdicts.get(bid, {}).get("verdict_with_assume"),
            "verdict_no_assume": verdicts.get(bid, {}).get("verdict_no_assume"),
            "handlers": sorted(hs),
            "n_lean": sum(1 for h in hs if handler_scope.get(h) == "lean_verified"),
            "n_pen_and_paper": sum(1 for h in hs if handler_scope.get(h) == "pen_and_paper"),
            "n_tested_only": sum(1 for h in hs if handler_scope.get(h) == "tested_only"),
            "n_uncovered": sum(1 for h in hs if handler_scope.get(h, "uncovered") == "uncovered"),
            "soundness_footprint": (
                "lean_or_pp_only" if only_lean else
                ("touches_tested_only" if any_tested_only else
                 ("uncovered_only" if any_uncovered else "no_handlers_detected"))
            ),
        })

    # Aggregate: split rows by verdict (only the 57 V blocks have verdicts populated)
    by_v_assume = defaultdict(lambda: Counter())
    for r in rows:
        v = r["verdict_with_assume"] or "unknown"
        by_v_assume[v][r["soundness_footprint"]] += 1
    by_v_noassume = defaultdict(lambda: Counter())
    for r in rows:
        v = r["verdict_no_assume"] or "unknown"
        by_v_noassume[v][r["soundness_footprint"]] += 1

    # Specifically: V (57) and CV (128) under headline regime.
    # Per-block JSON does not enumerate CV blocks; document this gap.
    out = {
        "_question": (
            "Reviewer Q1: of the 57 Verified-with-assume blocks (the only "
            "blocks for which the soundness theorem is invoked), how many "
            "touch only Lean-audited or pen-and-paper handlers vs. how "
            "many touch at least one of the 48 tested-only handlers? "
            "Detection is best-effort source-token regex; see HANDLER_TOKENS "
            "in this script for the table."),
        "n_lean_verified_handlers": len(lean_set),
        "n_pen_and_paper_handlers": len(pp_set),
        "n_tested_only_handlers": len(tested_only_set),
        "by_verdict_with_assume": {k: dict(v) for k, v in by_v_assume.items()},
        "by_verdict_no_assume": {k: dict(v) for k, v in by_v_noassume.items()},
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # Markdown
    v_rows = [r for r in rows if r["verdict_with_assume"] == "Verified"]
    v_lean = sum(1 for r in v_rows if r["soundness_footprint"] == "lean_or_pp_only")
    v_tested = sum(1 for r in v_rows if r["soundness_footprint"] == "touches_tested_only")
    v_uncov = sum(1 for r in v_rows if r["soundness_footprint"] == "uncovered_only")
    v_none = sum(1 for r in v_rows if r["soundness_footprint"] == "no_handlers_detected")
    md = [
        "# Per-block handler scope on the 488-block corpus",
        "",
        "Reviewer Q1 (round 1): of the verdicts inside the soundness theorem ",
        "(Verified, Refuted-Proof), how many touch *only* Lean-audited or ",
        "pen-and-paper handlers vs. at least one tested-only handler?",
        "",
        f"- **Total blocks scanned**: {len(rows)}",
        f"- **Lean-audited handlers**: {len(lean_set)}",
        f"- **Pen-and-paper handlers**: {len(pp_set)}",
        f"- **Tested-only handlers**: {len(tested_only_set)}",
        "",
        "## Headline regime (with synthesised assume_M): 57 Verified blocks",
        "",
        f"- only Lean-or-pen-paper handlers: **{v_lean}**",
        f"- touches at least one tested-only handler: **{v_tested}**",
        f"- only uncovered handlers: {v_uncov}",
        f"- no handlers detected: {v_none}",
        "",
        "Therefore the **soundness theorem applies tightly to "
        f"{v_lean}/{len(v_rows)}** Verified verdicts; the remaining ",
        f"{len(v_rows) - v_lean} touch handlers covered only by random ",
        "agreement testing.  The paper now reports both numbers in §4.4 ",
        "rather than the union under the 'Lean-audited' framing.",
        "",
        "## No-assume regime: 34 user-visible Verified blocks",
        "",
    ]
    v_rows_n = [r for r in rows if r["verdict_no_assume"] == "Verified"]
    v_lean_n = sum(1 for r in v_rows_n if r["soundness_footprint"] == "lean_or_pp_only")
    md.append(f"- only Lean-or-pen-paper handlers: **{v_lean_n}**")
    md.append(f"- touches at least one tested-only handler: "
              f"**{sum(1 for r in v_rows_n if r['soundness_footprint'] == 'touches_tested_only')}**")
    md.append(f"- (no handlers / uncovered): "
              f"{sum(1 for r in v_rows_n if r['soundness_footprint'] in ('uncovered_only', 'no_handlers_detected'))}")
    md.append("")
    md.append("## CV (128 blocks) under headline regime — round-3 Q1")
    md.append("")
    cv_rows = [r for r in rows if r["verdict_with_assume"] == "CV"]
    cv_lean = sum(1 for r in cv_rows if r["soundness_footprint"] == "lean_or_pp_only")
    cv_tested = sum(1 for r in cv_rows if r["soundness_footprint"] == "touches_tested_only")
    cv_uncov = sum(1 for r in cv_rows if r["soundness_footprint"] == "uncovered_only")
    cv_none = sum(1 for r in cv_rows if r["soundness_footprint"] == "no_handlers_detected")
    md.append(f"- only Lean-or-pen-paper handlers: **{cv_lean}**")
    md.append(f"- touches at least one tested-only handler: **{cv_tested}**")
    md.append(f"- only uncovered handlers: {cv_uncov}")
    md.append(f"- no handlers detected: {cv_none}")
    md.append("")
    md.append(
        f"Therefore on the 128 CV verdicts the soundness theorem applies "
        f"tightly (entire forward path inside the Lean-or-pen-paper "
        f"footprint) to **{cv_lean}/{len(cv_rows)}** verdicts; "
        f"{cv_tested} touch at least one of the 48 tested-only handlers.")
    md.append("")
    md.append("## Combined V+CV (185 in-soundness verdicts under assume regime)")
    md.append("")
    combo_rows = v_rows + cv_rows
    combo_lean = v_lean + cv_lean
    combo_tested = v_tested + cv_tested
    md.append(f"- only Lean-or-pen-paper handlers: **{combo_lean}/{len(combo_rows)}**")
    md.append(f"- touches at least one tested-only handler: **{combo_tested}/{len(combo_rows)}**")
    md.append("")
    md.append("## Detection methodology")
    md.append("")
    md.append(
        "Source-token regex (see HANDLER_TOKENS in "
        "`reproducibility/handler_scope_per_block.py`).  Conservative: "
        "ambiguous tokens are attributed to the handler.  This *over*-counts "
        "tested-only touches, so the reported 'tight Lean coverage' bucket "
        "is a lower bound and the 'tested-only' bucket is an upper bound on "
        "the true partition.")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    print(f"V-with-assume Lean-only={v_lean}/{len(v_rows)}, "
          f"tested-only-touch={v_tested}, uncovered-only={v_uncov}")
    cv_rows2 = [r for r in rows if r["verdict_with_assume"] == "CV"]
    cv_lean2 = sum(1 for r in cv_rows2 if r["soundness_footprint"] == "lean_or_pp_only")
    cv_tested2 = sum(1 for r in cv_rows2 if r["soundness_footprint"] == "touches_tested_only")
    print(f"CV-with-assume Lean-only={cv_lean2}/{len(cv_rows2)}, "
          f"tested-only-touch={cv_tested2}")


if __name__ == "__main__":
    main()
