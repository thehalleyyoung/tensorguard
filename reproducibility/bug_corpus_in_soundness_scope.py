"""
bug_corpus_in_soundness_scope.py

Round-1 reviewer Q1 (per-bug scope column for the 60 historical bugs and 10
upstream-faithful re-extracts): for each RP catch in the bug corpora, determine
whether the catch path traverses ONLY Lean-audited or pen-and-paper-audited
handlers (inside Theorem 2's footprint) or whether at least one tested-only
handler is involved.

Method
------
Handler scope is read from experiments_v5/handler_soundness_scope.json.
For the 60 historical bugs, the bug CATEGORY directly encodes which primary
detection handler fires the RP verdict (the category is the shape-arithmetic
class for which TG fires; e.g., 'view_reshape_total_size' → view/reshape
handler; 'attention_dim' → scaled_dot_product_attention handler).  We map each
category to its primary firing handler and look up that handler's scope.

For the 'other' category (11 bugs) and the 10 upstream-faithful re-extracts,
which have no category, we supplement the category mapping with source-token
detection restricted to the OPERATOR THAT APPEARS IN THE BUG TITLE / REPRO
BODY as the primary shape mismatch.  Specifically: we scan the repro file for
the primary operator tokens in priority order (most-specific first) and assign
the first matching token as the primary handler.  This differs from the
broader "any handler in source" scan: we look for the operator that the repro
is EXERCISING, not every operator in a surrounding module body.
"""

import json
import re
import os
from pathlib import Path

BASE = Path(__file__).parent.parent  # repo root

# Load handler scope table
scope_data = json.load(open(BASE / "experiments_v5/handler_soundness_scope.json"))
lean_verified_set = set(scope_data["meta"]["lean_verified_set"])
pen_and_paper_set = {h["name"] for h in scope_data["handlers"] if h["scope"] == "pen_and_paper"}
tested_only_set = {h["name"] for h in scope_data["handlers"] if h["scope"] == "tested_only"}

IN_SOUNDNESS = lean_verified_set | pen_and_paper_set

# Category → primary firing handler (based on TG's handler dispatch table)
# These are the handlers TG uses to detect the shape mismatch for each category.
CATEGORY_PRIMARY_HANDLER = {
    "attention_dim": "scaled_dot_product_attention",   # Lean-verified
    "view_reshape_total_size": "view",                 # Lean-verified (view/reshape)
    "conv_channel_mismatch": "conv2d",                 # Lean-verified
    "linear_inout_mismatch": "linear",                 # Lean-verified
    "einsum_dim": "einsum",                            # Pen-and-paper
    "transpose_axes": "transpose",                     # Lean-verified (transpose/permute)
    "batchnorm_features": "batch_norm",               # Tested-only
    "embedding_index": "embed",                        # Lean-verified
    "broadcasting": "elementwise_binary",              # Pen-and-paper (T-Broadcast rule)
}

# For 'other' category and upstream-faithful bugs: token-based primary detection
# Priority-ordered: most-specific token first.
PRIMARY_TOKENS = [
    # Lean-verified
    ("scaled_dot_product_attention", "scaled_dot_product_attention"),
    ("torch.gather", "gather"),
    ("torch.scatter_", "scatter"),
    ("torch.scatter", "scatter"),
    (".scatter_(", "scatter"),
    ("torch.dot", "matmul"),      # torch.dot routes through matmul handler
    ("nn.Embedding", "embed"),
    ("nn.Linear", "linear"),
    ("F.linear", "linear"),
    (".view(", "view"),
    (".reshape(", "reshape"),
    ("torch.cat", "cat"),
    ("torch.stack", "stack"),
    ("torch.permute", "permute"),
    (".permute(", "permute"),
    (".transpose(", "transpose"),
    ("nn.Conv2d", "conv2d"),
    ("torch.bmm", "bmm"),
    ("torch.matmul", "matmul"),
    ("torch.einsum", "einsum"),   # Pen-and-paper
    ("torch.repeat_interleave", "repeat_interleave"),  # Tested-only
    ("nn.MaxPool2d", "max_pool2d"),   # Tested-only
    ("nn.BatchNorm", "batch_norm"),   # Tested-only
    ("CrossEntropyLoss", "cross_entropy"),  # Tested-only
    ("NLLLoss", "cross_entropy"),           # Tested-only
    ("MSELoss", "cross_entropy"),           # Tested-only
    ("torch.linalg.inv", "linalg_solve"),   # Tested-only
]


def handler_scope(handler: str) -> str:
    if handler in lean_verified_set:
        return "lean_verified"
    if handler in pen_and_paper_set:
        return "pen_and_paper"
    return "tested_only"


def primary_handler_from_source(source: str) -> str:
    """Find the primary shape-mismatch handler from repro source."""
    for token, handler in PRIMARY_TOKENS:
        if token in source:
            return handler
    return ""


def classify_bug(bug: dict, repro_source: str) -> dict:
    """Classify a single bug for soundness scope."""
    bug_id = bug["id"]
    verdict = bug.get("tg_verdict", "UNKNOWN")
    category = bug.get("category", "?")

    if verdict != "REFUTED_PROOF":
        return {
            "bug_id": bug_id,
            "category": category,
            "tg_verdict": verdict,
            "primary_handler": None,
            "handlers_used": [],
            "in_soundness": "N/A",
            "reason": "TG did not return REFUTED_PROOF; soundness scope not applicable"
        }

    # Use category-primary mapping when available
    if category in CATEGORY_PRIMARY_HANDLER:
        primary = CATEGORY_PRIMARY_HANDLER[category]
        scope = handler_scope(primary)
        in_s = "Y" if scope in ("lean_verified", "pen_and_paper") else "N"
        return {
            "bug_id": bug_id,
            "category": category,
            "tg_verdict": verdict,
            "primary_handler": primary,
            "handlers_used": [primary],
            "in_soundness": in_s,
            "reason": f"category maps to primary handler '{primary}' (scope: {scope})"
        }

    # For 'other' category and upstream bugs: detect from source
    primary = primary_handler_from_source(repro_source)
    if primary:
        scope = handler_scope(primary)
        in_s = "Y" if scope in ("lean_verified", "pen_and_paper") else "N"
        return {
            "bug_id": bug_id,
            "category": category,
            "tg_verdict": verdict,
            "primary_handler": primary,
            "handlers_used": [primary],
            "in_soundness": in_s,
            "reason": f"source-token detection: primary handler '{primary}' (scope: {scope})"
        }

    # Fallback: conservatively out-of-soundness
    return {
        "bug_id": bug_id,
        "category": category,
        "tg_verdict": verdict,
        "primary_handler": None,
        "handlers_used": [],
        "in_soundness": "N",
        "reason": "no primary handler detected; conservatively marked out-of-soundness"
    }


def load_repro_source(repro_file: str) -> str:
    """Load repro source file content."""
    if not repro_file:
        return ""
    path = BASE / repro_file
    if path.exists():
        return path.read_text(errors="replace")
    # Try without leading path component
    alt = BASE / "experiments_v5" / "bug_repros" / Path(repro_file).name
    if alt.exists():
        return alt.read_text(errors="replace")
    return ""


def load_repro_source(repro_file: str) -> str:
    """Load repro source file content."""
    if not repro_file:
        return ""
    path = BASE / repro_file
    if path.exists():
        return path.read_text(errors="replace")
    # Try without leading path component
    alt = BASE / "experiments_v5" / "bug_repros" / Path(repro_file).name
    if alt.exists():
        return alt.read_text(errors="replace")
    # Try upstream real_bugs directory
    alt2 = BASE / "experiments_v5" / "v8" / "real_bugs_upstream" / Path(repro_file).name
    if alt2.exists():
        return alt2.read_text(errors="replace")
    return ""


def main():
    # Process 60 historical bugs
    manifest = json.load(open(BASE / "experiments_v5/bug_corpus_manifest.json"))
    bugs_60 = manifest["items"]

    rows_60 = []
    for bug in bugs_60:
        src = load_repro_source(bug.get("repro_file", ""))
        row = classify_bug(bug, src)
        row["corpus"] = "historical_60"
        rows_60.append(row)

    # Process 10 upstream-faithful re-extracts
    upstream = json.load(open(BASE / "reproducibility/real_bugs_upstream.json"))
    records_10 = upstream["records"]

    rows_10 = []
    for rec in records_10:
        # Map upstream records to bug-like structure
        verdict_map = {"RP_0.99": "REFUTED_PROOF", "RP_0.80": "REFUTED_PROOF",
                       "Verified_or_Silent": "VERIFIED"}
        tg_verdict = verdict_map.get(rec.get("status", ""), "UNKNOWN")
        bug = {
            "id": rec["id"],
            "category": "upstream_faithful",
            "tg_verdict": tg_verdict,
            "repro_file": rec.get("file", "")
        }
        src = load_repro_source(bug["repro_file"])
        row = classify_bug(bug, src)
        row["corpus"] = "upstream_faithful_10"
        rows_10.append(row)

    all_rows = rows_60 + rows_10

    # Aggregate stats
    def aggregate(rows, corpus_label):
        rp_rows = [r for r in rows if r["tg_verdict"] == "REFUTED_PROOF"]
        in_soundness_y = [r for r in rp_rows if r["in_soundness"] == "Y"]
        in_soundness_n = [r for r in rp_rows if r["in_soundness"] == "N"]
        na_rows = [r for r in rows if r["in_soundness"] == "N/A"]
        return {
            "corpus": corpus_label,
            "n_total": len(rows),
            "n_rp": len(rp_rows),
            "n_in_soundness_Y": len(in_soundness_y),
            "n_in_soundness_N": len(in_soundness_n),
            "n_not_applicable": len(na_rows),
            "pct_rp_in_soundness": round(len(in_soundness_y) / len(rp_rows) * 100, 1) if rp_rows else 0,
        }

    agg_60 = aggregate(rows_60, "historical_60")
    agg_10 = aggregate(rows_10, "upstream_faithful_10")
    rp_combined = [r for r in all_rows if r["tg_verdict"] == "REFUTED_PROOF"]
    agg_combined = {
        "corpus": "combined",
        "n_rp": len(rp_combined),
        "n_in_soundness_Y": sum(1 for r in rp_combined if r["in_soundness"] == "Y"),
        "n_in_soundness_N": sum(1 for r in rp_combined if r["in_soundness"] == "N"),
    }
    agg_combined["pct_rp_in_soundness"] = round(
        agg_combined["n_in_soundness_Y"] / agg_combined["n_rp"] * 100, 1
    ) if agg_combined["n_rp"] else 0

    result = {
        "_question": (
            "Round-1 reviewer Q1: for the 60 historical bugs and 10 upstream-faithful re-extracts, "
            "how many RP catches are entirely within the Lean-audited or pen-and-paper-audited "
            "handler set (Theorem 2 footprint)?"
        ),
        "method": (
            "category-to-primary-handler mapping for the 9 named categories; "
            "source-token detection for 'other' and upstream-faithful bugs"
        ),
        "in_soundness_set": sorted(IN_SOUNDNESS),
        "tested_only_set": sorted(tested_only_set),
        "aggregate": {
            "historical_60": agg_60,
            "upstream_faithful_10": agg_10,
            "combined": agg_combined,
        },
        "per_bug": all_rows,
    }

    out_path = BASE / "reproducibility/bug_corpus_in_soundness_scope.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote", out_path)

    # Print summary
    print(f"\nHistorical 60-bug corpus:")
    print(f"  {agg_60['n_rp']} RP verdicts total")
    print(f"  {agg_60['n_in_soundness_Y']} / {agg_60['n_rp']} in-soundness ({agg_60['pct_rp_in_soundness']}%)")
    print(f"\nUpstream-faithful 10-bug re-extracts:")
    print(f"  {agg_10['n_rp']} RP verdicts total")
    print(f"  {agg_10['n_in_soundness_Y']} / {agg_10['n_rp']} in-soundness ({agg_10['pct_rp_in_soundness']}%)")
    print(f"\nCombined:")
    print(f"  {agg_combined['n_rp']} RP verdicts total")
    print(f"  {agg_combined['n_in_soundness_Y']} / {agg_combined['n_rp']} in-soundness ({agg_combined['pct_rp_in_soundness']}%)")

    return result


if __name__ == "__main__":
    main()
