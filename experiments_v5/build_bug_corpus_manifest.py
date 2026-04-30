#!/usr/bin/env python3
"""
Build the auditable benchmark manifest for the 60-bug corpus.

Produces:
  experiments_v5/bug_corpus_manifest.json
  experiments_v5/bug_corpus_manifest.md
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
CORPUS_JSONL   = BASE / "v5_bug_corpus.jsonl"
RESULTS_JSON   = BASE / "v5_benchmark_results.json"
OUT_JSON       = BASE / "bug_corpus_manifest.json"
OUT_MD         = BASE / "bug_corpus_manifest.md"

# Tokens whose presence in a repro file marks it as out-of-fragment
OUT_OF_FRAGMENT_TOKENS = {
    "torch.compile":          "torch_compile",
    "torch.fx":               "torch_fx",
    "register_hook":          "custom_autograd",
    "autograd.Function":      "custom_autograd",
    "__torch_dispatch__":     "custom_autograd",
    "__torch_function__":     "custom_autograd",
    "torch._dynamo":          "torch_dynamo",
    "triton":                 "external_kernel",
    "cuda_extension":         "external_kernel",
    "load_inline":            "external_kernel",
    "cpp_extension":          "external_kernel",
}

BUCKET_MAP = {
    "Refuted":  "REFUTED_PROOF",
    "Verified": "VERIFIED",
    "Abstain":  "ABSTAIN",
    "N/A":      "NA",
}


def classify_fragment(repro_path: Path):
    """Return (in_fragment: bool, reason: str|None)."""
    if not repro_path.exists():
        return True, None  # missing → assume in-fragment, note elsewhere

    src = repro_path.read_text(errors="replace")
    for token, reason in OUT_OF_FRAGMENT_TOKENS.items():
        if token in src:
            return False, reason
    return True, None


def build_manifest():
    # -----------------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------------
    corpus = []
    with CORPUS_JSONL.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))

    with RESULTS_JSON.open() as fh:
        results = json.load(fh)

    verdict_map = {
        item["id"]: BUCKET_MAP.get(item["bucket"], "NA")
        for item in results["bug_corpus"]["per_input"]
    }

    # -----------------------------------------------------------------------
    # Build items
    # -----------------------------------------------------------------------
    notes = []
    missing_repros = []
    items = []

    for rec in corpus:
        bug_id   = rec["id"]
        repro_rel = rec.get("repro_file", "")
        repro_path = BASE.parent / repro_rel if repro_rel else None

        if repro_path is None or not repro_path.exists():
            missing_repros.append(bug_id)
            in_frag, oof_reason = True, None
        else:
            in_frag, oof_reason = classify_fragment(repro_path)

        tg_verdict = verdict_map.get(bug_id, "NA")

        items.append({
            "id":                       bug_id,
            "source_url":               rec.get("github_url", ""),
            "title":                    rec.get("title", ""),
            "category":                 rec.get("category", ""),
            "description":              rec.get("description", ""),
            "expected_error_substring": rec.get("expected_error_substring", ""),
            "in_fragment":              in_frag,
            "out_of_fragment_reason":   oof_reason,
            "tg_verdict":               tg_verdict,
            "repro_file":               repro_rel,
        })

    # -----------------------------------------------------------------------
    # Meta
    # -----------------------------------------------------------------------
    if missing_repros:
        notes.append(f"repro files missing for: {', '.join(missing_repros)}; in_fragment defaulted to true")

    tg_counts = {}
    for item in items:
        tg_counts[item["tg_verdict"]] = tg_counts.get(item["tg_verdict"], 0) + 1

    meta = {
        "total":        len(items),
        "tg_refuted":   tg_counts.get("REFUTED_PROOF", 0),
        "tg_verified":  tg_counts.get("VERIFIED", 0),
        "tg_abstain":   tg_counts.get("ABSTAIN", 0),
        "schema_version": 1,
        "generated_by": "experiments_v5/build_bug_corpus_manifest.py",
    }
    if notes:
        meta["notes"] = notes

    manifest = {"meta": meta, "items": items}

    # -----------------------------------------------------------------------
    # Write JSON
    # -----------------------------------------------------------------------
    with OUT_JSON.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {OUT_JSON} ({len(items)} items)")

    # -----------------------------------------------------------------------
    # Write Markdown
    # -----------------------------------------------------------------------
    n_total      = len(items)
    n_in_frag    = sum(1 for it in items if it["in_fragment"])
    n_out_frag   = n_total - n_in_frag
    n_refuted    = meta["tg_refuted"]
    n_verified   = meta["tg_verified"]
    n_abstain    = meta["tg_abstain"]

    from collections import Counter
    cat_tally = Counter(it["category"] for it in items)
    verdict_tally = Counter(it["tg_verdict"] for it in items)

    lines = []
    lines.append(f"# TensorGuard Bug Corpus Manifest (N={n_total})\n")
    lines.append(
        "**Provenance:** 60 historical PyTorch shape bugs mined from the "
        "pytorch/pytorch issue tracker; each has a self-contained ≤40-line "
        "CPU repro that raises the cited RuntimeError. TG verdicts use the "
        "v7 three-way refuted taxonomy (REFUTED_PROOF when a bug-corpus item "
        "is refuted, meaning TensorGuard successfully caught the bug; VERIFIED "
        "means TensorGuard reported safe — a silent miss; ABSTAIN means "
        "analysis was inconclusive; NA means no result recorded). "
        f"Of {n_total} bugs: {n_in_frag} are in-fragment (shape errors "
        f"expressible in TensorGuard's symbolic fragment) and {n_out_frag} "
        "are out-of-fragment (require torch.compile, custom autograd, "
        "external kernels, or similar). "
        f"TG caught {n_refuted} bugs (REFUTED_PROOF), silently missed "
        f"{n_verified} (VERIFIED), and abstained on {n_abstain}.\n"
    )
    lines.append("")

    # Table
    lines.append("| ID | Category | In-fragment | TG verdict | Source |")
    lines.append("|----|----------|-------------|------------|--------|")
    for it in items:
        in_frag_str = "yes" if it["in_fragment"] else f"no ({it['out_of_fragment_reason']})"
        issue_num = it["source_url"].rstrip("/").split("/")[-1]
        source_link = f"[{issue_num}]({it['source_url']})"
        lines.append(
            f"| {it['id']} | {it['category']} | {in_frag_str} "
            f"| {it['tg_verdict']} | {source_link} |"
        )

    lines.append("")
    lines.append("## Category tally\n")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    for cat, cnt in sorted(cat_tally.items()):
        lines.append(f"| {cat} | {cnt} |")

    lines.append("")
    lines.append("## TG verdict tally\n")
    lines.append("| Verdict | Count |")
    lines.append("|---------|-------|")
    for verdict, cnt in sorted(verdict_tally.items()):
        lines.append(f"| {verdict} | {cnt} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\nSummary: total={n_total}, in_fragment={n_in_frag}, "
          f"out_of_fragment={n_out_frag}, "
          f"REFUTED_PROOF={n_refuted}, VERIFIED={n_verified}, ABSTAIN={n_abstain}")


if __name__ == "__main__":
    build_manifest()
