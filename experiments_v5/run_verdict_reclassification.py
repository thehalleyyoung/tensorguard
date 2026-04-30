#!/usr/bin/env python3.11
"""
run_verdict_reclassification.py

Reclassify all Refuted items in the v5 benchmark results into the three
sub-verdicts: REFUTED_PROOF, CONTRACT_VIOLATION, LIBRARY_WARN.

Writes experiments_v5/verdict_reclassification.json.
"""

from __future__ import annotations

import json
import os
import sys

# Allow importing from src/v5
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.v5.verdict_taxonomy import Verdict, classify_refutation  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = os.path.join(_REPO_ROOT, "experiments_v5")
RESULTS_PATH = os.path.join(EXPERIMENTS_DIR, "v5_benchmark_results.json")
BLOCK_CORPUS_PATH = os.path.join(EXPERIMENTS_DIR, "v5_block_corpus.jsonl")
BUG_CORPUS_PATH = os.path.join(EXPERIMENTS_DIR, "v5_bug_corpus.jsonl")
OUTPUT_PATH = os.path.join(EXPERIMENTS_DIR, "verdict_reclassification.json")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> dict[str, dict]:
    """Load a .jsonl file and return a dict keyed by the 'id' field."""
    items: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items[obj["id"]] = obj
    return items


def main() -> None:
    print("Loading benchmark results …")
    with open(RESULTS_PATH) as fh:
        results = json.load(fh)

    print("Loading block corpus sources …")
    block_sources = load_jsonl(BLOCK_CORPUS_PATH)

    print("Loading bug corpus …")
    bug_corpus = load_jsonl(BUG_CORPUS_PATH)

    # -----------------------------------------------------------------------
    # Block corpus
    # -----------------------------------------------------------------------
    block_refuted = [
        item
        for item in results["block_corpus"]["per_input"]
        if item.get("bucket") == "Refuted"
    ]
    print(f"Block-corpus Refuted items: {len(block_refuted)}")

    block_per_item: list[dict] = []
    block_counts: dict[str, int] = {
        "REFUTED_PROOF": 0,
        "CONTRACT_VIOLATION": 0,
        "LIBRARY_WARN": 0,
    }
    by_library: dict[str, dict[str, int]] = {}

    for item in block_refuted:
        src_entry = block_sources.get(item["id"])
        source = src_entry.get("source") if src_entry else None
        verdict = classify_refutation(item, source=source)
        vname = verdict.name
        block_counts[vname] = block_counts.get(vname, 0) + 1

        lib = item.get("library", "unknown")
        by_library.setdefault(lib, {"REFUTED_PROOF": 0, "CONTRACT_VIOLATION": 0, "LIBRARY_WARN": 0})
        by_library[lib][vname] = by_library[lib].get(vname, 0) + 1

        block_per_item.append({
            "id": item["id"],
            "library": item.get("library", "unknown"),
            "category": item.get("category", ""),
            "verdict": vname,
        })

    # -----------------------------------------------------------------------
    # Bug corpus
    # -----------------------------------------------------------------------
    bug_refuted = [
        item
        for item in results["bug_corpus"]["per_input"]
        if item.get("bucket") == "Refuted"
    ]
    print(f"Bug-corpus Refuted items: {len(bug_refuted)}")

    bug_per_item: list[dict] = []
    for item in bug_refuted:
        bug_per_item.append({
            "id": item["id"],
            "category": item.get("category", ""),
            "verdict": Verdict.REFUTED_PROOF.name,
        })

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    output = {
        "meta": {
            "total_block_refutations": len(block_refuted),
            "total_bug_refutations": len(bug_refuted),
        },
        "block_corpus": {
            **block_counts,
            "by_library": by_library,
            "per_item": block_per_item,
        },
        "bug_corpus": {
            "REFUTED_PROOF": len(bug_refuted),
            "per_item": bug_per_item,
        },
    }

    with open(OUTPUT_PATH, "w") as fh:
        json.dump(output, fh, indent=2)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("VERDICT RECLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Block-corpus refutations ({len(block_refuted)} total):")
    print(f"  REFUTED_PROOF      : {block_counts['REFUTED_PROOF']:>4}")
    print(f"  CONTRACT_VIOLATION : {block_counts['CONTRACT_VIOLATION']:>4}")
    print(f"  LIBRARY_WARN       : {block_counts['LIBRARY_WARN']:>4}")
    print()
    print(f"Bug-corpus refutations ({len(bug_refuted)} total):")
    print(f"  REFUTED_PROOF      : {len(bug_refuted):>4}")
    print()
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
