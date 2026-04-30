#!/usr/bin/env python3
"""
mine_hf_shape_bugs.py — Mechanically extract HuggingFace transformer shape bugs.

Selection protocol (both live and offline modes):
  - diff regex:  view(|reshape(|permute(|transpose(|matmul(  must appear in the PR diff
  - title/body:  shape|broadcast|dim|size mismatch  must appear in the PR title or body

Usage (offline fixture, pre-staged PR data):
    python scripts/mine_hf_shape_bugs.py \\
        --offline-fixture experiments_v5/hf_pr_seed_list.txt

In --offline-fixture mode the script reads:
  - <seed_list>                        — one commit SHA per non-comment line
  - <seed_list_dir>/hf_pr_fixture.json — pre-staged PR diff data

Outputs:
  experiments_v5/hf_natural_bugs_mechanical.json
  experiments_v5/hf_natural_bugs_mechanical/<pr_number>.py   (one per bug)

The JSON output is deterministic (byte-identical across reruns) modulo the
top-level "generated_at" timestamp field.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Filter regexes ────────────────────────────────────────────────────────────

DIFF_REGEX = re.compile(r"view\(|reshape\(|permute\(|transpose\(|matmul\(")
KEYWORD_PATTERN = re.compile(r"shape|broadcast|dim|size mismatch", re.IGNORECASE)

# ── Repository root (two levels above this script) ────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_tensorguard(source: str) -> str:
    """Run TensorGuard on source code and return a verdict string."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from src.model_checker import verify_model  # type: ignore[import]

        result = verify_model(source)
        return "BUGGY" if not result.safe else "SAFE"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def apply_filters(entry: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (passes, regex_hit_substring).

    An entry passes if its diff_snippet matches DIFF_REGEX and its title/body
    matches KEYWORD_PATTERN.
    """
    diff_match = DIFF_REGEX.search(entry.get("diff_snippet", ""))
    if diff_match is None:
        return False, ""

    kw_text = f"{entry.get('title', '')} {entry.get('body', '')}"
    if not KEYWORD_PATTERN.search(kw_text):
        return False, ""

    return True, diff_match.group(0)


def load_seed_shas(seed_list_path: Path) -> set:
    """Load commit SHAs from a seed list file (one SHA per non-comment line)."""
    shas: set = set()
    with open(seed_list_path) as fh:
        for raw in fh:
            line = raw.strip()
            if line and not line.startswith("#"):
                shas.add(line)
    return shas


def process_offline(seed_list_path: Path) -> None:
    """Process pre-staged fixture data and write results."""
    base_dir = seed_list_path.parent
    fixture_path = base_dir / "hf_pr_fixture.json"

    if not fixture_path.exists():
        print(
            "INFEASIBLE: HF PR mining requires network or a pre-staged diff fixture "
            "not present in repo",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(fixture_path) as fh:
        fixture: List[Dict[str, Any]] = json.load(fh)

    seed_shas = load_seed_shas(seed_list_path)

    # ── Filter: SHA membership + diff regex + keyword ─────────────────────────
    matched: List[Dict[str, Any]] = []
    for entry in fixture:
        if entry["commit_sha"] not in seed_shas:
            continue
        passes, regex_hit = apply_filters(entry)
        if not passes:
            continue
        matched.append({**entry, "_regex_hit": regex_hit})

    # ── Sort by pr_number for determinism ─────────────────────────────────────
    matched.sort(key=lambda e: e["pr_number"])

    # ── Set up output locations ───────────────────────────────────────────────
    extracted_dir = base_dir / "hf_natural_bugs_mechanical"
    extracted_dir.mkdir(exist_ok=True)
    output_json = base_dir / "hf_natural_bugs_mechanical.json"

    # ── Process each matched entry ────────────────────────────────────────────
    results = []
    tg_buggy_count = 0

    for entry in matched:
        pr_num = entry["pr_number"]
        family = entry["family"]
        buggy_code = entry["buggy_code"]
        regex_hit = entry["_regex_hit"]

        # Write extracted class
        extracted_path = extracted_dir / f"{pr_num}.py"
        extracted_path.write_text(buggy_code)

        # Run TensorGuard
        verdict = run_tensorguard(buggy_code)
        if verdict == "BUGGY":
            tg_buggy_count += 1

        # Identify which keywords matched
        kw_text = f"{entry['title']} {entry['body']}"
        keywords_matched = sorted(
            kw
            for kw in ["shape", "broadcast", "dim", "size mismatch"]
            if re.search(re.escape(kw), kw_text, re.IGNORECASE)
        )

        results.append(
            {
                "family": family,
                "ground_truth": "BUGGY",
                "pr": f"hf_pr_{pr_num}",
                "provenance": {
                    "commit_sha": entry["commit_sha"],
                    "keywords_matched": keywords_matched,
                    "pr_number": pr_num,
                    "regex_hit": regex_hit,
                    "title": entry["title"],
                },
                "tg_verdict": verdict,
            }
        )

    # ── Write deterministic JSON output ───────────────────────────────────────
    output = {
        "entries": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_protocol": (
            "Offline fixture mode: mechanically filtered by regex "
            "(view|reshape|permute|transpose|matmul) in diff and "
            "keywords (shape|broadcast|dim|size mismatch) in title/body; "
            "ordered by pr_number ascending"
        ),
    }

    with open(output_json, "w") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)
        fh.write("\n")

    families = {r["family"] for r in results}
    print(
        f"MECHANICAL_HF: {tg_buggy_count}/{len(results)} across {len(families)} families"
    )
    print(f"Output written to {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mechanically mine HuggingFace transformer shape bugs."
    )
    parser.add_argument(
        "--offline-fixture",
        metavar="SEED_LIST",
        help=(
            "Path to seed-SHA list file; use pre-staged hf_pr_fixture.json "
            "in the same directory instead of live git access."
        ),
    )
    args = parser.parse_args()

    if args.offline_fixture:
        seed_list = Path(args.offline_fixture)
        if not seed_list.exists():
            print(f"Error: seed list not found: {seed_list}", file=sys.stderr)
            sys.exit(1)
        process_offline(seed_list)
    else:
        print(
            "INFEASIBLE: HF PR mining requires network or a pre-staged diff fixture "
            "not present in repo",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
