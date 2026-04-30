"""
build_bug_corpus.py
===================

Documentation + integrity check for the v5 historical PyTorch shape-bug
corpus.  The corpus itself was assembled by mining the
``pytorch/pytorch`` (and adjacent) GitHub issue trackers; the search
protocol, query list, scan/keep counts, and limitations are recorded in
``experiments_v5/bug_corpus_protocol.md``.

This script:

  1. Re-validates ``v5_bug_corpus.jsonl`` against its schema.
  2. Re-executes every repro in ``experiments_v5/bug_repros/`` to
     confirm it still produces a ``RuntimeError`` whose message contains
     the recorded ``expected_error_substring``.
  3. Records SHA-256 of every repro so we can detect drift.

Run:  python3.11 experiments_v5/build_bug_corpus.py

Outputs (alongside the existing corpus):
  experiments_v5/v5_bug_corpus_integrity.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
JSONL = ROOT / "v5_bug_corpus.jsonl"
REPRO_DIR = ROOT / "bug_repros"
PROTOCOL_MD = ROOT / "bug_corpus_protocol.md"
OUT_INTEGRITY = ROOT / "v5_bug_corpus_integrity.json"

SCHEMA_FIELDS = {
    "id", "github_url", "title", "category", "is_buggy",
    "description", "repro_file", "expected_error_substring",
}

CATEGORIES = {
    "conv_channel_mismatch", "linear_inout_mismatch",
    "view_reshape_total_size", "broadcasting", "attention_dim",
    "einsum_dim", "transpose_axes", "batchnorm_features",
    "embedding_index", "other",
}


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _validate(rec: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    missing = SCHEMA_FIELDS - set(rec)
    if missing:
        errs.append(f"missing fields: {sorted(missing)}")
    if rec.get("category") not in CATEGORIES:
        errs.append(f"unknown category: {rec.get('category')}")
    if not rec.get("is_buggy"):
        errs.append("is_buggy must be true")
    if not str(rec.get("github_url", "")).startswith("https://github.com/"):
        errs.append("github_url malformed")
    repro = REPO / rec["repro_file"]
    if not repro.exists():
        errs.append(f"repro file missing: {rec['repro_file']}")
    return errs


def _run_repro(path: Path, expected: str) -> Dict[str, Any]:
    res = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True, text=True, timeout=30,
    )
    out = (res.stdout or "") + (res.stderr or "")
    return {
        "exit_code": res.returncode,
        "raises_expected": expected.lower() in out.lower(),
        "tail": out[-300:],
    }


def main():
    if not JSONL.exists():
        sys.exit(f"missing {JSONL}")
    if not PROTOCOL_MD.exists():
        sys.exit(f"missing {PROTOCOL_MD} (search-protocol doc)")

    records = [json.loads(l) for l in JSONL.open()]
    print(f"Loaded {len(records)} bug-corpus records.")
    print(f"Search protocol: {PROTOCOL_MD}")

    integrity: Dict[str, Any] = {
        "build_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(records),
        "schema_errors": [],
        "category_distribution": dict(Counter(r["category"] for r in records)),
        "per_repro": [],
        "summary": {"valid": 0, "broken_schema": 0,
                    "raises_expected": 0, "no_raise_or_wrong_msg": 0},
    }
    t0 = time.time()
    for rec in records:
        row: Dict[str, Any] = {"id": rec.get("id")}
        errs = _validate(rec)
        if errs:
            row["schema_errors"] = errs
            integrity["summary"]["broken_schema"] += 1
            integrity["per_repro"].append(row)
            continue
        integrity["summary"]["valid"] += 1
        repro = REPO / rec["repro_file"]
        row["sha256"] = _sha256(repro.read_bytes())
        try:
            r = _run_repro(repro, rec["expected_error_substring"])
            row.update(r)
            if r["raises_expected"]:
                integrity["summary"]["raises_expected"] += 1
            else:
                integrity["summary"]["no_raise_or_wrong_msg"] += 1
        except subprocess.TimeoutExpired:
            row["exit_code"] = -1
            row["raises_expected"] = False
            row["tail"] = "TIMEOUT"
            integrity["summary"]["no_raise_or_wrong_msg"] += 1
        integrity["per_repro"].append(row)

    integrity["elapsed_s"] = round(time.time() - t0, 1)
    OUT_INTEGRITY.write_text(json.dumps(integrity, indent=2))
    print(f"\nWrote {OUT_INTEGRITY}")
    print(json.dumps(integrity["summary"], indent=2))
    print("Categories:", integrity["category_distribution"])


if __name__ == "__main__":
    main()
