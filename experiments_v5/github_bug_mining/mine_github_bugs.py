#!/usr/bin/env python3
"""Mine real PyTorch shape/device bugs from public GitHub history.

This harness queries the GitHub Search API (issues + pull requests across all
public repositories) for the *exact runtime error signatures* PyTorch emits on
shape and device faults, then distills each hit into a labeled record. Because
the matched string is the literal PyTorch error message, the bug *category* is
high-confidence: an issue body containing ``mat1 and mat2 shapes cannot be
multiplied`` is, by construction, a real matmul/linear shape mismatch.

Output (under ``experiments_v5/github_bug_mining/``):
  * ``mined_bugs_dataset.jsonl`` -- one labeled bug per line (provenance URL,
    repo, title, matched signature, domain, category, state, created_at).
  * ``mined_bugs_manifest.json`` -- frozen manifest: total count, per-domain and
    per-category and per-signature breakdowns, the query set, the mining date,
    and a sha256 over the dataset so it cannot silently drift.

Reproducibility note: GitHub Search is a *live* index, so re-mining later will
generally return *more* hits (the corpus only grows). The committed dataset is
the frozen snapshot; ``load.py`` verifies its hash. Re-mining is a network
operation (this is a network-qualified artifact, analogous to the repo's
CUDA/HF/Lean artifacts).

Usage:
  GH must be authenticated (``gh auth status``). Then:
    python experiments_v5/github_bug_mining/mine_github_bugs.py --target 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "mined_bugs_dataset.jsonl")
MANIFEST = os.path.join(HERE, "mined_bugs_manifest.json")

# Each signature is a verbatim PyTorch runtime error fragment mapped to the
# (domain, category) it unambiguously denotes.
SIGNATURES = [
    ("mat1 and mat2 shapes cannot be multiplied", "shape", "matmul_linear_mismatch"),
    ("Given groups=1, weight of size", "shape", "conv_channel_mismatch"),
    ("is invalid for input of size", "shape", "view_reshape_total_size"),
    ("The size of tensor a", "shape", "broadcast_mismatch"),
    ("Sizes of tensors must match except in dimension", "shape", "cat_stack_mismatch"),
    ("Dimension out of range", "shape", "dim_out_of_range"),
    ("Expected all tensors to be on the same device", "device", "device_mismatch"),
    ("Input type", "device", "dtype_device_input_mismatch"),
]

# Heuristic guards: only keep device-signature hits that also look PyTorch-ish,
# since a couple of the signatures are short enough to appear elsewhere.
_PYTORCH_HINTS = ("torch", "cuda", "tensor", "pytorch", "nn.", "conv", "cudnn")


def _gh_search(query, page, per_page=100):
    """Return the parsed search/issues response for one page, or None on error."""
    argv = [
        "gh", "api", "-X", "GET", "search/issues",
        "-f", f"q={query}",
        "-f", f"per_page={per_page}",
        "-f", f"page={page}",
    ]
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")
        if "rate limit" in err.lower() or "403" in err:
            return "RATE_LIMIT"
        sys.stderr.write(f"search error (page {page}): {err[:200]}\n")
        return None
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def _looks_pytorch(item):
    blob = ((item.get("title") or "") + " " + (item.get("repository_url") or "")).lower()
    return any(h in blob for h in _PYTORCH_HINTS)


def mine(target=500, per_signature_pages=4, sleep_s=2.3, guard_device=True):
    seen = {}  # html_url -> record
    per_sig_counts = {}
    for sig, domain, category in SIGNATURES:
        query = f'"{sig}" language:Python'
        got = 0
        for page in range(1, per_signature_pages + 1):
            resp = _gh_search(query, page)
            if resp == "RATE_LIMIT":
                print(f"  rate-limited on '{sig}' page {page}; sleeping 65s")
                time.sleep(65)
                resp = _gh_search(query, page)
            if not resp or not resp.get("items"):
                break
            for item in resp["items"]:
                url = item.get("html_url")
                if not url or url in seen:
                    continue
                # Short device signatures: require a pytorch-ish hint.
                if guard_device and domain == "device" and not _looks_pytorch(item):
                    continue
                seen[url] = {
                    "id": f"{_repo_of(item)}#{item.get('number')}",
                    "source_url": url,
                    "repository": _repo_of(item),
                    "title": (item.get("title") or "").strip()[:200],
                    "is_pull_request": "pull_request" in item,
                    "state": item.get("state"),
                    "created_at": item.get("created_at"),
                    "matched_signature": sig,
                    "domain": domain,
                    "category": category,
                }
                got += 1
            time.sleep(sleep_s)
            if len(resp["items"]) < 100:
                break
        per_sig_counts[sig] = got
        print(f"  [{sig}] +{got}  (running unique total: {len(seen)})")
        if len(seen) >= target and target > 0:
            # Keep going through all signatures for category balance, but we've
            # cleared the bar.
            pass
    records = sorted(seen.values(), key=lambda r: (r["domain"], r["category"], r["source_url"]))
    return records, per_sig_counts


def _repo_of(item):
    rurl = item.get("repository_url", "")
    return rurl.replace("https://api.github.com/repos/", "")


def _sha256_lines(records):
    h = hashlib.sha256()
    for r in records:
        h.update((json.dumps(r, sort_keys=True) + "\n").encode("utf-8"))
    return h.hexdigest()


def write_dataset(records, per_sig_counts):
    with open(DATASET, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    by_domain, by_category = {}, {}
    for r in records:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1

    manifest = {
        "meta": {
            "name": "tensorguard-github-mined-bugs",
            "description": (
                "Real PyTorch shape/device bugs mined from public GitHub issues "
                "and PRs by matching verbatim PyTorch runtime error signatures. "
                "Each record's category is the category the matched signature "
                "denotes. Frozen snapshot; re-mining is a live network op that "
                "only grows the count."
            ),
            "mined_at_utc": datetime.now(timezone.utc).isoformat(),
            "total": len(records),
            "by_domain": dict(sorted(by_domain.items())),
            "by_category": dict(sorted(by_category.items())),
            "signatures": [
                {"signature": s, "domain": d, "category": c,
                 "unique_contributed": per_sig_counts.get(s, 0)}
                for (s, d, c) in SIGNATURES
            ],
            "dataset_sha256": _sha256_lines(records),
            "generated_by": "experiments_v5/github_bug_mining/mine_github_bugs.py",
            "network_qualified": True,
        }
    }
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--pages", type=int, default=4,
                    help="max search pages (x100 results) per signature")
    args = ap.parse_args()

    print(f"Mining real PyTorch shape/device bugs from GitHub (target >= {args.target})...")
    records, per_sig = mine(target=args.target, per_signature_pages=args.pages)
    manifest = write_dataset(records, per_sig)
    m = manifest["meta"]
    print(f"\nMined {m['total']} unique labeled bugs.")
    print(f"  by domain:   {m['by_domain']}")
    print(f"  by category: {m['by_category']}")
    print(f"  sha256:      {m['dataset_sha256'][:16]}")
    if m["total"] < args.target:
        print(f"WARNING: below target {args.target}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
