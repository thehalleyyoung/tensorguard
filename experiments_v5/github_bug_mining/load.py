#!/usr/bin/env python3
"""Load and integrity-check the frozen GitHub-mined bug dataset.

Offline: operates only on the committed ``mined_bugs_dataset.jsonl`` +
``mined_bugs_manifest.json``. Verifies the dataset's sha256 against the frozen
manifest and that every record's (domain, category) label is consistent with
the PyTorch error signature it was matched on.
"""

from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "mined_bugs_dataset.jsonl")
MANIFEST = os.path.join(HERE, "mined_bugs_manifest.json")


class MinedCorpusError(RuntimeError):
    pass


def load_records():
    with open(DATASET) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def _sha256(records):
    h = hashlib.sha256()
    for r in records:
        h.update((json.dumps(r, sort_keys=True) + "\n").encode("utf-8"))
    return h.hexdigest()


def signature_label_map():
    """signature -> (domain, category), read back from the frozen manifest."""
    m = load_manifest()
    return {s["signature"]: (s["domain"], s["category"]) for s in m["meta"]["signatures"]}


def verify_integrity():
    """Raise MinedCorpusError on hash drift or label inconsistency."""
    records = load_records()
    manifest = load_manifest()
    problems = []

    actual_hash = _sha256(records)
    if actual_hash != manifest["meta"]["dataset_sha256"]:
        problems.append(
            f"hash drift: manifest {manifest['meta']['dataset_sha256'][:12]} "
            f"!= dataset {actual_hash[:12]}"
        )
    if len(records) != manifest["meta"]["total"]:
        problems.append(
            f"count drift: manifest {manifest['meta']['total']} != "
            f"dataset {len(records)}"
        )

    sigmap = signature_label_map()
    urls = set()
    for r in records:
        url = r.get("source_url", "")
        if "github.com" not in url:
            problems.append(f"non-GitHub url: {url!r}")
        if url in urls:
            problems.append(f"duplicate url: {url}")
        urls.add(url)
        sig = r.get("matched_signature")
        if sig not in sigmap:
            problems.append(f"unknown signature: {sig!r}")
            continue
        exp_domain, exp_cat = sigmap[sig]
        if (r.get("domain"), r.get("category")) != (exp_domain, exp_cat):
            problems.append(
                f"label mismatch for {r.get('id')}: "
                f"{(r.get('domain'), r.get('category'))} != {(exp_domain, exp_cat)}"
            )

    # Per-category counts must agree with the manifest breakdown.
    by_cat = {}
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    if by_cat != manifest["meta"]["by_category"]:
        problems.append(
            f"category breakdown drift: {by_cat} != {manifest['meta']['by_category']}"
        )

    if problems:
        raise MinedCorpusError(
            "mined corpus integrity failed:\n  " + "\n  ".join(problems[:20])
        )
    return records


if __name__ == "__main__":
    recs = verify_integrity()
    m = load_manifest()["meta"]
    print(f"OK: {len(recs)} mined bugs, hash {m['dataset_sha256'][:16]} verified.")
    print(f"  by domain:   {m['by_domain']}")
    print(f"  by category: {m['by_category']}")
