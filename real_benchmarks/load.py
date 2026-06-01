"""Load the frozen TensorGuard benchmark corpus and verify its integrity.

This module reads ``manifest.json`` and, for every entry, re-hashes the repro
file on disk and compares it to the frozen ``sha256`` recorded at build time.
A mismatch raises :class:`CorpusIntegrityError`, guaranteeing that downstream
consumers (tests, the reproducibility harness) operate on exactly the corpus
that was frozen.

It also provides :func:`verify_item`, a thin wrapper that runs TensorGuard's
``verify_architecture`` on a corpus entry using the flags recorded in the
manifest, and :func:`check_corpus`, a CLI entry point that asserts every item's
TensorGuard verdict matches its frozen ground-truth label.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(THIS_DIR, "manifest.json")


class CorpusIntegrityError(RuntimeError):
    """Raised when a repro file's hash does not match the frozen manifest."""


def load_manifest():
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


def _sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def verify_integrity(manifest=None):
    """Re-hash every repro file and confirm it matches the frozen manifest.

    Returns the manifest on success; raises :class:`CorpusIntegrityError`
    listing every drifted or missing file otherwise.
    """
    manifest = manifest or load_manifest()
    problems = []
    for item in manifest["items"]:
        path = os.path.join(THIS_DIR, item["repro_file"])
        if not os.path.exists(path):
            problems.append(f"missing file: {item['repro_file']}")
            continue
        actual = _sha256_file(path)
        if actual != item["sha256"]:
            problems.append(
                f"hash drift: {item['repro_file']} "
                f"expected {item['sha256'][:12]} got {actual[:12]}"
            )
    if problems:
        raise CorpusIntegrityError(
            "frozen corpus integrity check failed:\n  " + "\n  ".join(problems)
        )
    return manifest


def load_items(verify=True):
    """Return the list of corpus items, optionally after an integrity check."""
    manifest = load_manifest()
    if verify:
        verify_integrity(manifest)
    return manifest["items"]


def read_source(item):
    with open(os.path.join(THIS_DIR, item["repro_file"])) as fh:
        return fh.read()


def verify_item(item, **overrides):
    """Run TensorGuard on a single corpus item and return the AnalysisResult.

    Uses the per-item ``check_devices`` / ``check_gradients`` flags recorded in
    the manifest. Additional keyword arguments override ``verify_architecture``
    parameters (e.g. ``max_cegar_iterations``).
    """
    # Import lazily so integrity checks don't require torch/z3.
    repo_src = os.path.dirname(THIS_DIR)
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    from src.api import verify_architecture

    source = read_source(item)
    shapes = {k: tuple(v) for k, v in item["input_shapes"].items()}
    kwargs = dict(
        input_shapes=shapes,
        check_devices=item.get("check_devices", False),
        check_gradients=item.get("check_gradients", False),
        max_cegar_iterations=0,
    )
    kwargs.update(overrides)
    return verify_architecture(source, **kwargs)


def check_corpus():
    """Verify integrity and that every item's verdict matches its label.

    Returns ``(ok, rows)`` where ``rows`` is a list of per-item result dicts.
    """
    items = load_items(verify=True)
    rows = []
    ok = True
    for item in items:
        result = verify_item(item)
        actual = "UNSAFE" if result.bug_count > 0 else "SAFE"
        match = actual == item["expected_verdict"]
        ok = ok and match
        rows.append(
            {
                "id": item["id"],
                "label": item["label"],
                "expected": item["expected_verdict"],
                "actual": actual,
                "bugs": result.bug_count,
                "match": match,
            }
        )
    return ok, rows


if __name__ == "__main__":
    ok, rows = check_corpus()
    width = max(len(r["id"]) for r in rows)
    for r in rows:
        flag = "OK " if r["match"] else "XX "
        print(
            f"{flag} {r['id']:<{width}}  label={r['label']:<5} "
            f"expected={r['expected']:<6} actual={r['actual']:<6} bugs={r['bugs']}"
        )
    n = len(rows)
    n_ok = sum(1 for r in rows if r["match"])
    print(f"\n{n_ok}/{n} items match frozen ground-truth labels.")
    sys.exit(0 if ok else 1)
