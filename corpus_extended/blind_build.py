"""Build the frozen *blind* split: materialize, runtime-validate, hash.

Mirrors :mod:`corpus_extended.build` but for the held-out blind split
(:mod:`corpus_extended.blind_split`). Every case is executably validated against
real PyTorch, written to ``corpus_extended/blind_cases/<id>.py`` and recorded in
``corpus_extended/blind_manifest.json`` with a SHA-256 content hash, so the
split is frozen and content-addressed. ``--check`` re-verifies the on-disk split
byte-for-byte without rewriting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from corpus_extended.blind_split import all_blind_cases  # noqa: E402
from corpus_extended.build import GroundTruthError, _runtime_validate  # noqa: E402
from corpus_extended.generators import Case  # noqa: E402

CASES_DIR = os.path.join(THIS_DIR, "blind_cases")
MANIFEST_PATH = os.path.join(THIS_DIR, "blind_manifest.json")
SPLIT_VERSION = "1.0.0"


def _manifest_entry(case: Case) -> dict:
    return {
        "id": case.id,
        "family": case.family,
        "label": case.label,
        "domain": case.domain,
        "provenance_type": case.provenance_type,
        "seed_url": case.seed_url,
        "expected_error_substring": case.expected_error_substring,
        "expected_verdict": "SAFE" if case.label == "clean" else "UNSAFE",
        "input_shapes": {k: list(v) for k, v in case.input_shapes.items()},
        "repro_file": f"blind_cases/{case.id}.py",
        "note": case.note,
        "sha256": hashlib.sha256(case.source.encode("utf-8")).hexdigest(),
    }


def build(check: bool = False) -> int:
    cases = all_blind_cases()
    for case in cases:
        _runtime_validate(case)

    entries = [_manifest_entry(c) for c in cases]
    n_buggy = sum(1 for c in cases if c.label == "buggy")
    n_clean = sum(1 for c in cases if c.label == "clean")
    families: dict = {}
    for c in cases:
        families[c.family] = families.get(c.family, 0) + 1
    manifest = {
        "meta": {
            "name": "tensorguard-blind-split",
            "version": SPLIT_VERSION,
            "frozen": True,
            "held_out": True,
            "disjoint_from": "tensorguard-extended-benchmarks",
            "total": len(cases),
            "buggy": n_buggy,
            "clean": n_clean,
            "families": dict(sorted(families.items())),
            "generated_by": "corpus_extended/blind_build.py",
            "ground_truth": (
                "Every case runtime-validated against real PyTorch at build "
                "time; generated from parameter grids disjoint from the dev "
                "corpus so no case id can collide."
            ),
        },
        "items": entries,
    }
    new_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    if check:
        problems = []
        if not os.path.exists(MANIFEST_PATH):
            problems.append("blind_manifest.json missing")
        else:
            old = open(MANIFEST_PATH).read()
            if old != new_manifest:
                problems.append("blind_manifest.json differs from fresh build")
        for case in cases:
            path = os.path.join(CASES_DIR, f"{case.id}.py")
            if not os.path.exists(path):
                problems.append(f"missing {case.id}.py")
                continue
            if open(path).read() != case.source:
                problems.append(f"drift in {case.id}.py")
        if problems:
            print("BLIND SPLIT CHECK FAILED:")
            for p in problems[:20]:
                print("  -", p)
            return 1
        print(
            f"OK: blind split byte-identical ({len(cases)} cases, "
            f"{n_buggy} buggy / {n_clean} clean), all runtime-validated."
        )
        return 0

    os.makedirs(CASES_DIR, exist_ok=True)
    current = {f"{c.id}.py" for c in cases}
    for existing in os.listdir(CASES_DIR):
        if existing.endswith(".py") and existing not in current:
            os.remove(os.path.join(CASES_DIR, existing))
    for case in cases:
        with open(os.path.join(CASES_DIR, f"{case.id}.py"), "w") as fh:
            fh.write(case.source)
    with open(MANIFEST_PATH, "w") as fh:
        fh.write(new_manifest)
    print(
        f"Built blind split: {len(cases)} cases "
        f"({n_buggy} buggy / {n_clean} clean), all runtime-validated."
    )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(build(check=args.check))
