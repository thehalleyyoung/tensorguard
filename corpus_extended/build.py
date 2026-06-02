"""Build the frozen *extended* corpus: materialize, runtime-validate, hash.

This script:

1. asks :mod:`corpus_extended.generators` for every parameterized case;
2. **executably validates each label against real PyTorch** -- it `exec`s the
   generated source, instantiates the module, and runs a real forward pass with
   the recorded input shapes. A ``buggy`` case must raise an exception whose
   message contains its ``expected_error_substring``; a ``clean`` case must run
   without error. Any violation aborts the build (the label is ground truth by
   construction, not by assertion);
3. writes each validated case to ``corpus_extended/cases/<id>.py`` and records a
   SHA-256 content hash in ``corpus_extended/manifest.json`` so the corpus is
   content-addressed and cannot silently drift.

Run ``python -m corpus_extended.build`` to (re)build, or ``--check`` to verify
the on-disk corpus matches a fresh build byte-for-byte without rewriting files.
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

from corpus_extended.generators import Case, all_cases  # noqa: E402

CASES_DIR = os.path.join(THIS_DIR, "cases")
MANIFEST_PATH = os.path.join(THIS_DIR, "manifest.json")
CORPUS_VERSION = "1.0.0"


class GroundTruthError(RuntimeError):
    """Raised when a generated case does not behave as its label claims."""


def _runtime_validate(case: Case) -> str:
    """Run the real module forward; return the live behavior string.

    Returns "" if the forward ran cleanly, else the exception text. Raises
    :class:`GroundTruthError` if the observed behavior contradicts the label.
    """
    import torch

    ns: dict = {}
    exec(compile(case.source, f"<{case.id}>", "exec"), ns)
    module_cls = ns.get("M")
    if module_cls is None:
        raise GroundTruthError(f"{case.id}: generated source defines no class M")
    module = module_cls()
    module.eval()
    args = [torch.randn(*shape) for shape in case.input_shapes.values()]

    error_text = ""
    try:
        with torch.no_grad():
            module(*args)
    except Exception as exc:  # noqa: BLE001 - we are probing for failures
        error_text = f"{type(exc).__name__}: {exc}"

    if case.label == "clean":
        if error_text:
            raise GroundTruthError(
                f"{case.id}: labeled clean but forward raised: {error_text}"
            )
    else:  # buggy
        if not error_text:
            raise GroundTruthError(
                f"{case.id}: labeled buggy but forward ran without error"
            )
        sub = case.expected_error_substring
        if sub and sub not in error_text:
            raise GroundTruthError(
                f"{case.id}: expected error substring {sub!r} not in {error_text!r}"
            )
    return error_text


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
        "repro_file": f"cases/{case.id}.py",
        "note": case.note,
        "sha256": hashlib.sha256(case.source.encode("utf-8")).hexdigest(),
    }


def build(check: bool = False) -> int:
    cases = all_cases()
    # Runtime-validate every case against real torch (ground truth).
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
            "name": "tensorguard-extended-benchmarks",
            "version": CORPUS_VERSION,
            "frozen": True,
            "total": len(cases),
            "buggy": n_buggy,
            "clean": n_clean,
            "families": dict(sorted(families.items())),
            "generated_by": "corpus_extended/build.py",
            "ground_truth": (
                "Every case is runtime-validated against real PyTorch at build "
                "time: buggy cases raise (message contains expected_error_"
                "substring); clean cases run cleanly."
            ),
        },
        "items": entries,
    }
    new_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    if check:
        problems = []
        if not os.path.exists(MANIFEST_PATH):
            problems.append("manifest.json missing")
        else:
            old = open(MANIFEST_PATH).read()
            if old != new_manifest:
                problems.append("manifest.json differs from fresh build")
        for case in cases:
            path = os.path.join(CASES_DIR, f"{case.id}.py")
            if not os.path.exists(path):
                problems.append(f"missing {case.id}.py")
                continue
            disk = open(path).read()
            if disk != case.source:
                problems.append(f"drift in {case.id}.py")
        if problems:
            print("CORPUS CHECK FAILED:")
            for p in problems[:20]:
                print("  -", p)
            return 1
        print(
            f"OK: extended corpus byte-identical ({len(cases)} cases, "
            f"{n_buggy} buggy / {n_clean} clean), all runtime-validated."
        )
        return 0

    os.makedirs(CASES_DIR, exist_ok=True)
    # Remove stale case files not in the current generation.
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
        f"Built extended corpus: {len(cases)} cases "
        f"({n_buggy} buggy / {n_clean} clean), all runtime-validated."
    )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(build(check=args.check))
