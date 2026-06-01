"""Generate the reproducibility artifact backing the README's corpus claims.

Writes ``reproducibility/real_benchmarks_audit.json`` capturing, for the frozen
corpus:

* ``n_total`` / ``n_clean`` / ``n_buggy`` — corpus composition.
* ``n_verdict_match`` — items whose TensorGuard verdict matches the frozen label.
* ``n_buggy_runtime_error`` — buggy models that raise a real ``RuntimeError`` when
  executed in eager PyTorch on this host (the empirical ground truth).

This makes the README's ``16/16`` and ``6/8`` claims auditable by
``reproducibility/audit_numeric_claims.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS_DIR)
OUT = os.path.join(REPO, "reproducibility", "real_benchmarks_audit.json")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

from real_benchmarks import load  # noqa: E402


def _runtime_raises(item):
    """Execute a buggy model in eager PyTorch; return (raised, error_repr)."""
    import torch

    path = os.path.join(THIS_DIR, item["repro_file"])
    spec = importlib.util.spec_from_file_location("rb_probe", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        M = mod.BuggyModule
        shapes = mod.INPUT_SHAPES
        inst = M()
        args = [torch.randn(*s) for s in shapes.values()]
        inst(*args)
        return False, None
    except RuntimeError as e:
        return True, str(e).splitlines()[0][:120]
    except Exception as e:  # noqa: BLE001 - device bug surfaces as AssertionError on CPU
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"


def build():
    ok, rows = load.check_corpus()
    items = {i["id"]: i for i in load.load_items(verify=True)}

    detail = []
    n_runtime = 0
    for r in rows:
        item = items[r["id"]]
        entry = {
            "id": r["id"],
            "label": r["label"],
            "expected_verdict": r["expected"],
            "tensorguard_verdict": r["actual"],
            "verdict_match": r["match"],
        }
        if r["label"] == "buggy":
            raised, err = _runtime_raises(item)
            entry["eager_runtime_error"] = raised
            entry["eager_error"] = err
            if raised:
                n_runtime += 1
        detail.append(entry)

    n_total = len(rows)
    n_clean = sum(1 for r in rows if r["label"] == "clean")
    n_buggy = sum(1 for r in rows if r["label"] == "buggy")
    n_match = sum(1 for r in rows if r["match"])

    manifest_meta = json.load(open(os.path.join(THIS_DIR, "manifest.json")))["meta"]
    artifact = {
        "corpus_version": manifest_meta["version"],
        "n_total": n_total,
        "n_clean": n_clean,
        "n_buggy": n_buggy,
        "n_verdict_match": n_match,
        "all_verdicts_match": ok,
        "n_buggy_runtime_error": n_runtime,
        "items": detail,
    }
    with open(OUT, "w") as fh:
        json.dump(artifact, fh, indent=2)
        fh.write("\n")
    return artifact


if __name__ == "__main__":
    a = build()
    print(
        f"real_benchmarks audit: {a['n_verdict_match']}/{a['n_total']} verdicts match, "
        f"{a['n_buggy_runtime_error']}/{a['n_buggy']} buggy raise eager RuntimeError."
    )
