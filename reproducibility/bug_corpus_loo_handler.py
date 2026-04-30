#!/usr/bin/env python3.11
"""Real handler-LOO holdout (round 1, W5/Q5).

The previous LOO disabled `src/v5/*.py` orchestration modules whose
names happened to overlap a category label.  As the reviewer notes,
none of those modules contain operator handlers, so the LOO was a
literal no-op.

This script implements the holdout the reviewer asked for: for each
of the 10 bug categories, monkey-patch out the operator handler(s)
on the path of bugs in that category by stubbing the corresponding
shape-compute functions to return ``None`` (which makes the analyser
abstain on those nodes), then re-run TG on the full 60-bug corpus
and report the drop in RP-count.

Each category-to-handler mapping is given by
``CATEGORY_HANDLERS`` below.  We disable handlers by *patching* the
shape compute primitives that actually back the rule:
``compute_matmul_shape``, ``compute_broadcast_shape``,
``compute_reshape_shape``, plus removing entries from
``TORCH_SHAPE_OPS`` and friends.

Output: ``reproducibility/bug_corpus_loo_handler.json`` and ``.md``.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

OUT_JSON = os.path.join(ROOT, "reproducibility/bug_corpus_loo_handler.json")
OUT_MD = os.path.join(ROOT, "reproducibility/bug_corpus_loo_handler.md")
CORPUS = os.path.join(ROOT, "experiments_v5/v5_bug_corpus.jsonl")

# Map each bug category to the operator handlers most directly on its
# path.  When that category is held out, those handlers are stubbed.
CATEGORY_HANDLERS: Dict[str, List[str]] = {
    "view_reshape_total_size": ["view", "reshape"],
    "broadcasting": ["broadcast", "add", "mul", "sub", "div"],
    "conv_channel_mismatch": ["conv1d", "conv2d", "conv3d"],
    "linear_inout_mismatch": ["linear"],
    "einsum_dim": ["einsum", "matmul", "bmm"],
    "transpose_axes": ["transpose", "permute"],
    "attention_dim": ["scaled_dot_product_attention", "matmul", "bmm",
                      "softmax", "multihead_attention"],
    "batchnorm_features": ["batch_norm"],
    "embedding_index": ["embed", "index_select", "gather"],
    "other": [],
}


def _load_corpus() -> List[Dict[str, Any]]:
    items = []
    with open(CORPUS) as f:
        for ln in f:
            items.append(json.loads(ln))
    return items


def _read_repro(path: str) -> str:
    p = os.path.join(ROOT, path)
    with open(p) as f:
        return f.read()


def _score_corpus(disabled_handlers: List[str]) -> Dict[str, Any]:
    """Score the 60-bug corpus with the named handlers stubbed out."""
    # Re-import the analyser fresh each time to avoid leaking patches.
    for m in list(sys.modules):
        if m == "src" or m.startswith("src."):
            del sys.modules[m]

    import src.tensor_shapes as ts  # noqa: E402

    # Patch dispatch tables: drop entries for disabled ops.
    if disabled_handlers:
        for op in list(ts.TORCH_SHAPE_OPS):
            if op in disabled_handlers:
                ts.TORCH_SHAPE_OPS.pop(op, None)

    # Stub the shape compute primitives so that any handler that
    # falls back to them returns None (analyser then treats result as
    # unknown / abstain).
    if "matmul" in disabled_handlers or "bmm" in disabled_handlers:
        ts.compute_matmul_shape = lambda *a, **k: None
    if "broadcast" in disabled_handlers:
        ts.compute_broadcast_shape = lambda *a, **k: None
    if "view" in disabled_handlers or "reshape" in disabled_handlers:
        ts.compute_reshape_shape = lambda *a, **k: None

    # Also patch model_checker's bound names
    try:
        import src.model_checker as mc
        if "matmul" in disabled_handlers or "bmm" in disabled_handlers:
            mc.compute_matmul_shape = lambda *a, **k: None
        if "broadcast" in disabled_handlers:
            mc.compute_broadcast_shape = lambda *a, **k: None
        if "view" in disabled_handlers or "reshape" in disabled_handlers:
            mc.compute_reshape_shape = lambda *a, **k: None
    except Exception:
        pass

    # Drop modern ops dispatch
    try:
        import src.stdlib.modern_ops as mo
        for op in list(mo.MODERN_TORCH_SHAPE_OPS):
            if op in disabled_handlers:
                mo.MODERN_TORCH_SHAPE_OPS.pop(op, None)
    except Exception:
        pass
    try:
        import src.smt.encoder as enc
        for op in list(enc.FUNCTIONAL_SHAPE_RULES):
            if op in disabled_handlers:
                enc.FUNCTIONAL_SHAPE_RULES.pop(op, None)
    except Exception:
        pass

    # Now score.
    from src.api import verify_architecture  # noqa: E402

    items = _load_corpus()
    rp = silent = abst = err = 0
    per_cat: Dict[str, Dict[str, int]] = {}
    for it in items:
        cat = it["category"]
        per_cat.setdefault(cat, {"rp": 0, "silent": 0, "abst": 0, "err": 0, "n": 0})
        per_cat[cat]["n"] += 1
        try:
            src_str = _read_repro(it["repro_file"])
        except Exception:
            err += 1
            per_cat[cat]["err"] += 1
            continue
        try:
            r = verify_architecture(src_str)
            status = getattr(r, "status", "UNKNOWN")
            if status == "UNSAFE":
                rp += 1
                per_cat[cat]["rp"] += 1
            elif status == "SAFE":
                silent += 1
                per_cat[cat]["silent"] += 1
            else:
                abst += 1
                per_cat[cat]["abst"] += 1
        except Exception:
            err += 1
            per_cat[cat]["err"] += 1
    return {"rp": rp, "silent": silent, "abst": abst, "err": err,
            "per_category": per_cat}


def main():
    print("Scoring full pipeline (no LOO)...", flush=True)
    t0 = time.time()
    full = _score_corpus([])
    full["elapsed_s"] = round(time.time() - t0, 2)
    print(f"full RP={full['rp']}/60  ({full['elapsed_s']}s)", flush=True)

    runs: Dict[str, Dict[str, Any]] = {}
    for cat, handlers in CATEGORY_HANDLERS.items():
        if not handlers:
            print(f"[{cat}] skipped (no handler mapping)", flush=True)
            continue
        print(f"[{cat}] disabling handlers: {handlers}", flush=True)
        t0 = time.time()
        result = _score_corpus(handlers)
        result["disabled_handlers"] = handlers
        result["elapsed_s"] = round(time.time() - t0, 2)
        runs[cat] = result
        print(f"  RP={result['rp']}/60 (drop "
              f"{full['rp'] - result['rp']}); cat[{cat}] "
              f"{result['per_category'].get(cat, {}).get('rp', 0)} "
              f"vs full {full['per_category'].get(cat, {}).get('rp', 0)}",
              flush=True)

    out = {
        "_question": (
            "Reviewer W5 / Q5 (round 1).  The previous LOO disabled "
            "src/v5/*.py orchestration modules whose names happened to "
            "overlap a category label.  None of those modules contain "
            "operator handlers, so the LOO was a literal no-op (53/60 -> "
            "53/60).  This file documents the holdout that actually "
            "removes the operator handlers most directly on the path of "
            "each category's bugs and reports the resulting RP-count "
            "drop."),
        "category_to_handlers": CATEGORY_HANDLERS,
        "full_pipeline": full,
        "loo_runs": runs,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = ["# Bug-corpus handler LOO (real holdout)",
          "",
          "Reviewer W5/Q5 (round 1).  The earlier LOO disabled v5 ",
          "orchestration files whose names overlapped a category label; ",
          "those files contain no operator handlers, so the LOO was a ",
          "literal no-op (53/60 → 53/60).  This file replaces it with a ",
          "holdout that actually removes per-category handlers from the ",
          "shape dispatch tables (`TORCH_SHAPE_OPS`, ",
          "`MODERN_TORCH_SHAPE_OPS`, `FUNCTIONAL_SHAPE_RULES`) and stubs ",
          "the corresponding shape compute primitives.",
          "",
          f"## Full pipeline: **RP {full['rp']}/60**, silent {full['silent']}, "
          f"abstain {full['abst']}, error {full['err']}",
          "",
          "## Per-category drop after handler removal",
          "",
          "| category | disabled handlers | full RP (cat) | LOO RP (cat) | "
          "global RP drop |",
          "|---|---|---|---|---|"]
    for cat, r in runs.items():
        full_cat = full["per_category"].get(cat, {}).get("rp", 0)
        loo_cat = r["per_category"].get(cat, {}).get("rp", 0)
        drop = full["rp"] - r["rp"]
        md.append(f"| {cat} | `{', '.join(r['disabled_handlers'])}` | "
                  f"{full_cat} | {loo_cat} | {drop} |")
    md.append("")
    md.append("Each row holds out the named handlers and re-runs the "
              "full 60-bug corpus.  A non-zero per-category drop "
              "demonstrates that the held-out handlers are actually "
              "responsible for catching bugs in their category — the "
              "evidence the no-op LOO failed to provide.")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
