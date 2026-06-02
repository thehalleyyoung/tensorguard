#!/usr/bin/env python3
"""Measured CEGAR convergence on the frozen real-bug corpus (Step 92).

`src/cegar_convergence_theory.py` proves a *tight* convergence bound for the
shape-predicate CEGAR loop: it terminates in at most ``k = |P_final \\ P_seed|``
iterations (Houdini-style monotone predicate accumulation), which is far below
the naive predicate-universe bound ``|P_prog| = layers x dims x kinds``.

A proof is only as convincing as its agreement with the running code. This
harness *runs the real loop* (`src.shape_cegar.run_shape_cegar`) on every model
in the frozen `real_benchmarks` corpus and checks the theorem's two empirical
predictions on each one:

1. **Tight bound holds.** Observed ``iterations <= 1 + |discovered_predicates|``
   — each productive refinement iteration adds at least one new predicate, plus
   one terminal iteration that confirms SAFE or a real bug.
2. **The naive bound is wildly loose.** Observed iterations sit far below the
   naive ``layers x max_dims x 7`` predicate-universe bound, quantified per model
   as an improvement factor.

The loop is deterministic (Z3 over a fixed encoding), so per-model iteration and
predicate counts are stable; the harness records no wall-clock field and supports
``--check`` (regenerate and byte-diff) for the reproducibility pipeline.

Usage::

    cd tensorguard && PYTHONPATH=. python3 reproducibility/cegar_convergence.py
    cd tensorguard && PYTHONPATH=. python3 reproducibility/cegar_convergence.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.cegar_convergence_theory import (  # noqa: E402
    ConvergenceTheoremStatement,
    compute_tight_iteration_bound,
)
from src.shape_cegar import run_shape_cegar  # noqa: E402

MANIFEST = os.path.join(REPO_ROOT, "real_benchmarks", "manifest.json")
OUT_JSON = os.path.join(THIS_DIR, "cegar_convergence.json")
OUT_MD = os.path.join(THIS_DIR, "cegar_convergence.md")

MAX_ITERATIONS = 10
PREDICATE_KINDS = 7

_LAYER_RE = re.compile(r"\bnn\.[A-Z][A-Za-z0-9_]*\s*\(")


def _round(x, nd=4):
    return None if x is None else round(float(x), nd)


def _source_path(stem: str) -> Optional[str]:
    for sub in ("clean", "buggy"):
        p = os.path.join(REPO_ROOT, "real_benchmarks", sub, stem + ".py")
        if os.path.exists(p):
            return p
    return None


def _count_layers(src: str) -> int:
    """Parameterised-layer count (an upper proxy for graph edges ``|E|``)."""
    return max(1, len(_LAYER_RE.findall(src)))


def _max_rank(shapes: Dict[str, Any]) -> int:
    ranks = [len(v) for v in (shapes or {}).values() if isinstance(v, (list, tuple))]
    return max(ranks) if ranks else 2


def measure() -> List[Dict[str, Any]]:
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    rows: List[Dict[str, Any]] = []
    for it in manifest["items"]:
        stem = it["id"]
        path = _source_path(stem)
        if path is None:
            continue
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        shapes_raw = it.get("input_shapes") or {}
        shapes = {k: tuple(v) for k, v in shapes_raw.items()}
        result = run_shape_cegar(
            src, input_shapes=shapes or None, max_iterations=MAX_ITERATIONS)

        iterations = int(result.iterations)
        n_pred = len(result.discovered_predicates)
        layers = _count_layers(src)
        max_dims = _max_rank(shapes)
        bounds = compute_tight_iteration_bound(
            num_layers=layers, max_dims_per_layer=max_dims,
            num_predicate_kinds=PREDICATE_KINDS, estimated_coverage=0.0)
        naive = bounds["naive_bound"]

        tight_holds = iterations <= 1 + n_pred
        below_naive = iterations <= naive
        rows.append({
            "id": stem,
            "domain": it.get("domain"),
            "label": it.get("label"),
            "iterations": iterations,
            "discovered_predicates": n_pred,
            "predicates": sorted(p.pretty() for p in result.discovered_predicates),
            "final_status": result.final_status.name,
            "layers": layers,
            "max_dims": max_dims,
            "naive_bound": naive,
            "tight_bound_prediction": 1 + n_pred,
            "improvement_factor": _round(naive / max(iterations, 1), 2),
            "tight_bound_holds": tight_holds,
            "iterations_below_naive": below_naive,
        })
    rows.sort(key=lambda r: r["id"])
    return rows


def run(check: bool = False) -> Dict[str, Any]:
    rows = measure()
    if not rows:
        raise SystemExit("no corpus models measured")

    iters = [r["iterations"] for r in rows]
    n = len(rows)
    all_tight = all(r["tight_bound_holds"] for r in rows)
    all_below = all(r["iterations_below_naive"] for r in rows)
    mean_iters = sum(iters) / n

    artifact = {
        "meta": {
            "generated_by": "reproducibility/cegar_convergence.py",
            "command": "python3 reproducibility/cegar_convergence.py",
            "runs": "src.shape_cegar.run_shape_cegar (the real loop)",
            "corpus": "real_benchmarks (frozen)",
            "n_models": n,
            "max_iterations_budget": MAX_ITERATIONS,
            "predicate_kinds": PREDICATE_KINDS,
            "determinism": (
                "Z3 over a fixed encoding; per-model iteration and predicate "
                "counts are stable. No wall-clock field is recorded."
            ),
            "theorem": ConvergenceTheoremStatement().to_dict(),
            "tight_bound_prediction": (
                "iterations <= 1 + |discovered_predicates| (one terminal "
                "iteration past the last predicate-adding refinement)"
            ),
        },
        "summary": {
            "n_models": n,
            "max_iterations": max(iters),
            "min_iterations": min(iters),
            "mean_iterations": _round(mean_iters),
            "iteration_histogram": {
                str(k): iters.count(k) for k in sorted(set(iters))},
            "max_discovered_predicates": max(
                r["discovered_predicates"] for r in rows),
            "tight_bound_holds_all": all_tight,
            "iterations_below_naive_all": all_below,
            "max_naive_bound": max(r["naive_bound"] for r in rows),
            "max_improvement_factor": max(
                r["improvement_factor"] for r in rows),
        },
        "per_model": rows,
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            current = fh.read()
        if current != text:
            raise SystemExit("cegar_convergence.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(artifact: Dict[str, Any]) -> str:
    s = artifact["summary"]
    meta = artifact["meta"]
    hist = ", ".join(f"{k} iter -> {v}" for k, v in s["iteration_histogram"].items())
    lines = [
        "# Measured CEGAR convergence on the frozen corpus",
        "",
        "_Generated by `reproducibility/cegar_convergence.py` by running the "
        "real `src.shape_cegar.run_shape_cegar` loop. Do not edit by hand._",
        "",
        f"- Models: **{s['n_models']}** (frozen `real_benchmarks`)",
        f"- Iterations observed: min {s['min_iterations']}, "
        f"max **{s['max_iterations']}**, mean {s['mean_iterations']} "
        f"({hist}).",
        f"- Most predicates discovered on any model: "
        f"{s['max_discovered_predicates']}.",
        f"- **Tight bound `iterations <= 1 + |discovered_predicates|` holds on "
        f"every model: {s['tight_bound_holds_all']}.**",
        f"- Every run stays below the naive `layers x dims x 7` bound: "
        f"{s['iterations_below_naive_all']} (largest naive bound "
        f"{s['max_naive_bound']}, best improvement "
        f"{s['max_improvement_factor']} times).",
        "",
        "## Theorem (validated)",
        "",
        f"**{meta['theorem']['theorem_name']}.** {meta['theorem']['conclusion']}",
        "",
        f"_{meta['theorem']['corollary']}_",
        "",
        "## Per-model",
        "",
        "| Model | status | iters | preds | 1+preds | naive | improv | tight ok |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in artifact["per_model"]:
        lines.append(
            f"| {r['id']} | {r['final_status']} | {r['iterations']} | "
            f"{r['discovered_predicates']} | {r['tight_bound_prediction']} | "
            f"{r['naive_bound']} | {r['improvement_factor']}x | "
            f"{'yes' if r['tight_bound_holds'] else 'NO'} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="regenerate in-memory and fail if the committed "
                         "cegar_convergence.json differs")
    args = ap.parse_args()
    artifact = run(check=args.check)
    s = artifact["summary"]
    if args.check:
        print("cegar_convergence.json OK (byte-identical)")
    else:
        print("Wrote", os.path.relpath(OUT_JSON, REPO_ROOT))
        print("Wrote", os.path.relpath(OUT_MD, REPO_ROOT))
    print(f"  n={s['n_models']}  max_iters={s['max_iterations']}  "
          f"mean={s['mean_iterations']}  tight_bound_holds_all="
          f"{s['tight_bound_holds_all']}  below_naive_all="
          f"{s['iterations_below_naive_all']}")
    if not (s["tight_bound_holds_all"] and s["iterations_below_naive_all"]):
        raise SystemExit("convergence theorem VIOLATED on the real corpus")


if __name__ == "__main__":
    main()
