#!/usr/bin/env python3
"""Localization-effort proxy for the developer user study (Step 91).

A central user-facing claim is that TensorGuard does not merely *report* that a
module is unsafe but **points the developer at the offending line**, so bugs are
fixed faster.  A full human randomized controlled trial (RCT) is specified and
pre-registered in ``docs/user_study/protocol.md``; this harness computes the
*localization-effort proxy* that the literature (Parnin & Orso, ISSTA 2011;
fault-localization "lines-inspected" / EXAM-score studies) uses as a measurable
stand-in for time-to-localize, on the frozen real-bug corpus.

For every refuted real bug that carries an author-placed ``# BUG`` marker we
already know (from ``reproducibility/localization_marker_only_n30.json``) the
distance ``dist_v5`` between TensorGuard's reported line and the ground-truth
bug line.  We turn this into two *paired*, same-unit effort measurements:

* **TensorGuard-assisted effort** — a developer starts at TG's reported line and
  scans outward until reaching the true bug: ``effort = dist_v5 + 1`` lines.
* **Unaided effort** — with no localizer the developer scans the module's
  executable lines in arbitrary order; the *expected* number inspected before
  hitting the bug among ``N`` candidate lines is ``(N + 1) / 2`` (uniform random
  inspection order, the standard neutral model).  ``N`` is the count of
  executable source lines in the repro, recomputed deterministically here.

We then report **effect sizes**, not just point estimates:

* **Cliff's delta** (non-parametric; the effect size paired with Mann–Whitney),
  robust for this small, skewed sample, with a bootstrap CI.
* **Cohen's d** and small-sample-corrected **Hedges' g** on the paired
  per-bug reduction.
* The **median per-bug reduction factor** ``unaided / assisted`` with a seeded
  percentile-bootstrap CI.

The harness is deliberately a *consumer* of the committed localization artifact
plus the committed repro sources, so it is fast, torch-free and deterministic,
and supports ``--check`` (regenerate and diff) for the reproducibility pipeline.

Honesty: this is a *proxy*, not a human study.  Items where TG misleads
(``assisted > unaided``) are kept and reported; the human RCT remains future
work, pre-registered in the protocol document.

Usage::

    cd tensorguard && PYTHONPATH=. python3 evaluation/localization_effort.py
    cd tensorguard && PYTHONPATH=. python3 evaluation/localization_effort.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.statistical_rigor import (  # noqa: E402
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    hedges_g,
)

LOCALIZATION_JSON = os.path.join(
    REPO_ROOT, "reproducibility", "localization_marker_only_n30.json"
)
OUT_JSON = os.path.join(THIS_DIR, "localization_effort.json")
OUT_MD = os.path.join(THIS_DIR, "localization_effort.md")

CORPORA = [
    os.path.join(REPO_ROOT, "experiments_v5", "v8", "real_bugs_upstream"),
    os.path.join(REPO_ROOT, "experiments_v5", "v8", "real_bugs_postfreeze"),
    os.path.join(REPO_ROOT, "experiments_v5", "v8", "real_bugs_unfiltered"),
]

BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 0
CONFIDENCE = 0.95


def _round(x, nd=6):
    return None if x is None else round(float(x), nd)


def _find_repro(stem: str) -> Optional[str]:
    for cdir in CORPORA:
        p = os.path.join(cdir, stem + ".py")
        if os.path.exists(p):
            return p
    return None


def count_executable_lines(src: str) -> int:
    """Deterministic count of executable source lines (the search space ``N``).

    Blank lines, comment lines, ``import``/``from`` statements and triple-quoted
    docstring blocks are excluded — these are not plausible sites for a shape /
    dtype bug a developer would inspect.
    """
    n = 0
    in_doc = False
    for raw in src.splitlines():
        s = raw.strip()
        if not s:
            continue
        if in_doc:
            if s.endswith('"""') or s.endswith("'''"):
                in_doc = False
            continue
        if s.startswith('"""') or s.startswith("'''"):
            # toggle only if the block does not close on the same line
            body = s[3:]
            if not (body.endswith('"""') or body.endswith("'''")):
                in_doc = True
            continue
        if s.startswith("#"):
            continue
        if s.startswith(("import ", "from ")):
            continue
        n += 1
    return n


def collect() -> Dict[str, Any]:
    with open(LOCALIZATION_JSON, "r", encoding="utf-8") as fh:
        loc = json.load(fh)

    per_bug: List[Dict[str, Any]] = []
    for it in loc["per_item"]:
        if not it.get("refuted"):
            continue
        dist = it.get("dist_v5")
        if dist is None:
            continue
        path = _find_repro(it["id"])
        if path is None:
            continue
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        n_exec = count_executable_lines(src)
        assisted = float(dist + 1)
        unaided = (n_exec + 1) / 2.0
        per_bug.append({
            "id": it["id"],
            "corpus": it.get("corpus"),
            "gt_line": it.get("gt_line"),
            "tg_line_v5": it.get("tg_line_v5"),
            "dist_v5": dist,
            "search_space_lines": n_exec,
            "assisted_effort_lines": assisted,
            "unaided_effort_lines": unaided,
            "reduction_factor": _round(unaided / assisted) if assisted else None,
            "tg_helped": bool(assisted < unaided),
        })
    per_bug.sort(key=lambda r: r["id"])
    return {"localization_source": os.path.relpath(LOCALIZATION_JSON, REPO_ROOT),
            "per_bug": per_bug}


def run(check: bool = False) -> Dict[str, Any]:
    data = collect()
    per_bug = data["per_bug"]
    if not per_bug:
        raise SystemExit("no refuted+localized bugs found; cannot compute proxy")

    assisted = [r["assisted_effort_lines"] for r in per_bug]
    unaided = [r["unaided_effort_lines"] for r in per_bug]
    ratios = [u / a for u, a in zip(unaided, assisted)]

    cd = cliffs_delta(unaided, assisted)        # unaided > assisted favourable
    cohen = cohens_d(unaided, assisted)
    g = hedges_g(unaided, assisted)

    delta_ci = bootstrap_ci(
        list(range(len(per_bug))),
        lambda idx: cliffs_delta([unaided[i] for i in idx],
                                 [assisted[i] for i in idx]).value,
        n_resamples=BOOTSTRAP_RESAMPLES, confidence=CONFIDENCE, seed=BOOTSTRAP_SEED,
    )
    ratio_ci = bootstrap_ci(
        ratios, statistics.median,
        n_resamples=BOOTSTRAP_RESAMPLES, confidence=CONFIDENCE, seed=BOOTSTRAP_SEED,
    )

    n = len(per_bug)
    n_helped = sum(1 for r in per_bug if r["tg_helped"])

    artifact = {
        "meta": {
            "generated_by": "evaluation/localization_effort.py",
            "command": "python3 evaluation/localization_effort.py",
            "reads": data["localization_source"],
            "n_bugs": n,
            "measure": (
                "lines a developer inspects to reach the true bug; assisted = "
                "dist_v5 + 1 (scan outward from TG's reported line); unaided = "
                "(N+1)/2 expected linear scan over N executable lines"
            ),
            "design": "paired, within-bug; same line unit for both arms",
            "effect_sizes": "Cliff's delta (primary), Cohen's d, Hedges' g",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "proxy_disclaimer": (
                "This is a localization-effort proxy on the frozen corpus, not a "
                "human RCT. The pre-registered human study is specified in "
                "docs/user_study/protocol.md. Bugs where TG misleads are kept."
            ),
        },
        "summary": {
            "n_bugs": n,
            "n_tg_helped": n_helped,
            "n_tg_hurt": n - n_helped,
            "median_assisted_lines": _round(statistics.median(assisted)),
            "median_unaided_lines": _round(statistics.median(unaided)),
            "mean_assisted_lines": _round(statistics.fmean(assisted)),
            "mean_unaided_lines": _round(statistics.fmean(unaided)),
            "median_reduction_factor": _round(statistics.median(ratios)),
            "median_reduction_factor_ci": [
                _round(ratio_ci.ci_low), _round(ratio_ci.ci_high)],
            "cliffs_delta": _round(cd.value),
            "cliffs_delta_magnitude": cd.magnitude,
            "cliffs_delta_ci": [_round(delta_ci.ci_low), _round(delta_ci.ci_high)],
            "cohens_d": _round(cohen.value),
            "cohens_d_magnitude": cohen.magnitude,
            "hedges_g": _round(g.value),
            "hedges_g_magnitude": g.magnitude,
        },
        "per_bug": per_bug,
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            current = fh.read()
        if current != text:
            raise SystemExit("localization_effort.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(artifact: Dict[str, Any]) -> str:
    meta = artifact["meta"]
    s = artifact["summary"]
    lines = [
        "# Localization-effort proxy — does TensorGuard help developers localize "
        "bugs?",
        "",
        "_Generated by `evaluation/localization_effort.py` from "
        f"`{meta['reads']}`. Do not edit by hand._",
        "",
        f"- Bugs analyzed: **{s['n_bugs']}** (refuted real bugs with a `# BUG` "
        "marker)",
        f"- Measure: {meta['measure']}",
        f"- Design: {meta['design']}",
        "",
        "> **Proxy, not an RCT.** " + meta["proxy_disclaimer"],
        "",
        "## Headline",
        "",
        f"- Median lines inspected: **{s['median_assisted_lines']}** with "
        f"TensorGuard vs **{s['median_unaided_lines']}** unaided.",
        f"- Median per-bug reduction factor: **{s['median_reduction_factor']}×** "
        f"(95% bootstrap CI {s['median_reduction_factor_ci'][0]}–"
        f"{s['median_reduction_factor_ci'][1]}×).",
        f"- Cliff's delta: **{s['cliffs_delta']}** ({s['cliffs_delta_magnitude']}; "
        f"95% CI {s['cliffs_delta_ci'][0]}–{s['cliffs_delta_ci'][1]}).",
        f"- Cohen's d: {s['cohens_d']} ({s['cohens_d_magnitude']}); "
        f"Hedges' g: {s['hedges_g']} ({s['hedges_g_magnitude']}).",
        f"- TensorGuard reduced effort on **{s['n_tg_helped']}/{s['n_bugs']}** "
        f"bugs and increased it on {s['n_tg_hurt']} (kept, not hidden).",
        "",
        "## Per-bug",
        "",
        "| Bug | search space N | assisted | unaided | reduction | TG helped? |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in artifact["per_bug"]:
        lines.append(
            f"| {r['id']} | {r['search_space_lines']} | "
            f"{r['assisted_effort_lines']:g} | {r['unaided_effort_lines']:g} | "
            f"{r['reduction_factor']}× | {'yes' if r['tg_helped'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="regenerate in-memory and fail if the committed "
                         "localization_effort.json differs")
    args = ap.parse_args()
    artifact = run(check=args.check)
    s = artifact["summary"]
    if args.check:
        print("localization_effort.json OK (byte-identical)")
    else:
        print("Wrote", os.path.relpath(OUT_JSON, REPO_ROOT))
        print("Wrote", os.path.relpath(OUT_MD, REPO_ROOT))
    print(f"  n={s['n_bugs']}  median {s['median_assisted_lines']} vs "
          f"{s['median_unaided_lines']} lines  "
          f"reduction {s['median_reduction_factor']}x  "
          f"Cliff's d={s['cliffs_delta']} ({s['cliffs_delta_magnitude']})")


if __name__ == "__main__":
    main()
