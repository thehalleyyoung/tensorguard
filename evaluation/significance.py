#!/usr/bin/env python3
"""Significance testing for the TensorGuard precision/recall study (Step 88).

The headline confusion matrices in ``evaluation/confusion_matrices.json`` give
point estimates of accuracy / precision / recall for each detector on the frozen
``real_benchmarks`` corpus.  Point estimates alone do not tell you whether
TensorGuard's advantage over a baseline is *statistically* distinguishable from
noise on this corpus.  This harness answers that question **rigorously** using
the same per-item predictions, with no new ground truth:

* **Paired McNemar exact test** (``src.statistical_rigor.mcnemar_exact_test``)
  for each ``tensorguard`` vs. baseline pair.  McNemar is the correct test for
  two classifiers evaluated on the *same* items: it conditions on the discordant
  pairs (one method right, the other wrong) and tests whether the split is
  fairer than a coin flip.
* **Holm–Bonferroni** family-wise correction
  (``src.statistical_rigor.holm_bonferroni``) over the family of pairwise
  comparisons, so running several baselines does not inflate the false-positive
  rate.
* **Paired percentile bootstrap CI** for the accuracy *difference*
  (``src.statistical_rigor.paired_bootstrap_accuracy_diff``), resampling items
  jointly to preserve the pairing.

It is deliberately a *consumer* of ``confusion_matrices.json`` (the upstream
artifact), so it is fast, deterministic and re-runs without torch/pytea.  Like
the other evaluation artifacts it supports ``--check`` (regenerate and diff).

Usage::

    cd tensorguard && PYTHONPATH=. python3 evaluation/significance.py
    cd tensorguard && PYTHONPATH=. python3 evaluation/significance.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.statistical_rigor import (  # noqa: E402
    holm_bonferroni,
    mcnemar_from_correctness,
    paired_bootstrap_accuracy_diff,
)

CONFUSION_JSON = os.path.join(THIS_DIR, "confusion_matrices.json")
OUT_JSON = os.path.join(THIS_DIR, "significance.json")
OUT_MD = os.path.join(THIS_DIR, "significance.md")

REFERENCE = "tensorguard"
ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 0


def _round(x, nd=6):
    return None if x is None else round(float(x), nd)


def _correctness_vectors(confusion: Dict[str, Any]):
    """Per-item correctness (pred == label) for every method, plus NA coverage.

    NA (e.g. an unavailable baseline, or a detector that abstains) is counted as
    *wrong*, mirroring the headline ``na_policy`` of the confusion artifact, but
    the NA fraction is reported alongside so a fully-unavailable baseline can be
    excluded from the family.
    """
    per_model: List[Dict[str, Any]] = confusion["per_model"]
    methods: List[str] = confusion["meta"]["methods"]
    correctness: Dict[str, List[bool]] = {m: [] for m in methods}
    na_counts: Dict[str, int] = {m: 0 for m in methods}
    item_ids = [row["id"] for row in per_model]
    for row in per_model:
        label = row["label"]
        for m in methods:
            pred = row["predictions"][m]["pred"]
            if pred == "na":
                na_counts[m] += 1
            correctness[m].append(pred == label)
    return correctness, na_counts, item_ids


def run(check: bool = False) -> Dict[str, Any]:
    with open(CONFUSION_JSON, "r", encoding="utf-8") as fh:
        confusion = json.load(fh)

    correctness, na_counts, item_ids = _correctness_vectors(confusion)
    methods = confusion["meta"]["methods"]
    n_items = len(item_ids)

    if REFERENCE not in correctness:
        raise SystemExit("reference method %r not in artifact" % REFERENCE)
    ref_correct = correctness[REFERENCE]

    baselines = [m for m in methods if m != REFERENCE]

    # First pass: McNemar per baseline (skip baselines that are entirely NA).
    comparisons: List[Dict[str, Any]] = []
    raw_p: List[float] = []
    eligible: List[int] = []  # indices into comparisons that get FWER-corrected
    for m in baselines:
        na = na_counts[m]
        usable = na < n_items
        mc = mcnemar_from_correctness(ref_correct, correctness[m])
        boot = paired_bootstrap_accuracy_diff(
            ref_correct, correctness[m],
            n_resamples=BOOTSTRAP_RESAMPLES, confidence=1.0 - ALPHA,
            seed=BOOTSTRAP_SEED,
        )
        comp = {
            "baseline": m,
            "na_items": na,
            "usable": usable,
            "ref_accuracy": _round(sum(ref_correct) / n_items),
            "baseline_accuracy": _round(sum(correctness[m]) / n_items),
            "mcnemar": {
                "b_ref_right_base_wrong": mc.b,
                "c_ref_wrong_base_right": mc.c,
                "n_discordant": mc.n_discordant,
                "statistic": mc.statistic,
                "p_value": _round(mc.p_value),
                "odds_ratio": _round(mc.odds_ratio),
            },
            "accuracy_diff_bootstrap": {
                "point_estimate": _round(boot.point_estimate),
                "ci_low": _round(boot.ci_low),
                "ci_high": _round(boot.ci_high),
                "confidence": boot.confidence,
                "n_resamples": boot.n_resamples,
                "fraction_above_zero": _round(boot.fraction_above_zero),
            },
        }
        if usable:
            eligible.append(len(comparisons))
            raw_p.append(mc.p_value)
        comparisons.append(comp)

    # Holm-Bonferroni over the eligible (usable) family.
    holm = holm_bonferroni(raw_p, alpha=ALPHA)
    for slot, comp_idx in enumerate(eligible):
        comparisons[comp_idx]["mcnemar"]["holm_adjusted_p"] = _round(
            holm.adjusted_p_values[slot]
        )
        comparisons[comp_idx]["mcnemar"]["significant_at_alpha"] = bool(
            holm.rejected[slot]
        )

    artifact = {
        "meta": {
            "generated_by": "evaluation/significance.py",
            "command": "python3 evaluation/significance.py",
            "reads": "evaluation/confusion_matrices.json",
            "reference_method": REFERENCE,
            "alpha": ALPHA,
            "n_items": n_items,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "corpus": confusion["meta"].get("corpus"),
            "corpus_version": confusion["meta"].get("corpus_version"),
            "tests": {
                "paired_significance": "McNemar exact (binomial), two-sided",
                "fwer_correction": "Holm-Bonferroni over usable baselines",
                "interval": "paired percentile bootstrap of accuracy difference",
            },
            "na_policy": (
                "NA counts as wrong (mirrors confusion_matrices.json); a "
                "baseline that is NA on every item is excluded from the "
                "FWER-corrected family but still reported."
            ),
            "interpretation": (
                "A small Holm-adjusted p-value means TensorGuard and the "
                "baseline disagree on the discordant items more lopsidedly than "
                "a coin flip would explain on this corpus; a bootstrap CI that "
                "excludes zero means the accuracy gap is unlikely to be noise. "
                "Both are honest about the small (n=%d) corpus." % n_items
            ),
        },
        "comparisons": comparisons,
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            current = fh.read()
        if current != text:
            raise SystemExit("significance.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(artifact: Dict[str, Any]) -> str:
    meta = artifact["meta"]
    lines = [
        "# Significance tests — TensorGuard vs. baselines",
        "",
        "_Generated by `evaluation/significance.py` from "
        "`evaluation/confusion_matrices.json`. Do not edit by hand._",
        "",
        f"- Reference: **{meta['reference_method']}**",
        f"- Corpus: {meta['corpus']} (n = {meta['n_items']} models)",
        f"- Paired test: {meta['tests']['paired_significance']}",
        f"- FWER correction: {meta['tests']['fwer_correction']} "
        f"(alpha = {meta['alpha']})",
        f"- Interval: {meta['tests']['interval']} "
        f"({meta['bootstrap_resamples']} resamples)",
        "",
        "| Baseline | acc(TG) | acc(base) | b | c | McNemar p | Holm p | "
        "sig | acc-diff (CI) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in artifact["comparisons"]:
        mc = c["mcnemar"]
        bs = c["accuracy_diff_bootstrap"]
        holm = mc.get("holm_adjusted_p")
        sig = mc.get("significant_at_alpha")
        sig_s = "—" if sig is None else ("yes" if sig else "no")
        holm_s = "n/a" if holm is None else f"{holm:.4f}"
        ci = f"{bs['point_estimate']:+.3f} [{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}]"
        lines.append(
            f"| {c['baseline']} | {c['ref_accuracy']:.3f} | "
            f"{c['baseline_accuracy']:.3f} | {mc['b_ref_right_base_wrong']} | "
            f"{mc['c_ref_wrong_base_right']} | {mc['p_value']:.4f} | {holm_s} | "
            f"{sig_s} | {ci} |"
        )
    lines += [
        "",
        "**b** = items where TensorGuard is right and the baseline is wrong; "
        "**c** = the reverse.",
        "",
        meta["interpretation"],
        "",
        meta["na_policy"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed artifact is up to date")
    args = ap.parse_args()
    artifact = run(check=args.check)
    if args.check:
        print("significance.json is up to date")
    else:
        sig = [c["baseline"] for c in artifact["comparisons"]
               if c["mcnemar"].get("significant_at_alpha")]
        print("wrote", OUT_JSON)
        print("significant (Holm, alpha=%.2f) vs %s: %s"
              % (ALPHA, REFERENCE, ", ".join(sig) or "none"))


if __name__ == "__main__":
    main()
