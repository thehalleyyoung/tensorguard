"""Paired effect sizes + dual multiple-comparison correction (Step 121).

The significance harness (``evaluation/significance.py``) reports, per
TensorGuard-vs-baseline comparison, an exact McNemar p-value, a Holm-Bonferroni
family-wise correction, and a bootstrap CI on the accuracy gap. A p-value tells
you *whether* a difference is distinguishable from noise; it does not tell you
*how large* the difference is, and a single correction tells only one of the two
multiple-comparison stories. This harness closes both gaps for **every**
comparative claim, reading the same per-item correctness vectors out of
``evaluation/confusion_matrices.json`` so it stays a pure consumer (fast,
deterministic, no torch).

For each comparison it computes a battery of *paired* effect sizes appropriate
for a 2x2 McNemar table with discordant counts ``b`` (TensorGuard right, baseline
wrong) and ``c`` (TensorGuard wrong, baseline right):

* **Cohen's g** = ``b/(b+c) - 0.5`` -- the departure of the discordant split from
  an even coin, with the conventional negligible/small/medium/large bands.
* **Haldane-Anscombe odds ratio** = ``(b+0.5)/(c+0.5)`` -- a finite, continuity-
  corrected odds ratio that stays defined when ``c = 0`` (where the raw ``b/c``
  odds ratio diverges).
* **Risk difference** = ``(b-c)/n`` -- the accuracy gap as a proportion of all
  items, plus the **number-needed** ``1/|risk difference|`` (how many items you
  must evaluate to expect one net TensorGuard win).

It then applies **both** standard families of multiple-comparison correction
over the usable baselines -- Holm-Bonferroni (controls the family-wise error
rate) *and* Benjamini-Hochberg (controls the false-discovery rate) -- and reports
the adjusted p-value and reject/keep decision under each, so a reviewer can see
that the comparisons survive whichever error notion they prefer.

Closed-form and rational throughout, so the artifact is byte-identical across
machines; ``--check`` regenerates and diffs it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.statistical_rigor import (  # noqa: E402
    benjamini_hochberg,
    holm_bonferroni,
    mcnemar_from_correctness,
)

CONFUSION = REPO / "evaluation" / "confusion_matrices.json"
OUT_JSON = REPO / "reproducibility" / "effect_sizes.json"
OUT_MD = REPO / "reproducibility" / "effect_sizes.md"

REFERENCE = "tensorguard"
ALPHA = 0.05


def _r(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def cohen_g_magnitude(g: float) -> str:
    a = abs(g)
    if a < 0.05:
        return "negligible"
    if a < 0.15:
        return "small"
    if a < 0.25:
        return "medium"
    return "large"


def _correctness(confusion: Dict[str, Any]):
    methods = confusion["meta"]["methods"]
    correctness: Dict[str, List[bool]] = {m: [] for m in methods}
    na: Dict[str, int] = {m: 0 for m in methods}
    for row in confusion["per_model"]:
        label = row["label"]
        for m in methods:
            pred = row["predictions"][m]["pred"]
            if pred == "na":
                na[m] += 1
            correctness[m].append(pred == label)
    return correctness, na, methods


def measure() -> Dict[str, Any]:
    confusion = json.loads(CONFUSION.read_text())
    correctness, na, methods = _correctness(confusion)
    n_items = len(confusion["per_model"])
    ref = correctness[REFERENCE]
    baselines = [m for m in methods if m != REFERENCE]

    comps: List[Dict[str, Any]] = []
    raw_p: List[float] = []
    eligible: List[int] = []

    for m in baselines:
        mc = mcnemar_from_correctness(ref, correctness[m])
        b, c, n_disc = mc.b, mc.c, mc.n_discordant
        usable = na[m] < n_items

        cohen_g = (b / n_disc - 0.5) if n_disc > 0 else 0.0
        haldane_or = (b + 0.5) / (c + 0.5)
        risk_diff = (b - c) / n_items
        number_needed = (1.0 / abs(risk_diff)) if risk_diff != 0 else None

        comp = {
            "baseline": m,
            "usable": usable,
            "na_items": na[m],
            "b_tg_right_base_wrong": b,
            "c_tg_wrong_base_right": c,
            "n_discordant": n_disc,
            "mcnemar_p_value": _r(mc.p_value),
            "effect_sizes": {
                "cohen_g": _r(cohen_g),
                "cohen_g_magnitude": cohen_g_magnitude(cohen_g),
                "raw_odds_ratio": _r(mc.odds_ratio),
                "haldane_anscombe_odds_ratio": _r(haldane_or),
                "risk_difference": _r(risk_diff),
                "number_needed_to_evaluate": _r(number_needed, 4),
            },
        }
        if usable:
            eligible.append(len(comps))
            raw_p.append(mc.p_value)
        comps.append(comp)

    # Dual multiple-comparison correction over the usable family.
    holm = holm_bonferroni(raw_p, alpha=ALPHA)
    bh = benjamini_hochberg(raw_p, alpha=ALPHA)
    for slot, idx in enumerate(eligible):
        comps[idx]["corrections"] = {
            "holm_bonferroni_adjusted_p": _r(holm.adjusted_p_values[slot]),
            "holm_bonferroni_reject": bool(holm.rejected[slot]),
            "benjamini_hochberg_adjusted_p": _r(bh.adjusted_p_values[slot]),
            "benjamini_hochberg_reject": bool(bh.rejected[slot]),
        }
    for idx, comp in enumerate(comps):
        if "corrections" not in comp:
            comp["corrections"] = {
                "holm_bonferroni_adjusted_p": None,
                "holm_bonferroni_reject": False,
                "benjamini_hochberg_adjusted_p": None,
                "benjamini_hochberg_reject": False,
            }

    n_usable = len(eligible)
    return {
        "step": 121,
        "reference_method": REFERENCE,
        "reads": "evaluation/confusion_matrices.json",
        "alpha": ALPHA,
        "n_items": n_items,
        "n_comparisons": len(comps),
        "n_usable_comparisons": n_usable,
        "comparisons": comps,
        "family_size": n_usable,
        "n_holm_significant": holm.n_rejected,
        "n_bh_significant": bh.n_rejected,
        "corrections_agree": holm.n_rejected == bh.n_rejected,
        "every_comparison_has_effect_size": all(
            "effect_sizes" in c for c in comps),
    }


def render_markdown(d: Dict[str, Any]) -> str:
    lines = [
        "# Paired effect sizes & dual multiple-comparison correction (Step 121)",
        "",
        f"Consumer of `evaluation/confusion_matrices.json` (n = {d['n_items']} "
        f"items, family of {d['n_usable_comparisons']} usable baselines). "
        "Every comparison carries a paired effect size and is corrected under "
        "both a family-wise (Holm-Bonferroni) and a false-discovery "
        "(Benjamini-Hochberg) procedure.",
        "",
        "| baseline | b | c | Cohen's g | magnitude | Haldane OR | risk diff "
        "| McNemar p | Holm p | Holm? | BH p | BH? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in d["comparisons"]:
        es = c["effect_sizes"]
        cr = c["corrections"]
        lines.append(
            f"| {c['baseline']} | {c['b_tg_right_base_wrong']} "
            f"| {c['c_tg_wrong_base_right']} | {es['cohen_g']} "
            f"| {es['cohen_g_magnitude']} | {es['haldane_anscombe_odds_ratio']} "
            f"| {es['risk_difference']} | {c['mcnemar_p_value']} "
            f"| {cr['holm_bonferroni_adjusted_p']} "
            f"| {cr['holm_bonferroni_reject']} "
            f"| {cr['benjamini_hochberg_adjusted_p']} "
            f"| {cr['benjamini_hochberg_reject']} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- comparisons carrying a paired effect size: "
        f"**{d['every_comparison_has_effect_size']}** (all)",
        f"- significant after Holm-Bonferroni (FWER): **{d['n_holm_significant']}**",
        f"- significant after Benjamini-Hochberg (FDR): **{d['n_bh_significant']}**",
        f"- the two corrections agree on the count: **{d['corrections_agree']}**",
        "",
    ]
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if ok:
            print("effect_sizes: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
