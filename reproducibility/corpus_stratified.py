"""Deterministic harness: stratified per-class metrics with Wilson CIs (Step 104).

The aggregate score (`corpus_extended_score.py`) can hide weak spots: a tool can
look great on average while missing an entire bug class. This harness stratifies
the extended corpus **by class** and reports, for the `sound` and `balanced`
modes:

* per **bug class** (the buggy families): recall with a Wilson 95% CI;
* per **clean class** (the clean families): specificity (true-negative rate)
  with a Wilson 95% CI;
* **macro-averaged** recall and specificity (unweighted mean across classes, so
  small classes count as much as large ones);
* the **worst-class** recall and its lower confidence bound -- the honest
  headline a reviewer should look at, since a sound tool must not collapse on any
  single class.

Every buggy/clean case is scored (no cherry-picking). Only counts, verdict
strings and rounded rates/CIs are recorded, so the artifact is byte-identical
across machines.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.generators import all_cases  # noqa: E402
from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "corpus_stratified.json"
OUT_MD = REPO / "reproducibility" / "corpus_stratified.md"


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"point": None, "low": None, "high": None, "k": k, "n": n}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return {
        "point": round(p, 4),
        "low": round(max(0.0, center - half), 4),
        "high": round(min(1.0, center + half), 4),
        "k": k,
        "n": n,
    }


def _verdict(case, mode: str) -> str:
    res = verify_architecture(
        case.source,
        input_shapes={k: tuple(v) for k, v in case.input_shapes.items()},
        soundness_mode=mode,
    )
    return str(res.verdict)


def _stratify(cases, mode: str) -> dict:
    # Per-family tallies. For buggy families we count "caught" (UNSAFE);
    # for clean families we count "true negative" (SAFE), with abstains excluded
    # from the decided denominator.
    buggy = {}
    clean = {}
    for case in cases:
        v = _verdict(case, mode)
        if case.label == "buggy":
            d = buggy.setdefault(case.family, {"caught": 0, "decided": 0, "total": 0})
            d["total"] += 1
            if v == "UNSAFE":
                d["caught"] += 1
                d["decided"] += 1
            elif v == "SAFE":
                d["decided"] += 1
            # UNKNOWN -> abstain, excluded from decided
        else:
            d = clean.setdefault(case.family, {"tn": 0, "decided": 0, "total": 0})
            d["total"] += 1
            if v == "SAFE":
                d["tn"] += 1
                d["decided"] += 1
            elif v == "UNSAFE":
                d["decided"] += 1

    per_class_recall = {
        fam: {
            "total": d["total"],
            "caught": d["caught"],
            "decided": d["decided"],
            "recall": _wilson(d["caught"], d["decided"]),
        }
        for fam, d in sorted(buggy.items())
    }
    per_class_specificity = {
        fam: {
            "total": d["total"],
            "true_negative": d["tn"],
            "decided": d["decided"],
            "specificity": _wilson(d["tn"], d["decided"]),
        }
        for fam, d in sorted(clean.items())
    }

    recall_points = [v["recall"]["point"] for v in per_class_recall.values()
                     if v["recall"]["point"] is not None]
    spec_points = [v["specificity"]["point"]
                   for v in per_class_specificity.values()
                   if v["specificity"]["point"] is not None]
    macro_recall = round(sum(recall_points) / len(recall_points), 4) if recall_points else None
    macro_spec = round(sum(spec_points) / len(spec_points), 4) if spec_points else None

    # Worst buggy class by recall lower bound.
    worst_fam = None
    worst = None
    for fam, v in per_class_recall.items():
        lo = v["recall"]["low"]
        if lo is None:
            continue
        if worst is None or lo < worst:
            worst = lo
            worst_fam = fam

    return {
        "mode": mode,
        "n_buggy_classes": len(per_class_recall),
        "n_clean_classes": len(per_class_specificity),
        "per_class_recall": per_class_recall,
        "per_class_specificity": per_class_specificity,
        "macro_recall": macro_recall,
        "macro_specificity": macro_spec,
        "worst_class_recall": {
            "family": worst_fam,
            "recall_low": worst,
            "recall": per_class_recall.get(worst_fam, {}).get("recall")
            if worst_fam else None,
        },
        # Every buggy class fully caught (point recall == 1.0 on decided)?
        "every_buggy_class_fully_caught": all(
            v["recall"]["point"] == 1.0 for v in per_class_recall.values()
            if v["recall"]["point"] is not None
        ),
        # Every clean class free of false positives?
        "every_clean_class_no_false_positive": all(
            v["specificity"]["point"] == 1.0
            for v in per_class_specificity.values()
            if v["specificity"]["point"] is not None
        ),
    }


def measure() -> dict:
    cases = all_cases()
    return {
        "n_total": len(cases),
        "balanced": _stratify(cases, "balanced"),
        "sound": _stratify(cases, "sound"),
    }


def _ci(c) -> str:
    if c is None or c.get("point") is None:
        return "n/a"
    return f"{c['point']:.4f} [{c['low']:.4f}, {c['high']:.4f}] (n={c['n']})"


def render_markdown(data: dict) -> str:
    lines = [
        "# Stratified per-class metrics (Wilson 95% CIs)",
        "",
        f"Every one of the **{data['n_total']}** cases is scored and stratified by "
        "class, so no weak bug class can hide behind a strong average. Macro "
        "averages weight each class equally.",
        "",
    ]
    for mode in ("balanced", "sound"):
        m = data[mode]
        lines += [
            f"## `{mode}` mode",
            "",
            f"- macro recall: **{m['macro_recall']}**, macro specificity: "
            f"**{m['macro_specificity']}**",
            f"- worst buggy class by recall lower bound: "
            f"`{m['worst_class_recall']['family']}` "
            f"({_ci(m['worst_class_recall']['recall'])})",
            f"- every buggy class fully caught: "
            f"{m['every_buggy_class_fully_caught']}; every clean class "
            f"false-positive-free: {m['every_clean_class_no_false_positive']}",
            "",
            "Per buggy class (recall on decided cases):",
            "",
            "| bug class | caught | decided | total | recall (95% CI) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for fam, v in m["per_class_recall"].items():
            lines.append(
                f"| {fam} | {v['caught']} | {v['decided']} | {v['total']} | "
                f"{_ci(v['recall'])} |"
            )
        lines += [
            "",
            "Per clean class (specificity on decided cases):",
            "",
            "| clean class | true neg | decided | total | specificity (95% CI) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for fam, v in m["per_class_specificity"].items():
            lines.append(
                f"| {fam} | {v['true_negative']} | {v['decided']} | {v['total']} | "
                f"{_ci(v['specificity'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        old_json = OUT_JSON.read_text() if OUT_JSON.exists() else ""
        old_md = OUT_MD.read_text() if OUT_MD.exists() else ""
        if old_json != new_json or old_md != new_md:
            print("MISMATCH: corpus_stratified artifacts differ")
            return 1
        print("OK: corpus_stratified artifacts byte-identical")
        return 0
    OUT_JSON.write_text(new_json)
    OUT_MD.write_text(new_md)
    print(f"Wrote {OUT_JSON.name} and {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(run(check=args.check))
