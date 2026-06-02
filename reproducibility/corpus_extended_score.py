"""Deterministic harness: score TensorGuard over the *extended* corpus.

Loads the frozen, runtime-validated extended benchmark corpus
(:mod:`corpus_extended`, 227 content-addressed cases spanning nine shape-error
families) and runs the real :func:`src.api.verify_architecture` over **every**
case in both ``balanced`` and ``sound`` soundness modes. Because each case's
label is ground truth (executably validated against real PyTorch at build
time), this yields an honest confusion matrix from which we report:

* recall (sensitivity) on buggy cases,
* specificity / false-positive rate on clean cases,
* precision,
* per-family recall,
* abstention rates (cases the sound mode declines to judge),

each with a **Wilson score 95% confidence interval** so the numbers are
reported with their statistical uncertainty rather than as bare point
estimates. We do *not* cherry-pick: every runtime-validated case is scored and
counted, including the ones TensorGuard gets wrong.

Only integer counts, verdict strings and rounded (4-dp) rate/CI values are
recorded, so the artifact is byte-identical across machines.
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

OUT_JSON = REPO / "reproducibility" / "corpus_extended_score.json"
OUT_MD = REPO / "reproducibility" / "corpus_extended_score.md"


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
    """Wilson score interval for a binomial proportion k/n (95% by default)."""
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


def _verdict_for(case, mode: str) -> str:
    res = verify_architecture(
        case.source,
        input_shapes={k: tuple(v) for k, v in case.input_shapes.items()},
        soundness_mode=mode,
    )
    # Plain string verdict: "SAFE" | "UNSAFE" | "UNKNOWN".
    return str(res.verdict)


def _score_mode(cases, mode: str) -> dict:
    # Confusion bookkeeping. "positive" = predicts a bug (UNSAFE).
    tp = fp = tn = fn = 0
    abstain_buggy = abstain_clean = 0
    per_family_recall: dict = {}

    for case in cases:
        verdict = _verdict_for(case, mode)
        is_buggy = case.label == "buggy"
        if is_buggy:
            fam = per_family_recall.setdefault(
                case.family, {"caught": 0, "total": 0, "abstained": 0}
            )
            fam["total"] += 1
            if verdict == "UNSAFE":
                tp += 1
                fam["caught"] += 1
            elif verdict == "SAFE":
                fn += 1
            else:  # UNKNOWN / abstain
                abstain_buggy += 1
                fam["abstained"] += 1
        else:  # clean
            if verdict == "UNSAFE":
                fp += 1
            elif verdict == "SAFE":
                tn += 1
            else:
                abstain_clean += 1

    n_buggy = sum(1 for c in cases if c.label == "buggy")
    n_clean = sum(1 for c in cases if c.label == "clean")
    decided_buggy = tp + fn
    decided_clean = tn + fp
    pred_pos = tp + fp

    fam_out = {}
    for fam in sorted(per_family_recall):
        d = per_family_recall[fam]
        decided = d["caught"] + (d["total"] - d["caught"] - d["abstained"])
        fam_out[fam] = {
            "total": d["total"],
            "caught": d["caught"],
            "abstained": d["abstained"],
            "recall_decided": _wilson(d["caught"], decided),
        }

    return {
        "mode": mode,
        "n_total": len(cases),
        "n_buggy": n_buggy,
        "n_clean": n_clean,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "abstained_buggy": abstain_buggy,
        "abstained_clean": abstain_clean,
        # Recall over the buggy cases on which a definite verdict was issued.
        "recall_on_decided": _wilson(tp, decided_buggy),
        # Recall over *all* buggy cases (abstentions count against recall).
        "recall_on_all_buggy": _wilson(tp, n_buggy),
        "specificity_on_decided": _wilson(tn, decided_clean),
        "false_positive_rate_on_decided": _wilson(fp, decided_clean),
        "precision": _wilson(tp, pred_pos),
        "per_family_recall": fam_out,
        # The headline soundness invariant for the sound mode: zero false
        # positives on clean code (no clean module is ever called buggy).
        "no_false_positive": fp == 0,
    }


def measure() -> dict:
    cases = all_cases()
    balanced = _score_mode(cases, "balanced")
    sound = _score_mode(cases, "sound")
    return {
        "corpus": {
            "total": len(cases),
            "buggy": balanced["n_buggy"],
            "clean": balanced["n_clean"],
            "families": sorted({c.family for c in cases}),
            "ground_truth": (
                "Every case runtime-validated against real PyTorch at build "
                "time (buggy raises with expected substring; clean runs clean)."
            ),
        },
        "balanced": balanced,
        "sound": sound,
        "sound_mode_has_no_false_positive": sound["no_false_positive"],
    }


def _fmt_ci(ci: dict) -> str:
    if ci["point"] is None:
        return "n/a"
    return f"{ci['point']:.4f} [{ci['low']:.4f}, {ci['high']:.4f}] (n={ci['n']})"


def render_markdown(data: dict) -> str:
    c = data["corpus"]
    lines = [
        "# TensorGuard on the extended benchmark corpus",
        "",
        f"Scored over **{c['total']}** content-addressed cases "
        f"(**{c['buggy']}** buggy / **{c['clean']}** clean) across "
        f"**{len(c['families'])}** shape-error families. Every case is "
        "runtime-validated against real PyTorch at build time, so the labels "
        "are ground truth and nothing is cherry-picked. Rates are reported "
        "with **Wilson score 95 percent confidence intervals**.",
        "",
    ]
    for mode in ("balanced", "sound"):
        m = data[mode]
        conf = m["confusion"]
        lines += [
            f"## `{mode}` mode",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| confusion (tp / fp / tn / fn) | "
            f"{conf['tp']} / {conf['fp']} / {conf['tn']} / {conf['fn']} |",
            f"| abstained (buggy / clean) | "
            f"{m['abstained_buggy']} / {m['abstained_clean']} |",
            f"| recall on decided | {_fmt_ci(m['recall_on_decided'])} |",
            f"| recall on all buggy | {_fmt_ci(m['recall_on_all_buggy'])} |",
            f"| specificity on decided | {_fmt_ci(m['specificity_on_decided'])} |",
            f"| false-positive rate on decided | "
            f"{_fmt_ci(m['false_positive_rate_on_decided'])} |",
            f"| precision | {_fmt_ci(m['precision'])} |",
            f"| no false positive on clean | {m['no_false_positive']} |",
            "",
            "Per-family recall (on decided buggy cases):",
            "",
            "| family | caught | total | abstained | recall (95% CI) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for fam in sorted(m["per_family_recall"]):
            d = m["per_family_recall"][fam]
            lines.append(
                f"| {fam} | {d['caught']} | {d['total']} | {d['abstained']} | "
                f"{_fmt_ci(d['recall_decided'])} |"
            )
        lines.append("")
    lines += [
        f"**Sound mode has zero false positives on clean code: "
        f"{data['sound_mode_has_no_false_positive']}.** This is the core "
        "soundness promise: no clean module is ever flagged as buggy.",
        "",
    ]
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        old_json = OUT_JSON.read_text() if OUT_JSON.exists() else ""
        old_md = OUT_MD.read_text() if OUT_MD.exists() else ""
        if old_json != new_json or old_md != new_md:
            print("MISMATCH: corpus_extended_score artifacts differ")
            return 1
        print("OK: corpus_extended_score artifacts byte-identical")
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
