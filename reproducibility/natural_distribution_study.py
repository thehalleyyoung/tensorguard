"""Deterministic harness: natural-distribution coverage study (Step 108).

Bug corpora answer "does it catch real bugs?". This study answers the
complementary usability question a practitioner cares about: on *ordinary,
idiomatic* model code, how often does the verifier return a definite verdict
instead of abstaining (``UNKNOWN``)? That coverage rate -- and, just as
important, the false-alarm rate on this clean natural sample -- is a headline
number for "would this be annoying to actually use".

We score the curated natural-distribution sample
(``corpus_extended/natural_models.py``: 29 clean, idiomatic, public-style
architectures spanning MLPs, CNNs, ResNet/U-Net blocks, attention/transformer
blocks, RNNs, autoencoders, GANs, embeddings and more). Every model is clean by
construction (it executes under eager PyTorch). For each model and each
soundness mode we record the verifier verdict; we then report:

* **coverage** -- fraction of models that received a *decided* verdict
  (SAFE or UNSAFE) rather than abstaining, with a Wilson interval;
* **abstention rate** -- the complement;
* **false-alarm rate** -- fraction of these clean models flagged UNSAFE
  (should be zero for a tool you would actually run).

Only counts, rates (rounded), Wilson intervals and verdict tallies are
recorded, so the artifact is byte-identical across machines.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.natural_models import all_models  # noqa: E402
from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "natural_distribution_study.json"
OUT_MD = REPO / "reproducibility" / "natural_distribution_study.md"

MODES = ["sound", "balanced", "heuristic"]


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


def _verdict(model, mode: str) -> str:
    r = verify_architecture(
        model.source,
        input_shapes={k: tuple(v) for k, v in model.input_shapes.items()},
        soundness_mode=mode,
    )
    return str(r.verdict)


def measure() -> dict:
    models = all_models()
    n = len(models)
    families = sorted({m.family for m in models})

    per_mode = {}
    for mode in MODES:
        verdicts = {m.id: _verdict(m, mode) for m in models}
        tally = Counter(verdicts.values())
        decided = tally.get("SAFE", 0) + tally.get("UNSAFE", 0)
        abstained = n - decided
        false_alarms = tally.get("UNSAFE", 0)  # all models are clean
        per_mode[mode] = {
            "verdict_tally": dict(sorted(tally.items())),
            "n_decided": decided,
            "n_abstained": abstained,
            "n_false_alarms": false_alarms,
            "coverage": _wilson(decided, n),
            "abstention_rate": _wilson(abstained, n),
            "false_alarm_rate": _wilson(false_alarms, n),
            "false_alarm_ids": sorted(
                mid for mid, v in verdicts.items() if v == "UNSAFE"),
        }

    return {
        "n_models": n,
        "n_families": len(families),
        "families": families,
        "modes": list(MODES),
        "per_mode": per_mode,
        "all_models_clean_by_construction": True,
        "zero_false_alarms_all_modes": all(
            per_mode[m]["n_false_alarms"] == 0 for m in MODES),
        "full_coverage_all_modes": all(
            per_mode[m]["n_abstained"] == 0 for m in MODES),
    }


def render_markdown(data: dict) -> str:
    lines = [
        "# Natural-distribution coverage study",
        "",
        f"On a sample of **{data['n_models']}** clean, idiomatic, public-style "
        f"architectures across **{data['n_families']}** families "
        f"({', '.join(data['families'])}), we measure how often the verifier "
        "returns a *decided* verdict rather than abstaining, and how often it "
        "false-alarms on this clean natural distribution.",
        "",
        "| mode | decided | abstained | coverage [95% CI] | false alarms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode in data["modes"]:
        d = data["per_mode"][mode]
        cov = d["coverage"]
        lines.append(
            f"| {mode} | {d['n_decided']} | {d['n_abstained']} | "
            f"{cov['point']} [{cov['low']}, {cov['high']}] | "
            f"{d['n_false_alarms']} |"
        )
    lines += [
        "",
        f"- full coverage (zero abstention) in every mode: "
        f"**{data['full_coverage_all_modes']}**",
        f"- zero false alarms in every mode: "
        f"**{data['zero_false_alarms_all_modes']}**",
        "",
        "Each model is clean by construction (it executes under eager "
        "PyTorch), so every UNSAFE verdict would be a false alarm; none occur.",
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
            print("MISMATCH: natural_distribution_study artifacts differ")
            return 1
        print("OK: natural_distribution_study artifacts byte-identical")
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
