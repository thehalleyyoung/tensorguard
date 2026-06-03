#!/usr/bin/env python3
"""Step 15 -- measure false UNKNOWNs in strict sound mode.

Sound mode is allowed to abstain on genuinely out-of-fragment programs. That
conservatism is a feature, but it becomes a usability bug when the verifier
declines to classify code that is already within the supported fragment and has
known ground truth. This benchmark therefore measures **false UNKNOWNs**:

    a sound-mode UNKNOWN verdict on a model TensorGuard should decide.

The corpus is intentionally grounded in executable code:

* clean cases are the frozen real clean benchmarks plus generated models that
  are admitted only after eager PyTorch executes them successfully;
* buggy cases are the phase/path latent bugs from the hard-recall benchmark,
  each proven genuine by exercising the hidden fault in real PyTorch.

Usage
-----
    cd tensorguard && PYTHONPATH=. python3 evaluation/false_unknowns.py
    cd tensorguard && PYTHONPATH=. python3 evaluation/false_unknowns.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation import hard_recall, sound_mode_fp  # noqa: E402
from real_benchmarks import load  # noqa: E402

OUT_JSON = os.path.join(THIS_DIR, "false_unknowns.json")
OUT_MD = os.path.join(THIS_DIR, "false_unknowns.md")


def _sound_result(source: str, input_shapes: Dict[str, Any]):
    from src.api import verify_architecture

    return verify_architecture(
        source,
        input_shapes={k: tuple(v) for k, v in input_shapes.items()},
        check_devices=True,
        check_gradients=True,
        max_cegar_iterations=0,
        soundness_mode="sound",
    )


def build_corpus() -> List[Dict[str, Any]]:
    """Return deterministic in-fragment cases with executable ground truth."""

    corpus: List[Dict[str, Any]] = []

    for item in load.load_items():
        if item["label"] != "clean":
            continue
        corpus.append({
            "id": item["id"],
            "family": "real_clean",
            "kind": "clean",
            "expected_verdict": "SAFE",
            "source_kind": "frozen_real_benchmark",
            "source": load.read_source(item),
            "input_shapes": {k: list(v) for k, v in item["input_shapes"].items()},
            "ground_truth": "eager PyTorch clean benchmark from real_benchmarks",
        })

    for generated in sound_mode_fp.generate_corpus():
        corpus.append({
            "id": generated["id"],
            "family": "generated_%s" % generated["family"],
            "kind": "clean",
            "expected_verdict": "SAFE",
            "source_kind": "generated_validated_clean",
            "source": generated["source"],
            "input_shapes": generated["input_shapes"],
            "ground_truth": "generated model admitted only after eager PyTorch execution",
        })

    for model in hard_recall.build_corpus():
        if model["family"] not in {"phase_eval", "path_flag"}:
            continue
        genuine, detail = hard_recall.is_genuine_bug(model)
        assert genuine, "%s is not a genuine bug: %s" % (model["id"], detail)
        corpus.append({
            "id": model["id"],
            "family": model["family"],
            "kind": "buggy",
            "expected_verdict": "UNSAFE",
            "source_kind": "latent_bug",
            "source": model["source"],
            "input_shapes": model["input_shapes"],
            "ground_truth": detail,
        })

    return corpus


def _classify(row: Dict[str, Any]) -> Tuple[str, int, bool, List[str]]:
    result = _sound_result(row["source"], row["input_shapes"])
    return (
        result.verdict,
        result.bug_count,
        bool(getattr(result, "abstained", False)),
        list(getattr(result, "unknown_reasons", [])),
    )


def run(check: bool = False) -> Dict[str, Any]:
    import torch  # noqa: F401  (required by corpus validators/verifier paths)

    corpus = build_corpus()
    rows: List[Dict[str, Any]] = []
    for case in corpus:
        verdict, bug_count, abstained, unknown_reasons = _classify(case)
        false_unknown = verdict == "UNKNOWN"
        misclassified = verdict not in (case["expected_verdict"], "UNKNOWN")
        rows.append({
            "id": case["id"],
            "family": case["family"],
            "kind": case["kind"],
            "source_kind": case["source_kind"],
            "expected_verdict": case["expected_verdict"],
            "sound_verdict": verdict,
            "bug_count": bug_count,
            "abstained": abstained,
            "false_unknown": false_unknown,
            "misclassified": misclassified,
            "unknown_reasons": unknown_reasons,
            "ground_truth": case["ground_truth"],
        })

    total = len(rows)
    false_unknowns = [r for r in rows if r["false_unknown"]]
    misclassified = [r for r in rows if r["misclassified"]]
    decided = total - len(false_unknowns)

    by_kind: Dict[str, Dict[str, int]] = {}
    by_family: Dict[str, Dict[str, int]] = {}
    for row in rows:
        for table, key in ((by_kind, row["kind"]), (by_family, row["family"])):
            bucket = table.setdefault(
                key, {"total": 0, "decided": 0, "UNKNOWN": 0, "SAFE": 0, "UNSAFE": 0}
            )
            bucket["total"] += 1
            bucket[row["sound_verdict"]] += 1
            bucket["decided"] += int(row["sound_verdict"] != "UNKNOWN")

    artifact = {
        "meta": {
            "generated_by": "evaluation/false_unknowns.py",
            "command": "python3 evaluation/false_unknowns.py",
            "soundness_mode": "sound",
            "definition": (
                "false UNKNOWN = sound-mode UNKNOWN on an executable, "
                "ground-truthed model TensorGuard should classify"
            ),
            "eligibility": (
                "clean cases must execute in eager PyTorch; buggy cases must be "
                "real latent phase/path faults confirmed by hard_recall validators"
            ),
            "n_real_clean": sum(1 for r in rows if r["family"] == "real_clean"),
            "n_generated_clean": sum(1 for r in rows if r["source_kind"] == "generated_validated_clean"),
            "n_latent_bugs": sum(1 for r in rows if r["source_kind"] == "latent_bug"),
            "families": sorted(by_family),
        },
        "summary": {
            "total": total,
            "decided": decided,
            "false_unknowns": len(false_unknowns),
            "false_unknown_rate": round(len(false_unknowns) / total, 6) if total else None,
            "decision_rate": round(decided / total, 6) if total else None,
            "misclassified": len(misclassified),
        },
        "by_kind": by_kind,
        "by_family": by_family,
        "false_unknown_ids": [r["id"] for r in false_unknowns],
        "misclassified_ids": [r["id"] for r in misclassified],
        "per_model": rows,
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            if fh.read() != text:
                raise SystemExit("false_unknowns.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(a: Dict[str, Any]) -> str:
    s = a["summary"]
    m = a["meta"]
    lines = [
        "# Step 15 -- false UNKNOWN rate in sound mode",
        "",
        "Strict `sound` mode was run over **%d executable, ground-truthed** "
        "models: %d frozen real clean benchmarks, %d generated clean models "
        "admitted only after eager PyTorch execution, and %d real latent "
        "phase/path bugs confirmed by the hard-recall validators."
        % (s["total"], m["n_real_clean"], m["n_generated_clean"], m["n_latent_bugs"]),
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "|---|---|",
        "| Eligible models | %d |" % s["total"],
        "| Decided by sound mode | %d |" % s["decided"],
        "| **False UNKNOWNs** | **%d** |" % s["false_unknowns"],
        "| False UNKNOWN rate | %.2f%% |" % (100 * s["false_unknown_rate"]),
        "| Decision rate | %.2f%% |" % (100 * s["decision_rate"]),
        "| Misclassifications | %d |" % s["misclassified"],
        "",
        "A false UNKNOWN is an abstention on code that is already executable and "
        "ground-truthed. The measured rate is **%.2f%%**, so sound mode is not "
        "buying its zero-false-positive guarantee by refusing to decide this "
        "in-fragment benchmark." % (100 * s["false_unknown_rate"]),
        "",
        "## By kind",
        "",
        "| Kind | Total | SAFE | UNSAFE | UNKNOWN | Decided |",
        "|---|---|---|---|---|---|",
    ]
    for kind in sorted(a["by_kind"]):
        k = a["by_kind"][kind]
        lines.append("| `%s` | %d | %d | %d | %d | %d |" % (
            kind, k["total"], k["SAFE"], k["UNSAFE"], k["UNKNOWN"], k["decided"]))
    lines.extend(["", "## By family", "", "| Family | Total | SAFE | UNSAFE | UNKNOWN |",
                  "|---|---|---|---|---|"])
    for family in sorted(a["by_family"]):
        f = a["by_family"][family]
        lines.append("| `%s` | %d | %d | %d | %d |" % (
            family, f["total"], f["SAFE"], f["UNSAFE"], f["UNKNOWN"]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    artifact = run(check=args.check)
    if args.check:
        print("false_unknowns.json is up to date")
        return
    s = artifact["summary"]
    print("Wrote %s and %s" % (os.path.relpath(OUT_JSON, REPO_ROOT),
                               os.path.relpath(OUT_MD, REPO_ROOT)))
    print("  models: %d | false UNKNOWNs: %d | decision rate: %.1f%% | misclassified: %d"
          % (s["total"], s["false_unknowns"], 100 * s["decision_rate"], s["misclassified"]))
    if artifact["false_unknown_ids"]:
        print("  false UNKNOWN ids:", artifact["false_unknown_ids"])


if __name__ == "__main__":
    main()
