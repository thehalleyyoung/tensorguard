"""Pre-specified evaluation protocol registry (Step 251).

This artifact is the methodology contract for the next evaluation tranche.  It
does not score TensorGuard.  Instead it freezes, in one deterministic place, the
case splits, tuning locks, metric formulas, and analysis scripts that downstream
scoring artifacts must use.  The generated JSON/Markdown hashes every referenced
manifest and script so a reviewer can tell whether a result was produced under
the registered protocol or after the protocol changed.

The protocol deliberately distinguishes two notions:

* **historical pre-registration** is a git-history property (the protocol commit
  must predate later scoring-result commits);
* **pipeline order** is a consistency guard: in ``reproduce_all.py`` the protocol
  is regenerated before the downstream scoring scripts it governs.

``--check`` regenerates and byte-diffs the committed artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "reproducibility" / "evaluation_protocol.json"
OUT_MD = REPO / "reproducibility" / "evaluation_protocol.md"

DEV_MANIFEST = "corpus_extended/manifest.json"
BLIND_MANIFEST = "corpus_extended/blind_manifest.json"
REAL_MANIFEST = "real_benchmarks/manifest.json"
BLIND_PREREG = "corpus_extended/PRE_REGISTRATION.md"

GOVERNED_SCORING_SCRIPTS = [
    "reproducibility/corpus_extended_score.py",
    "reproducibility/corpus_stratified.py",
    "reproducibility/blind_split_eval.py",
    "reproducibility/baseline_head_to_head.py",
    "reproducibility/fp_stress_eval.py",
    "reproducibility/natural_distribution_study.py",
    "reproducibility/mutation_clean_models.py",
    "reproducibility/differential_dispatcher.py",
    "reproducibility/domain_ablation.py",
    "reproducibility/reduced_product_ablation.py",
    "reproducibility/cegar_depth_ablation.py",
    "reproducibility/effect_sizes.py",
    "reproducibility/statistical_power.py",
]

LEGACY_HASH_REGISTERED_SCRIPTS = [
    "evaluation/significance.py",
    "evaluation/stratified_precision_recall.py",
    "evaluation/precision_recall.py",
]


def _sha256(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def _load_json(rel: str) -> Any:
    return json.loads((REPO / rel).read_text())


def _items(rel: str) -> List[dict]:
    data = _load_json(rel)
    return list(data.get("items", []))


def _label_counts(items: Iterable[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        label = str(item.get("label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _split_manifest_summary(rel: str, split_id: str, role: str,
                            held_out: bool, tuning_allowed: str) -> dict:
    items = _items(rel)
    families = sorted({str(i.get("family", i.get("category", "unknown")))
                       for i in items})
    domains = sorted({str(i.get("domain", "unknown")) for i in items})
    return {
        "id": split_id,
        "role": role,
        "manifest": rel,
        "manifest_sha256": _sha256(rel),
        "n_cases": len(items),
        "label_counts": _label_counts(items),
        "n_families": len(families),
        "families": families,
        "domains": domains,
        "held_out": held_out,
        "tuning_allowed": tuning_allowed,
    }


def _blind_preregistration() -> dict:
    text = (REPO / BLIND_PREREG).read_text()
    hashes = re.findall(r"`([0-9a-f]{64})`", text)
    count_match = re.search(
        r"\((\d+) cases,\s*(\d+) buggy\s*/\s*(\d+) clean,\s*(\d+) families\)",
        text,
        re.S,
    )
    counts = None
    if count_match:
        counts = {
            "n_cases": int(count_match.group(1)),
            "buggy": int(count_match.group(2)),
            "clean": int(count_match.group(3)),
            "n_families": int(count_match.group(4)),
        }
    observed = _sha256(BLIND_MANIFEST)
    registered = hashes[0] if hashes else None
    return {
        "document": BLIND_PREREG,
        "registered_manifest_sha256": registered,
        "observed_manifest_sha256": observed,
        "hash_matches_document": registered == observed,
        "registered_counts": counts,
    }


def _disjointness() -> dict:
    dev_ids = {str(i["id"]) for i in _items(DEV_MANIFEST)}
    blind_ids = {str(i["id"]) for i in _items(BLIND_MANIFEST)}
    overlap = sorted(dev_ids & blind_ids)
    return {
        "development_vs_blind_disjoint": not overlap,
        "development_n_ids": len(dev_ids),
        "blind_n_ids": len(blind_ids),
        "intersection_n": len(overlap),
        "intersection_ids": overlap,
    }


def _script_entry(path: str, stage: str, case_universe: str,
                  governed: bool) -> dict:
    full = REPO / path
    return {
        "path": path,
        "stage": stage,
        "case_universe": case_universe,
        "governed_by_protocol": governed,
        "present": full.exists(),
        "sha256": _sha256(path) if full.exists() else None,
    }


def _analysis_scripts() -> List[dict]:
    rows = [
        _script_entry(
            "reproducibility/evaluation_protocol.py",
            "registration",
            "split manifests + analysis source hashes",
            False,
        )
    ]
    for path in GOVERNED_SCORING_SCRIPTS:
        if "blind_split" in path:
            universe = BLIND_MANIFEST
        elif "fp_stress" in path or "natural_distribution" in path:
            universe = "clean-model holdout corpus generated by the script"
        elif "effect_sizes" in path:
            universe = "evaluation/confusion_matrices.json"
        elif "statistical_power" in path:
            universe = "committed headline-result artifacts"
        else:
            universe = DEV_MANIFEST
        rows.append(_script_entry(path, "governed_scoring", universe, True))
    for path in LEGACY_HASH_REGISTERED_SCRIPTS:
        rows.append(_script_entry(
            path,
            "legacy_fixed_input",
            "hash-registered pre-existing artifact",
            False,
        ))
    return rows


def _step_index(path: str) -> int | None:
    import reproducibility.reproduce_all as ra

    for idx, (_name, argv, _stdout_path) in enumerate(ra.STEPS):
        if any(path == str(arg) for arg in argv):
            return idx
    return None


def _reproduction_order() -> dict:
    protocol_idx = _step_index("reproducibility/evaluation_protocol.py")
    governed = []
    for path in GOVERNED_SCORING_SCRIPTS:
        idx = _step_index(path)
        governed.append({
            "path": path,
            "step_index": idx,
            "after_protocol": (
                protocol_idx is not None and idx is not None and protocol_idx < idx
            ),
        })
    return {
        "protocol_step_index": protocol_idx,
        "governed_scoring_scripts": governed,
        "protocol_precedes_governed_scoring": all(
            row["after_protocol"] for row in governed
        ),
        "scope_note": (
            "This is a reproduction-order consistency guard for scripts in the "
            "Step-251 protocol. Historical pre-registration must be assessed "
            "from git history, not from a same-commit rebuild."
        ),
    }


def _metric_definitions() -> List[dict]:
    return [
        {
            "id": "recall_on_decided",
            "formula": "TP / (TP + FN), excluding UNKNOWN verdicts from the denominator",
            "scripts": [
                "reproducibility/corpus_extended_score.py",
                "reproducibility/blind_split_eval.py",
            ],
        },
        {
            "id": "recall_on_all_buggy",
            "formula": "TP / all buggy cases, counting UNKNOWN as not caught",
            "scripts": ["reproducibility/corpus_extended_score.py"],
        },
        {
            "id": "specificity_on_decided",
            "formula": "TN / (TN + FP), excluding UNKNOWN clean verdicts",
            "scripts": [
                "reproducibility/corpus_extended_score.py",
                "reproducibility/fp_stress_eval.py",
            ],
        },
        {
            "id": "false_positive_rate",
            "formula": "FP / (FP + TN) on clean cases with definite verdicts",
            "scripts": [
                "reproducibility/corpus_extended_score.py",
                "reproducibility/natural_distribution_study.py",
            ],
        },
        {
            "id": "precision",
            "formula": "TP / (TP + FP) over positive UNSAFE predictions",
            "scripts": ["reproducibility/corpus_extended_score.py"],
        },
        {
            "id": "abstention_rate",
            "formula": "UNKNOWN verdicts / cases in the evaluated split",
            "scripts": [
                "reproducibility/corpus_extended_score.py",
                "reproducibility/natural_distribution_study.py",
            ],
        },
        {
            "id": "wilson_95_ci",
            "formula": "Wilson score interval with z=1.959963984540054",
            "scripts": [
                "reproducibility/corpus_extended_score.py",
                "reproducibility/statistical_power.py",
            ],
        },
        {
            "id": "paired_mcnemar",
            "formula": "exact paired McNemar over discordant correctness counts b,c",
            "scripts": [
                "evaluation/significance.py",
                "reproducibility/effect_sizes.py",
            ],
        },
        {
            "id": "paired_effect_size",
            "formula": "Cohen's g, Haldane-Anscombe odds ratio, risk difference",
            "scripts": ["reproducibility/effect_sizes.py"],
        },
        {
            "id": "power_and_sample_size",
            "formula": "exact binomial power / rule-of-three / McNemar discordant floor",
            "scripts": ["reproducibility/statistical_power.py"],
        },
        {
            "id": "overfitting_gap",
            "formula": "abs(blind recall_on_decided - development recall_on_decided)",
            "scripts": ["reproducibility/blind_split_eval.py"],
        },
    ]


def _tuning_freeze() -> List[dict]:
    return [
        {
            "id": "split_roles_locked",
            "rule": (
                "corpus_extended/manifest.json is the development/tuning corpus; "
                "corpus_extended/blind_manifest.json is the primary held-out split."
            ),
            "enforced_by": "manifest hashes + dev/blind disjointness check",
        },
        {
            "id": "no_blind_threshold_tuning",
            "rule": (
                "Soundness mode, CEGAR budgets, bug-category mapping, and operator "
                "confidence thresholds must be chosen before reading blind verdicts."
            ),
            "enforced_by": "protocol hash and reproduction-order guard",
        },
        {
            "id": "modes_fixed",
            "rule": "Report balanced and sound modes; heuristic mode is exploratory unless explicitly labeled.",
            "enforced_by": "metric registry and downstream artifact schema tests",
        },
        {
            "id": "baseline_case_policy",
            "rule": (
                "Baselines added after this protocol must score the same registered "
                "case IDs when they support the input format, and must disclose "
                "unsupported/NA cases rather than dropping them silently."
            ),
            "enforced_by": "analysis-script registry + future Step-252 benchmark tests",
        },
        {
            "id": "negative_result_policy",
            "rule": (
                "False positives, false negatives, UNKNOWN verdicts, unsupported "
                "baselines, and under-powered strata remain in the artifact and are "
                "reported; no post-hoc exclusion without a new protocol version."
            ),
            "enforced_by": "metric definitions + paper-evidence index",
        },
    ]


def measure() -> Dict[str, Any]:
    splits = [
        _split_manifest_summary(
            DEV_MANIFEST,
            "development_corpus",
            "development/tuning corpus",
            False,
            "permitted before this protocol version; locked for downstream scoring",
        ),
        _split_manifest_summary(
            BLIND_MANIFEST,
            "primary_heldout_blind_split",
            "primary held-out generalization split",
            True,
            "forbidden",
        ),
        _split_manifest_summary(
            REAL_MANIFEST,
            "external_real_benchmark",
            "small provenance-rich external sanity corpus",
            True,
            "forbidden",
        ),
    ]
    scripts = _analysis_scripts()
    return {
        "step": 251,
        "schema": 1,
        "protocol_name": "TensorGuard pre-specified evaluation protocol",
        "protocol_version": "1.0.0",
        "scope": (
            "Registers split roles, tuning freezes, metric definitions, and "
            "analysis-script hashes for downstream Step-251+ evaluation artifacts."
        ),
        "splits": splits,
        "blind_preregistration": _blind_preregistration(),
        "split_disjointness": _disjointness(),
        "tuning_freeze": _tuning_freeze(),
        "metric_definitions": _metric_definitions(),
        "analysis_scripts": scripts,
        "all_analysis_scripts_present": all(s["present"] for s in scripts),
        "reproduction_order": _reproduction_order(),
    }


def render_markdown(d: Dict[str, Any]) -> str:
    lines = [
        "# Pre-specified evaluation protocol (Step 251)",
        "",
        f"**{d['protocol_name']} v{d['protocol_version']}** registers the "
        "case splits, tuning locks, metric formulas, and analysis scripts used "
        "by downstream TensorGuard evaluations. It is a methodology artifact, "
        "not a scorer: generated outputs below are hashes, counts, formulas, "
        "and consistency checks.",
        "",
        "> Historical pre-registration is established by git history: this "
        "protocol commit must predate future scoring-result commits. The "
        "reproduction-order check below is a consistency guard for one-command "
        "rebuilds, not a substitute for temporal provenance.",
        "",
        "## Registered splits",
        "",
        "| split | role | held out? | cases | labels | families | manifest sha256 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split in d["splits"]:
        labels = ", ".join(
            f"{k}={v}" for k, v in split["label_counts"].items()
        )
        lines.append(
            f"| {split['id']} | {split['role']} | {split['held_out']} "
            f"| {split['n_cases']} | {labels} | {split['n_families']} "
            f"| `{split['manifest_sha256'][:16]}...` |"
        )
    dis = d["split_disjointness"]
    blind = d["blind_preregistration"]
    lines += [
        "",
        "## Held-out registration checks",
        "",
        "| check | value |",
        "| --- | --- |",
        f"| development/blind id intersection | {dis['intersection_n']} |",
        f"| development vs blind disjoint | {dis['development_vs_blind_disjoint']} |",
        f"| blind manifest hash matches PRE_REGISTRATION.md | {blind['hash_matches_document']} |",
        f"| governed scoring scripts after protocol in reproduce_all.py | "
        f"{d['reproduction_order']['protocol_precedes_governed_scoring']} |",
        "",
        "## Tuning freeze",
        "",
        "| rule | enforcement |",
        "| --- | --- |",
    ]
    for rule in d["tuning_freeze"]:
        lines.append(f"| {rule['id']} | {rule['enforced_by']} |")
    lines += [
        "",
        "## Metric definitions",
        "",
        "| metric | formula |",
        "| --- | --- |",
    ]
    for metric in d["metric_definitions"]:
        lines.append(f"| {metric['id']} | {metric['formula']} |")
    lines += [
        "",
        "## Analysis-script registry",
        "",
        "| script | stage | governed? | case universe | sha256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for script in d["analysis_scripts"]:
        lines.append(
            f"| `{script['path']}` | {script['stage']} "
            f"| {script['governed_by_protocol']} | {script['case_universe']} "
            f"| `{str(script['sha256'])[:16]}...` |"
        )
    lines.append("")
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
            print("evaluation_protocol: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
