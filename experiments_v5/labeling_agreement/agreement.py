#!/usr/bin/env python3
"""Regenerate the Step-255 mined-bug labeling agreement artifact.

The checker is intentionally offline: it validates a dual-pass metadata audit
against the frozen provenance corpus and recomputes agreement/adjudication
summaries. It does not fetch issue bodies or patches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORPUS = REPO / "experiments_v5" / "provenance_bug_corpus" / "corpus.jsonl"
CORPUS_MANIFEST = REPO / "experiments_v5" / "provenance_bug_corpus" / "manifest.json"
ANNOTATIONS = HERE / "annotations.jsonl"
TAXONOMY = HERE / "ambiguity_taxonomy.json"
OUT_JSON = HERE / "agreement.json"
OUT_LOG = HERE / "adjudication_log.md"

SCHEMA_VERSION = "tensorguard.labeling-agreement.v1"
SAMPLE_SEED = "tensorguard-step255-labeling-v1"
SAMPLE_PER_CATEGORY = 4
RATERS = ("pass_a", "pass_b")
AXES: Dict[str, Tuple[str, ...]] = {
    "include_decision": ("include", "defer", "exclude"),
    "root_cause_family": (
        "shape_contract",
        "device_placement",
        "dtype_device_contract",
        "data_or_preprocess",
        "unknown_from_metadata",
        "out_of_scope",
    ),
    "evidence_strength": ("strong", "moderate", "weak"),
}


class AgreementError(RuntimeError):
    """Raised when committed labeling artifacts drift or become inconsistent."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_corpus() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = _load_jsonl(CORPUS)
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    computed = _sha256_text("".join(_canonical(record) + "\n" for record in records))
    expected = manifest["corpus_sha256"]
    if computed != expected:
        raise AgreementError(f"provenance corpus hash drift: {computed} != {expected}")
    return records, manifest


def expected_sample_ids(records: Sequence[Dict[str, Any]]) -> List[str]:
    by_category: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for record in records:
        digest = hashlib.sha256(f"{SAMPLE_SEED}\n{record['id']}".encode("utf-8")).hexdigest()
        by_category.setdefault(record["category"], []).append((digest, record))
    sample: List[str] = []
    for category in sorted(by_category):
        ranked = sorted(by_category[category], key=lambda item: (item[0], item[1]["id"]))
        if len(ranked) < SAMPLE_PER_CATEGORY:
            raise AgreementError(f"category {category!r} has fewer than {SAMPLE_PER_CATEGORY} rows")
        sample.extend(record["id"] for _, record in ranked[:SAMPLE_PER_CATEGORY])
    return sample


def _validate_label(label: Dict[str, Any], taxonomy_codes: set[str], where: str) -> None:
    for axis, values in AXES.items():
        if label.get(axis) not in values:
            raise AgreementError(f"{where}: invalid {axis}={label.get(axis)!r}")
    codes = label.get("ambiguity_codes")
    if not isinstance(codes, list) or not codes:
        raise AgreementError(f"{where}: ambiguity_codes must be a non-empty list")
    unknown = sorted(set(codes) - taxonomy_codes)
    if unknown:
        raise AgreementError(f"{where}: undefined ambiguity codes {unknown}")
    if label["include_decision"] == "exclude" and label["root_cause_family"] != "out_of_scope":
        raise AgreementError(f"{where}: exclude decisions must use root_cause_family=out_of_scope")


def load_and_validate() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    corpus, manifest = load_corpus()
    by_id = {record["id"]: record for record in corpus}
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    taxonomy_codes = set(taxonomy["codes"])
    rows = _load_jsonl(ANNOTATIONS)

    ids = [row.get("record_id") for row in rows]
    if ids != expected_sample_ids(corpus):
        raise AgreementError("annotations.jsonl does not match the deterministic stratified sample")
    if len(ids) != len(set(ids)):
        raise AgreementError("annotations contain duplicate record_id values")

    used_codes: set[str] = set()
    for row in rows:
        record_id = row["record_id"]
        if record_id not in by_id:
            raise AgreementError(f"annotation references unknown record {record_id!r}")
        for rater in RATERS:
            _validate_label(row[rater], taxonomy_codes, f"{record_id}/{rater}")
            used_codes.update(row[rater]["ambiguity_codes"])
        adjudication = row.get("adjudication")
        if not isinstance(adjudication, dict):
            raise AgreementError(f"{record_id}: missing adjudication")
        _validate_label(adjudication, taxonomy_codes, f"{record_id}/adjudication")
        used_codes.update(adjudication["ambiguity_codes"])
        disagreements = sorted(
            axis for axis in AXES if row["pass_a"][axis] != row["pass_b"][axis]
        )
        if sorted(adjudication.get("disagreement_axes", [])) != disagreements:
            raise AgreementError(
                f"{record_id}: disagreement_axes {adjudication.get('disagreement_axes')} != {disagreements}"
            )
        rationale = adjudication.get("rationale", "")
        if disagreements and len(rationale) < 20:
            raise AgreementError(f"{record_id}: disagreements require a rationale")

    unused_codes = sorted(taxonomy_codes - used_codes)
    if unused_codes:
        raise AgreementError(f"taxonomy defines unused codes: {unused_codes}")
    return corpus, rows, manifest, taxonomy


def _confusion(pairs: Sequence[Tuple[str, str]], labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    table = {a: {b: 0 for b in labels} for a in labels}
    for a, b in pairs:
        table[a][b] += 1
    return table


def cohen_kappa(pairs: Sequence[Tuple[str, str]], labels: Sequence[str]) -> Optional[float]:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum(left[label] * right[label] for label in labels) / (n * n)
    denom = 1.0 - expected
    if abs(denom) < 1e-12:
        return None
    return (observed - expected) / denom


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of an empty sequence")
    ranked = sorted(values)
    idx = int(round((len(ranked) - 1) * q))
    return ranked[max(0, min(len(ranked) - 1, idx))]


def _bootstrap_kappa_ci(
    axis: str,
    pairs: Sequence[Tuple[str, str]],
    labels: Sequence[str],
) -> Optional[List[float]]:
    rng = random.Random(f"{SAMPLE_SEED}:{axis}:bootstrap")
    vals: List[float] = []
    for _ in range(1000):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        kappa = cohen_kappa(sample, labels)
        if kappa is not None:
            vals.append(kappa)
    if not vals:
        return None
    return [round(_percentile(vals, 0.025), 6), round(_percentile(vals, 0.975), 6)]


def _metric(axis: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = AXES[axis]
    pairs = [(row["pass_a"][axis], row["pass_b"][axis]) for row in rows]
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    kappa = cohen_kappa(pairs, labels)
    return {
        "n": n,
        "labels": list(labels),
        "observed_agreement": round(agree / n, 6),
        "agreements": agree,
        "disagreements": n - agree,
        "cohen_kappa": None if kappa is None else round(kappa, 6),
        "cohen_kappa_bootstrap_95ci": _bootstrap_kappa_ci(axis, pairs, labels),
        "confusion_pass_a_rows_pass_b_columns": _confusion(pairs, labels),
    }


def _count(items: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(items).items()))


def build() -> Tuple[Dict[str, Any], str]:
    corpus, rows, manifest, taxonomy = load_and_validate()
    by_id = {record["id"]: record for record in corpus}
    selected = [by_id[row["record_id"]] for row in rows]
    disagreement_rows = [
        row for row in rows if row["adjudication"]["disagreement_axes"]
    ]
    metrics = {axis: _metric(axis, rows) for axis in AXES}
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "experiments_v5/labeling_agreement/agreement.py",
        "annotation_provenance": {
            "status": "dual_pass_repository_metadata_audit",
            "not_human_subjects_study": True,
            "external_independent_annotators_claimed": False,
            "measured_object": (
                "reviewer judgment over inclusion, root-cause family, and evidence "
                "strength; mechanical runtime-signature category labels are not the "
                "headline agreement measure"
            ),
        },
        "source_corpus": {
            "path": "experiments_v5/provenance_bug_corpus/corpus.jsonl",
            "manifest_path": "experiments_v5/provenance_bug_corpus/manifest.json",
            "records": manifest["total"],
            "corpus_sha256": manifest["corpus_sha256"],
        },
        "sample": {
            "seed": SAMPLE_SEED,
            "per_category": SAMPLE_PER_CATEGORY,
            "records": len(rows),
            "by_category": _count(record["category"] for record in selected),
            "by_github_kind": _count(record["github_kind"] for record in selected),
            "by_project_family": _count(record["project_family"] for record in selected),
            "record_ids_sha256": _sha256_text("\n".join(row["record_id"] for row in rows) + "\n"),
        },
        "inputs": {
            "annotations_sha256": _sha256_file(ANNOTATIONS),
            "taxonomy_sha256": _sha256_file(TAXONOMY),
            "rubric_path": "experiments_v5/labeling_agreement/rubric.md",
            "taxonomy_path": "experiments_v5/labeling_agreement/ambiguity_taxonomy.md",
        },
        "metrics": metrics,
        "adjudication": {
            "records_with_any_disagreement": len(disagreement_rows),
            "records_without_disagreement": len(rows) - len(disagreement_rows),
            "disagreement_axes": _count(
                axis for row in rows for axis in row["adjudication"]["disagreement_axes"]
            ),
            "final_include_decision": _count(row["adjudication"]["include_decision"] for row in rows),
            "final_root_cause_family": _count(row["adjudication"]["root_cause_family"] for row in rows),
            "final_evidence_strength": _count(row["adjudication"]["evidence_strength"] for row in rows),
            "ambiguity_codes_used": _count(
                code for row in rows for code in row["adjudication"]["ambiguity_codes"]
            ),
        },
        "limitations": [
            "This is a repository-authored dual-pass audit, not a completed external human-subjects study.",
            "The sample is stratified by mechanical runtime-signature category, so it is not prevalence-weighted.",
            "Rows marked defer should not be used as gold precision/recall positives until third-party context is refetched or a fix is inspected.",
        ],
    }
    return artifact, render_log(artifact, rows, selected, taxonomy)


def _md(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_log(
    artifact: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    taxonomy: Dict[str, Any],
) -> str:
    by_id = {record["id"]: record for record in selected}
    out: List[str] = [
        "# Step 255 labeling adjudication log",
        "",
        "This log is generated by `agreement.py` from `annotations.jsonl`.",
        "It records a dual-pass repository metadata audit, not a human-subjects study.",
        "",
        "## Agreement metrics",
        "",
        "| Axis | Observed agreement | Cohen kappa | 95% bootstrap CI | Disagreements |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for axis, metric in artifact["metrics"].items():
        ci = metric["cohen_kappa_bootstrap_95ci"]
        ci_text = "n/a" if ci is None else f"[{ci[0]}, {ci[1]}]"
        out.append(
            f"| `{axis}` | {metric['observed_agreement']:.3f} | "
            f"{metric['cohen_kappa']:.3f} | {ci_text} | {metric['disagreements']} |"
        )

    out.extend([
        "",
        "## Adjudicated disagreements",
        "",
        "| Record | Category | Title | Pass A | Pass B | Final | Ambiguity | Rationale |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in rows:
        axes = row["adjudication"]["disagreement_axes"]
        if not axes:
            continue
        record = by_id[row["record_id"]]
        def compact(label: Dict[str, Any]) -> str:
            return "/".join(label[axis] for axis in AXES)
        out.append(
            "| "
            + " | ".join(
                [
                    f"`{_md(row['record_id'])}`",
                    f"`{_md(record['category'])}`",
                    _md(record["title"]),
                    f"`{compact(row['pass_a'])}`",
                    f"`{compact(row['pass_b'])}`",
                    f"`{compact(row['adjudication'])}`",
                    ", ".join(f"`{code}`" for code in row["adjudication"]["ambiguity_codes"]),
                    _md(row["adjudication"]["rationale"]),
                ]
            )
            + " |"
        )

    out.extend([
        "",
        "## Taxonomy coverage",
        "",
        "| Code | Summary |",
        "| --- | --- |",
    ])
    used = artifact["adjudication"]["ambiguity_codes_used"]
    for code, meta in sorted(taxonomy["codes"].items()):
        out.append(f"| `{code}` ({used[code]} uses) | {_md(meta['summary'])} |")
    out.append("")
    return "\n".join(out)


def write() -> Dict[str, Any]:
    artifact, log = build()
    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_LOG.write_text(log, encoding="utf-8")
    return artifact


def check() -> int:
    artifact, log = build()
    expected_json = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    problems: List[str] = []
    if not OUT_JSON.exists() or OUT_JSON.read_text(encoding="utf-8") != expected_json:
        problems.append(str(OUT_JSON.relative_to(REPO)))
    if not OUT_LOG.exists() or OUT_LOG.read_text(encoding="utf-8") != log:
        problems.append(str(OUT_LOG.relative_to(REPO)))
    if problems:
        print("MISMATCH: regenerate " + ", ".join(problems))
        return 1
    print(
        "OK: Step 255 labeling agreement verified "
        f"({artifact['sample']['records']} records, "
        f"include kappa {artifact['metrics']['include_decision']['cohen_kappa']})."
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed agreement artifacts")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    artifact = write()
    print(
        f"Wrote {OUT_JSON.relative_to(REPO)} and {OUT_LOG.relative_to(REPO)} "
        f"for {artifact['sample']['records']} labeled records."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
