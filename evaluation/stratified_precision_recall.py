#!/usr/bin/env python3
"""Step 250: stratified precision/recall with honest sample-size gates.

This script is deliberately a post-processor: it consumes the committed
``evaluation/confusion_matrices.json`` predictions instead of re-running
TensorGuard, PyTea, or runtime baselines.  That keeps ``--check`` deterministic
on hosts without Node, CUDA, or a complete PyTorch stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation import precision_recall as pr  # noqa: E402
from real_benchmarks import load  # noqa: E402

OUT_JSON = THIS_DIR / "stratified_precision_recall.json"
OUT_MD = THIS_DIR / "stratified_precision_recall.md"
PROVENANCE_JSONL = REPO / "experiments_v5" / "provenance_bug_corpus" / "corpus.jsonl"

SCHEMA_VERSION = "tensorguard.stratified-precision-recall.v1"
Z_95 = 1.959963984540054

EXECUTABLE_GATES = {
    "min_total_n": 4,
    "min_precision_denominator": 5,
    "min_recall_denominator": 5,
    "min_false_positive_denominator": 5,
    "min_accuracy_denominator": 10,
    "min_coverage_denominator": 10,
}
PROVENANCE_GATES = {"min_records_per_stratum": 30}

EXECUTABLE_DIMENSIONS = (
    "operator_family",
    "framework",
    "bug_class",
    "model_family",
    "source",
)

OPERATOR_FAMILY_BY_ID = {
    "clean_mlp": "matmul_linear",
    "clean_cnn": "convolution",
    "clean_resblock": "residual_add",
    "clean_layernorm_mlp": "normalization_linear",
    "clean_conv_bn_pool": "convolution_pool_linear",
    "clean_self_attention": "attention_matmul",
    "clean_groupnorm": "normalization_convolution",
    "clean_dropout_mlp": "phase_dropout_linear",
    "buggy_linear_inout_mismatch": "matmul_linear",
    "buggy_view_total_size": "reshape_view",
    "buggy_conv_channel_mismatch": "convolution",
    "buggy_flatten_fc_mismatch": "matmul_linear",
    "buggy_cat_dim_mismatch": "broadcast_concat",
    "buggy_matmul_inner_mismatch": "matmul_linear",
    "buggy_device_mismatch": "device_placement",
    "buggy_gradient_detach": "gradient_flow",
}

MODEL_FAMILY_BY_ID = {
    "clean_mlp": "mlp",
    "clean_cnn": "cnn",
    "clean_resblock": "residual_cnn",
    "clean_layernorm_mlp": "norm_mlp",
    "clean_conv_bn_pool": "conv_pool_classifier",
    "clean_self_attention": "attention",
    "clean_groupnorm": "normalization_cnn",
    "clean_dropout_mlp": "phase_mlp",
    "buggy_linear_inout_mismatch": "mlp",
    "buggy_view_total_size": "reshape_module",
    "buggy_conv_channel_mismatch": "cnn",
    "buggy_flatten_fc_mismatch": "cnn",
    "buggy_cat_dim_mismatch": "branching_mlp",
    "buggy_matmul_inner_mismatch": "matrix_module",
    "buggy_device_mismatch": "device_module",
    "buggy_gradient_detach": "gradient_mlp",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def _wilson(successes: int, total: int) -> List[float] | None:
    if total <= 0:
        return None
    p = successes / total
    z2 = Z_95 * Z_95
    denom = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    half = (
        Z_95
        * math.sqrt((p * (1 - p) / total) + (z2 / (4 * total * total)))
        / denom
    )
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return [round(lo, 4), round(hi, 4)]


def _binomial_metric(successes: int, total: int, min_total: int) -> Dict[str, Any]:
    estimate = _round(successes / total) if total else None
    ci = _wilson(successes, total)
    claimable = total >= min_total if total else False
    return {
        "estimate": estimate,
        "ci95": ci,
        "successes": successes,
        "n": total,
        "min_n": min_total,
        "claimable": claimable,
        "gate": "pass" if claimable else "insufficient_n",
    }


def _load_confusion_artifact() -> Dict[str, Any]:
    path = Path(pr.OUT_JSON)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(REPO)} is required; run evaluation/precision_recall.py "
            "in an environment with the intended baselines, then commit it."
        )
    with path.open(encoding="utf-8") as fh:
        artifact = json.load(fh)
    if artifact.get("meta", {}).get("corpus") != "real_benchmarks":
        raise ValueError("confusion_matrices.json is not the real_benchmarks artifact")
    return artifact


def _strata_for_item(item: Mapping[str, Any]) -> Dict[str, str]:
    item_id = item["id"]
    try:
        operator_family = OPERATOR_FAMILY_BY_ID[item_id]
        model_family = MODEL_FAMILY_BY_ID[item_id]
    except KeyError as exc:
        raise KeyError(f"missing Step-250 stratum mapping for {item_id}") from exc
    return {
        "operator_family": operator_family,
        "framework": "pytorch_nn_module",
        "bug_class": item["category"] if item["label"] == pr.BUGGY else "clean",
        "model_family": model_family,
        "source": item["provenance_type"],
    }


def _executable_rows(confusion_artifact: Mapping[str, Any]) -> List[Dict[str, Any]]:
    items = {item["id"]: item for item in load.load_items(verify=True)}
    rows: List[Dict[str, Any]] = []
    for predicted in confusion_artifact["per_model"]:
        item = items[predicted["id"]]
        strata = _strata_for_item(item)
        rows.append({
            "id": item["id"],
            "label": item["label"],
            "domain": item["domain"],
            "category": item["category"],
            "provenance_type": item["provenance_type"],
            "source_url_present": item.get("source_url") is not None,
            "strata": strata,
            "predictions": predicted["predictions"],
        })
    return rows


def _score_rows(rows: Sequence[Mapping[str, Any]], method: str) -> Dict[str, Any]:
    labels = [(row["label"], row["predictions"][method]["pred"]) for row in rows]
    c = pr.confusion(labels)
    tp, fp, tn, fn, na, n = (c[k] for k in ("TP", "FP", "TN", "FN", "NA", "N"))
    precision_den = tp + fp
    recall_den = tp + fn
    fpr_den = fp + tn
    metrics = {
        "precision": _binomial_metric(
            tp, precision_den, EXECUTABLE_GATES["min_precision_denominator"]
        ),
        "recall": _binomial_metric(
            tp, recall_den, EXECUTABLE_GATES["min_recall_denominator"]
        ),
        "false_positive_rate": _binomial_metric(
            fp, fpr_den, EXECUTABLE_GATES["min_false_positive_denominator"]
        ),
        "accuracy": _binomial_metric(
            tp + tn, n, EXECUTABLE_GATES["min_accuracy_denominator"]
        ),
        "coverage": _binomial_metric(
            n - na, n, EXECUTABLE_GATES["min_coverage_denominator"]
        ),
    }
    claimable_metrics = sorted(k for k, v in metrics.items() if v["claimable"])
    sparse_reasons = [
        f"{name}: n={metric['n']} < {metric['min_n']}"
        for name, metric in sorted(metrics.items())
        if not metric["claimable"]
    ]
    return {
        "N": n,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "NA": na,
        "positive_n": recall_den,
        "negative_n": fpr_den,
        "conservative_predicted_positive_n": precision_den,
        "item_ids": sorted(row["id"] for row in rows),
        "metrics": metrics,
        "claimable_metrics": claimable_metrics,
        "publication_gate": {
            "status": (
                "precision_recall_claimable"
                if {"precision", "recall"}.issubset(claimable_metrics)
                else "exploratory_only"
            ),
            "min_total_n": EXECUTABLE_GATES["min_total_n"],
            "passes_min_total_n": n >= EXECUTABLE_GATES["min_total_n"],
            "sparse_reasons": sparse_reasons,
        },
    }


def _stratified_executable(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, Any] = {}
    for method in pr.METHODS:
        method_out: Dict[str, Any] = {}
        for dim in EXECUTABLE_DIMENSIONS:
            buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                buckets[row["strata"][dim]].append(row)
            method_out[dim] = {
                key: _score_rows(sorted(vals, key=lambda r: r["id"]), method)
                for key, vals in sorted(buckets.items())
            }
        by_method[method] = method_out
    return by_method


def _dimension_summary(stratified: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for dim in EXECUTABLE_DIMENSIONS:
        strata = stratified["tensorguard"][dim]
        summary[dim] = {
            "n_strata": len(strata),
            "claimable_precision_recall_strata_by_method": {
                method: sum(
                    1
                    for result in stratified[method][dim].values()
                    if result["publication_gate"]["status"]
                    == "precision_recall_claimable"
                )
                for method in pr.METHODS
            },
            "tensor_guard_exploratory_strata": sorted(
                key
                for key, result in strata.items()
                if result["publication_gate"]["status"] == "exploratory_only"
            ),
        }
    summary["framework"]["degenerate_axis"] = (
        summary["framework"]["n_strata"] == 1
    )
    return summary


def _load_provenance_records() -> List[Dict[str, Any]]:
    if not PROVENANCE_JSONL.exists():
        raise FileNotFoundError(PROVENANCE_JSONL)
    records = []
    with PROVENANCE_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _record_framework(record: Mapping[str, Any]) -> str:
    # The Step-249 corpus is mined from PyTorch runtime-error signatures.  Keep
    # framework as the literal library under test; project/ecosystem diversity is
    # represented by model_family instead of being mislabeled as framework.
    return "pytorch"


def _record_source(record: Mapping[str, Any]) -> str:
    return "github_" + record["github_kind"]


def _provenance_value(record: Mapping[str, Any], dim: str) -> str:
    if dim == "operator_family":
        return record["runtime_signature"]["operator_family"]
    if dim == "framework":
        return _record_framework(record)
    if dim == "bug_class":
        return record["category"]
    if dim == "model_family":
        return record["project_family"]
    if dim == "source":
        return _record_source(record)
    raise KeyError(dim)


def _count_provenance(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dim in EXECUTABLE_DIMENSIONS:
        counts = Counter(_provenance_value(record, dim) for record in records)
        strata = {}
        for value, count in sorted(counts.items()):
            metric = _binomial_metric(
                count, len(records), PROVENANCE_GATES["min_records_per_stratum"]
            )
            claimable = count >= PROVENANCE_GATES["min_records_per_stratum"]
            metric["claimable"] = claimable
            metric["gate"] = "pass" if claimable else "insufficient_n"
            metric["min_n"] = PROVENANCE_GATES["min_records_per_stratum"]
            metric["share_of_positive_corpus"] = metric.pop("estimate")
            metric["records"] = count
            strata[value] = metric
        out[dim] = {
            "n_strata": len(strata),
            "strata": strata,
            "claimable_sample_size_strata": sorted(
                key for key, value in strata.items() if value["claimable"]
            ),
        }
    out["framework"]["degenerate_axis"] = out["framework"]["n_strata"] == 1
    return out


def measure() -> Dict[str, Any]:
    confusion = _load_confusion_artifact()
    rows = _executable_rows(confusion)
    stratified = _stratified_executable(rows)
    provenance_records = _load_provenance_records()
    overall = {method: _score_rows(rows, method) for method in pr.METHODS}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "evaluation/stratified_precision_recall.py",
        "inputs": {
            "confusion_matrices_json": "evaluation/confusion_matrices.json",
            "confusion_matrices_sha256": _sha256(Path(pr.OUT_JSON)),
            "real_benchmarks_version": load.load_manifest()["meta"]["version"],
            "provenance_bug_corpus_jsonl": (
                "experiments_v5/provenance_bug_corpus/corpus.jsonl"
            ),
            "provenance_bug_corpus_sha256": _sha256(PROVENANCE_JSONL),
            "provenance_positive_records": len(provenance_records),
        },
        "minimum_sample_gates": {
            "executable_precision_recall": EXECUTABLE_GATES,
            "provenance_positive_only_coverage": PROVENANCE_GATES,
        },
        "honesty_notes": [
            "Executable precision/recall strata are scored from the frozen real-code "
            "predictions; the script never re-runs detectors.",
            "The framework axis is intentionally degenerate in the current evidence: "
            "all executable models are PyTorch nn.Module cases and all mined records "
            "come from PyTorch runtime-error signatures.",
            "Operator-family, bug-class, and model-family slices are sparse and "
            "partly collinear on the 16-case executable corpus; sparse slices are "
            "marked exploratory instead of being promoted to paper claims.",
            "The 2,704-record provenance corpus is positive-only and is used here "
            "only for sample-size coverage, not for precision/recall scoring.",
        ],
        "methods": pr.METHODS,
        "dimensions": list(EXECUTABLE_DIMENSIONS),
        "executable_rows": rows,
        "overall_by_method": overall,
        "executable_strata": stratified,
        "executable_dimension_summary": _dimension_summary(stratified),
        "provenance_positive_only_sample_sizes": _count_provenance(
            provenance_records
        ),
    }


def _fmt_metric(metric: Mapping[str, Any]) -> str:
    est = metric.get("estimate")
    if est is None:
        est = metric.get("share_of_positive_corpus")
    ci = metric["ci95"]
    if est is None or ci is None:
        return "--"
    suffix = "" if metric["claimable"] else " (expl.)"
    return f"{est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]{suffix}"


def render_markdown(artifact: Mapping[str, Any]) -> str:
    lines = [
        "# Stratified precision/recall with sample-size gates (Step 250)",
        "",
        "This artifact post-processes the committed real-code predictions in "
        "`evaluation/confusion_matrices.json`. It does **not** re-run detectors, "
        "so PyTea/runtime availability cannot silently change the strata.",
        "",
        "Wilson score confidence intervals are shown as `estimate [low, high]`; "
        "`(expl.)` marks a metric whose denominator is below the publication gate.",
        "",
        "## Executable corpus: TensorGuard strata",
        "",
    ]
    tg = artifact["executable_strata"]["tensorguard"]  # type: ignore[index]
    for dim in EXECUTABLE_DIMENSIONS:
        lines.extend([
            f"### {dim}",
            "",
            "| stratum | N | TP | FP | TN | FN | precision | recall | FPR | gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ])
        for stratum, row in tg[dim].items():
            metrics = row["metrics"]
            lines.append(
                "| `%s` | %d | %d | %d | %d | %d | %s | %s | %s | %s |"
                % (
                    stratum,
                    row["N"],
                    row["TP"],
                    row["FP"],
                    row["TN"],
                    row["FN"],
                    _fmt_metric(metrics["precision"]),
                    _fmt_metric(metrics["recall"]),
                    _fmt_metric(metrics["false_positive_rate"]),
                    row["publication_gate"]["status"],
                )
            )
        lines.append("")
    lines.extend([
        "## Provenance corpus sample-size coverage",
        "",
        "The GitHub-mined corpus below is positive-only; it supports sample-size "
        "coverage claims, not precision/recall claims.",
        "",
    ])
    prov = artifact["provenance_positive_only_sample_sizes"]  # type: ignore[index]
    for dim in EXECUTABLE_DIMENSIONS:
        lines.extend([
            f"### {dim}",
            "",
            "| stratum | records | share with Wilson CI | sample-size gate |",
            "| --- | ---: | --- | --- |",
        ])
        for stratum, row in prov[dim]["strata"].items():
            lines.append(
                "| `%s` | %d | %s | %s |"
                % (
                    stratum,
                    row["records"],
                    _fmt_metric(row),
                    "pass" if row["claimable"] else "insufficient_n",
                )
            )
        lines.append("")
    lines.append("## Honesty notes")
    lines.append("")
    for note in artifact["honesty_notes"]:  # type: ignore[index]
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _json_text(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True) + "\n"


def run(check: bool = False) -> int:
    artifact = measure()
    json_text = _json_text(artifact)
    md_text = render_markdown(artifact)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text(encoding="utf-8") != json_text:
            print(f"MISMATCH: {OUT_JSON.relative_to(REPO)}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text(encoding="utf-8") != md_text:
            print(f"MISMATCH: {OUT_MD.relative_to(REPO)}")
            ok = False
        if ok:
            print("stratified_precision_recall: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(json_text, encoding="utf-8")
    OUT_MD.write_text(md_text, encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed artifacts")
    args = parser.parse_args(argv)
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
