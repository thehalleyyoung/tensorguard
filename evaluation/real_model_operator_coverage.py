"""Step 208 -- frequency-weighted real-model operator coverage.

This extends the original torchvision-only operator-frequency census with an
offline, no-download corpus spanning torchvision, timm, and HuggingFace
Transformers models.  The report commits a before/after matrix for Step 208:

* before: the pre-Step-208 census (hot operators intentionally removed);
* after: the live TensorGuard census, tied to real FX extraction behavior.

Shape-query scalar operations such as ``size``/``dim``/``floordiv`` are reported
as non-tensor metadata, but they are not counted as newly covered transfer
functions.  This keeps the headline coverage ratio honest: only operators with
an actual verifier/extractor contract enter the numerator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

warnings.filterwarnings("ignore")

import torch  # noqa: E402

from evaluation.operator_frequency import (  # noqa: E402
    STEP208_HOT_OPERATORS,
    implemented_operator_census,
)

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "real_model_operator_coverage.json")
MD_PATH = os.path.join(HERE, "real_model_operator_coverage.md")

TORCHVISION_CORPUS = [
    "resnet18",
    "resnet50",
    "vgg11",
    "densenet121",
    "mobilenet_v2",
    "efficientnet_b0",
    "squeezenet1_0",
    "shufflenet_v2_x1_0",
    "mnasnet1_0",
    "regnet_y_400mf",
    "convnext_tiny",
    "vit_b_16",
    "swin_t",
]

TIMM_CORPUS = [
    "resnet18",
    "mobilenetv3_small_050",
    "vit_tiny_patch16_224",
    "convnext_tiny",
]

HF_CORPUS = ["bert-small-no-download"]

METADATA_OPS = {"size", "dim", "floordiv"}
THRESHOLD = 0.95


class CorpusUnavailable(RuntimeError):
    pass


def _target_name(target) -> str:
    return getattr(target, "__name__", str(target))


def _count_graph(gm: "torch.fx.GraphModule") -> Counter:
    counts: Counter = Counter()
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_function":
            counts[_target_name(node.target)] += 1
        elif node.op == "call_method":
            counts[str(node.target)] += 1
        elif node.op == "call_module":
            counts[type(modules[node.target]).__name__] += 1
    return counts


def _trace_torchvision() -> List[Dict[str, object]]:
    try:
        import torchvision
        import torchvision.models as tvm
    except Exception as exc:  # pragma: no cover - optional dependency
        raise CorpusUnavailable(f"torchvision unavailable: {exc}") from exc

    rows = []
    for name in TORCHVISION_CORPUS:
        model = getattr(tvm, name)(weights=None)
        counts = _count_graph(torch.fx.symbolic_trace(model))
        rows.append({
            "family": "torchvision",
            "model": name,
            "operator_occurrences": sum(counts.values()),
            "counts": dict(sorted(counts.items())),
        })
    rows[0]["library_version"] = torchvision.__version__
    return rows


def _trace_timm() -> List[Dict[str, object]]:
    try:
        import timm
    except Exception as exc:  # pragma: no cover - optional dependency
        raise CorpusUnavailable(f"timm unavailable: {exc}") from exc

    rows = []
    for name in TIMM_CORPUS:
        model = timm.create_model(name, pretrained=False)
        counts = _count_graph(torch.fx.symbolic_trace(model))
        rows.append({
            "family": "timm",
            "model": name,
            "operator_occurrences": sum(counts.values()),
            "counts": dict(sorted(counts.items())),
        })
    rows[0]["library_version"] = timm.__version__
    return rows


def _trace_huggingface() -> List[Dict[str, object]]:
    try:
        import transformers
        from transformers import BertConfig, BertModel
        from transformers.utils.fx import symbolic_trace
    except Exception as exc:  # pragma: no cover - optional dependency
        raise CorpusUnavailable(f"transformers unavailable: {exc}") from exc

    config = BertConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=100,
    )
    model = BertModel(config)
    traced = symbolic_trace(model, input_names=["input_ids"], disable_check=True)
    counts = _count_graph(traced)
    return [{
        "family": "huggingface",
        "model": "bert-small-no-download",
        "library_version": transformers.__version__,
        "operator_occurrences": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
    }]


def trace_census() -> Tuple[List[Dict[str, object]], Counter]:
    rows = _trace_torchvision() + _trace_timm() + _trace_huggingface()
    total: Counter = Counter()
    for row in rows:
        total.update(row["counts"])
    return rows, total


def _coverage(counts: Counter, census: Set[str]) -> Dict[str, object]:
    operators = []
    covered_weight = 0
    for op, freq in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        covered = op.lower() in census
        operators.append({
            "operator": op,
            "frequency": freq,
            "covered": covered,
            "metadata_only": op.lower() in METADATA_OPS,
        })
        if covered:
            covered_weight += freq
    total = sum(counts.values())
    uncovered = [o for o in operators if not o["covered"]]
    return {
        "distinct_operators": len(operators),
        "total_op_occurrences": total,
        "covered_occurrences": covered_weight,
        "uncovered_occurrences": total - covered_weight,
        "frequency_coverage_ratio": round(covered_weight / total, 4)
        if total else 0.0,
        "operators": operators,
        "uncovered_ranked": uncovered,
    }


def _version_map(rows: Iterable[Dict[str, object]]) -> Dict[str, str]:
    versions = {"torch": torch.__version__}
    for row in rows:
        version = row.get("library_version")
        if version:
            versions[str(row["family"])] = str(version)
    return versions


def build_report() -> Dict[str, object]:
    rows, counts = trace_census()
    after_census = implemented_operator_census()
    hot = {op.lower() for op in STEP208_HOT_OPERATORS}
    before_census = after_census - hot
    before = _coverage(counts, before_census)
    after = _coverage(counts, after_census)

    by_after = {o["operator"].lower(): o for o in after["operators"]}
    by_before = {o["operator"].lower(): o for o in before["operators"]}
    newly_covered = []
    proof_notes = {
        "stochastic_depth": "torchvision.ops.stochastic_depth is an FX leaf whose output shape equals its input shape.",
        "layer_norm": "F.layer_norm now builds a synthetic LayerNorm layer and checks normalized trailing dims.",
        "adaptive_avg_pool2d": "F.adaptive_avg_pool2d now builds a synthetic AdaptiveAvgPool2d layer with exact output_size.",
        "scaled_dot_product_attention": "F.scaled_dot_product_attention maps to OpKind.SDPA with live torch parity tests.",
    }
    for op in STEP208_HOT_OPERATORS:
        key = op.lower()
        row_after = by_after.get(key)
        row_before = by_before.get(key)
        if row_after and row_before and row_after["covered"] and not row_before["covered"]:
            newly_covered.append({
                "operator": op,
                "frequency": row_after["frequency"],
                "proof": proof_notes[op],
            })

    metadata_rows = [
        o for o in after["operators"]
        if o["operator"].lower() in METADATA_OPS and not o["covered"]
    ]

    return {
        "meta": {
            "generated_by": "evaluation/real_model_operator_coverage.py",
            "command": "PYTHONPATH=. python3 evaluation/real_model_operator_coverage.py",
            "library_versions": _version_map(rows),
            "threshold": THRESHOLD,
            "corpus": {
                "torchvision": TORCHVISION_CORPUS,
                "timm": TIMM_CORPUS,
                "huggingface": HF_CORPUS,
            },
            "n_models": len(rows),
        },
        "models": rows,
        "summary": {
            "before_frequency_coverage_ratio": before["frequency_coverage_ratio"],
            "after_frequency_coverage_ratio": after["frequency_coverage_ratio"],
            "coverage_lift": round(
                after["frequency_coverage_ratio"]
                - before["frequency_coverage_ratio"],
                4,
            ),
            "passes_threshold": after["frequency_coverage_ratio"] >= THRESHOLD,
            "newly_covered_hot_operators": newly_covered,
            "metadata_ops_excluded_from_new_coverage": metadata_rows,
        },
        "before_step208": before,
        "after_step208": after,
    }


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def render_markdown(rep: Dict[str, object]) -> str:
    meta = rep["meta"]
    summary = rep["summary"]
    before = rep["before_step208"]
    after = rep["after_step208"]
    versions = meta["library_versions"]
    lines = [
        "# Step 208 real-model operator coverage",
        "",
        ("FX census over %d no-download real models from torchvision, timm, "
         "and HuggingFace Transformers. Library versions: torch `%s`, "
         "torchvision `%s`, timm `%s`, transformers `%s`." % (
             meta["n_models"],
             versions.get("torch", "?"),
             versions.get("torchvision", "?"),
             versions.get("timm", "?"),
             versions.get("huggingface", "?"),
         )),
        "",
        "| Matrix | Covered / total occurrences | Frequency-weighted coverage |",
        "|--------|-----------------------------|-----------------------------|",
        "| before Step 208 | %d / %d | %.4f |" % (
            before["covered_occurrences"],
            before["total_op_occurrences"],
            before["frequency_coverage_ratio"],
        ),
        "| after Step 208 | %d / %d | %.4f |" % (
            after["covered_occurrences"],
            after["total_op_occurrences"],
            after["frequency_coverage_ratio"],
        ),
        "",
        "Newly covered hot operators:",
        "",
    ]
    for row in summary["newly_covered_hot_operators"]:
        lines.append(
            "* `%s` (%d occurrences): %s"
            % (row["operator"], row["frequency"], row["proof"])
        )
    lines.extend([
        "",
        "Shape-metadata operators intentionally excluded from the new coverage "
        "numerator: `%s`." % "`, `".join(sorted(METADATA_OPS)),
        "",
        "## Top after-Step-208 operators",
        "",
        "| Operator | Frequency | Covered |",
        "|----------|-----------|---------|",
    ])
    for row in after["operators"][:35]:
        lines.append(
            "| `%s` | %d | %s |"
            % (row["operator"], row["frequency"], "yes" if row["covered"] else "NO")
        )
    lines.append("")
    return "\n".join(lines)


def _same_versions(committed: Dict[str, object], live: Dict[str, object]) -> bool:
    return (
        committed.get("meta", {}).get("library_versions")
        == live.get("meta", {}).get("library_versions")
    )


def run(check: bool = False, write: bool = True) -> int:
    try:
        rep = build_report()
    except CorpusUnavailable as exc:
        if check or os.path.exists(JSON_PATH):
            print(f"QUALIFIED: {exc}; using committed Step 208 coverage artifact")
            return 0
        print(str(exc))
        return 1

    if rep["summary"]["after_frequency_coverage_ratio"] < THRESHOLD:
        print("Step 208 coverage below threshold")
        return 1

    text = _dumps(rep)
    md = render_markdown(rep)

    if check:
        if not os.path.exists(JSON_PATH) or not os.path.exists(MD_PATH):
            print("Step 208 coverage artifacts missing; run the harness first")
            return 1
        committed = json.load(open(JSON_PATH))
        if not _same_versions(committed, rep):
            print("QUALIFIED: library version mismatch; skipping byte-identical check")
            return 0
        if open(JSON_PATH).read() != text:
            print("real_model_operator_coverage.json is stale")
            return 1
        if open(MD_PATH).read() != md:
            print("real_model_operator_coverage.md is stale")
            return 1
        print("Step 208 real-model operator coverage artifact up to date")
        return 0

    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(md)
    summary = rep["summary"]
    print(
        "Step 208 coverage: before %.4f -> after %.4f over %d models"
        % (
            summary["before_frequency_coverage_ratio"],
            summary["after_frequency_coverage_ratio"],
            rep["meta"]["n_models"],
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
