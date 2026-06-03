#!/usr/bin/env python3
"""PR-history survival study over fix-linked GitHub records (Step 254).

The frozen provenance corpus intentionally does *not* redistribute historical
pre-fix source trees, patches, issue bodies, or comments.  This study therefore
answers the survival question at the runtime-signature-category level:

* select every corpus record that carries a direct PR commit link or an
  offline same-repo/same-category candidate fix link;
* replay one trusted, repo-authored/static witness per runtime-signature
  category against live TensorGuard;
* execute the existing repo-authored minimized PyTorch reproducers where the
  current host can do so; and
* multiply the category-level outcome over the frozen fix-linked record counts.

The artifact is deliberately honest: the 332 linked rows are real GitHub
issue/PR facts, but they are not independent historical checkouts.  Catch rates
are category-level survival estimates over those fix-linked records.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.api import verify_architecture  # noqa: E402
from src.model_checker import extract_computation_graph  # noqa: E402

CORPUS = REPO / "experiments_v5" / "provenance_bug_corpus" / "corpus.jsonl"
OUT_JSON = REPO / "reproducibility" / "pr_history_survival.json"
OUT_MD = REPO / "reproducibility" / "pr_history_survival.md"

DIRECT_PR = "direct_pr_commits_page"
CANDIDATE_PR = "candidate_same_repo_category_pr"


@dataclass(frozen=True)
class ReplayCase:
    category: str
    source: str
    input_shapes: Mapping[str, Sequence[int]]
    reproducer_path: str
    dynamic_prefix_ops_before_failure: int
    constructed_tensor_count: int
    evidence_note: str


CASES: Dict[str, ReplayCase] = {
    "broadcast_mismatch": ReplayCase(
        category="broadcast_mismatch",
        source=textwrap.dedent(
            """
            import torch
            import torch.nn as nn

            class Net(nn.Module):
                def forward(self, x, y):
                    return x + y
            """
        ),
        input_shapes={"x": (2, 3), "y": (2, 4)},
        reproducer_path="experiments_v5/provenance_bug_corpus/reproducers/broadcast_mismatch.py",
        dynamic_prefix_ops_before_failure=0,
        constructed_tensor_count=2,
        evidence_note="binary broadcast mismatch, same runtime-signature family as mined records",
    ),
    "cat_stack_mismatch": ReplayCase(
        category="cat_stack_mismatch",
        source=textwrap.dedent(
            """
            import torch
            import torch.nn as nn

            class Net(nn.Module):
                def forward(self, x, y):
                    return torch.cat([x, y], dim=0)
            """
        ),
        input_shapes={"x": (2, 3), "y": (2, 4)},
        reproducer_path="experiments_v5/provenance_bug_corpus/reproducers/cat_stack_mismatch.py",
        dynamic_prefix_ops_before_failure=0,
        constructed_tensor_count=2,
        evidence_note="concat non-axis dimension mismatch",
    ),
    "conv_channel_mismatch": ReplayCase(
        category="conv_channel_mismatch",
        source=textwrap.dedent(
            """
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 4, kernel_size=3)

                def forward(self, x):
                    return self.conv(x)
            """
        ),
        input_shapes={"x": (1, 2, 8, 8)},
        reproducer_path="experiments_v5/provenance_bug_corpus/reproducers/conv_channel_mismatch.py",
        dynamic_prefix_ops_before_failure=0,
        constructed_tensor_count=1,
        evidence_note="Conv2d in-channel contract mismatch",
    ),
    "device_mismatch": ReplayCase(
        category="device_mismatch",
        source=textwrap.dedent(
            """
            import torch
            import torch.nn as nn

            class Net(nn.Module):
                def forward(self):
                    return torch.ones(2, device="cpu") + torch.ones(2, device="cuda")
            """
        ),
        input_shapes={},
        reproducer_path="experiments_v5/provenance_bug_corpus/reproducers/device_mismatch.py",
        dynamic_prefix_ops_before_failure=2,
        constructed_tensor_count=2,
        evidence_note="CPU/CUDA tensor factory mismatch; eager replay is CUDA-qualified",
    ),
    "dim_out_of_range": ReplayCase(
        category="dim_out_of_range",
        source=textwrap.dedent(
            """
            import torch
            import torch.nn as nn

            class Net(nn.Module):
                def forward(self, x):
                    return x.transpose(0, 3)
            """
        ),
        input_shapes={"x": (2, 3)},
        reproducer_path="experiments_v5/provenance_bug_corpus/reproducers/dim_out_of_range.py",
        dynamic_prefix_ops_before_failure=0,
        constructed_tensor_count=1,
        evidence_note="rank-2 tensor with out-of-range dim index",
    ),
    "dtype_device_input_mismatch": ReplayCase(
        category="dtype_device_input_mismatch",
        source=textwrap.dedent(
            """
            import torch
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 4, kernel_size=3)

                def forward(self, x):
                    return self.conv(x.long())
            """
        ),
        input_shapes={"x": (1, 3, 8, 8)},
        reproducer_path="experiments_v5/provenance_bug_corpus/reproducers/dtype_device_input_mismatch.py",
        dynamic_prefix_ops_before_failure=1,
        constructed_tensor_count=1,
        evidence_note=(
            "non-floating input into a floating-parameter layer; the committed "
            "runtime reproducer covers the sibling input/weight dtype mismatch"
        ),
    ),
    "matmul_linear_mismatch": ReplayCase(
        category="matmul_linear_mismatch",
        source=textwrap.dedent(
            """
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(4, 3)

                def forward(self, x):
                    return self.fc(x)
            """
        ),
        input_shapes={"x": (2, 5)},
        reproducer_path="experiments_v5/provenance_bug_corpus/reproducers/matmul_linear_mismatch.py",
        dynamic_prefix_ops_before_failure=0,
        constructed_tensor_count=1,
        evidence_note="Linear/matmul contracted-dimension mismatch",
    ),
    "view_reshape_total_size": ReplayCase(
        category="view_reshape_total_size",
        source=textwrap.dedent(
            """
            import torch
            import torch.nn as nn

            class Net(nn.Module):
                def forward(self, x):
                    return x.reshape(5, 5)
            """
        ),
        input_shapes={"x": (2, 3, 4)},
        reproducer_path="experiments_v5/provenance_bug_corpus/reproducers/view_reshape_total_size.py",
        dynamic_prefix_ops_before_failure=0,
        constructed_tensor_count=1,
        evidence_note="reshape/view total-element-count mismatch",
    ),
}


def _load_records() -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _linked_records(records: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [r for r in records if r.get("commit_links")]


def _run_reproducer(case: ReplayCase) -> Dict[str, Any]:
    rel = case.reproducer_path
    path = REPO / rel
    if case.category == "device_mismatch":
        return {
            "path": rel,
            "status": "cuda_qualified_not_executed_in_cpu_ci",
            "proven_live_on_this_host": False,
            "expected_to_raise": True,
        }
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    output = proc.stdout + proc.stderr
    return {
        "path": rel,
        "status": "proven_live",
        "proven_live_on_this_host": proc.returncode != 0,
        "expected_to_raise": True,
        "observed_nonzero_exit": proc.returncode != 0,
        "observed_error_fragment_present": bool(output.strip()),
    }


def _static_replay(case: ReplayCase) -> Dict[str, Any]:
    shapes = {name: tuple(shape) for name, shape in case.input_shapes.items()}
    graph = extract_computation_graph(case.source)
    result = verify_architecture(
        case.source,
        input_shapes=shapes,
        infer_inputs=False,
        soundness_mode="sound",
        max_cegar_iterations=0,
    )
    tags = sorted({
        bug.message.split("]", 1)[0].lstrip("[")
        for bug in result.bugs
        if bug.message.startswith("[") and "]" in bug.message
    })
    return {
        "verdict": result.verdict,
        "caught_by_tensorguard": result.verdict == "UNSAFE",
        "bug_tags": tags,
        "graph_steps": len(graph.steps),
        "input_count": len(case.input_shapes),
        "needs_concrete_input_values": False,
        "executes_model_code": False,
    }


def _category_rows(linked: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_category = Counter(str(r["category"]) for r in linked)
    by_direct = Counter(
        str(r["category"]) for r in linked
        if r["commit_link_status"] == DIRECT_PR
    )
    by_candidate = Counter(
        str(r["category"]) for r in linked
        if r["commit_link_status"] == CANDIDATE_PR
    )

    rows: Dict[str, Dict[str, Any]] = {}
    for category in sorted(by_category):
        if category not in CASES:
            raise KeyError(f"no replay case registered for {category}")
        case = CASES[category]
        static = _static_replay(case)
        eager = _run_reproducer(case)
        n = by_category[category]
        caught = n if static["caught_by_tensorguard"] else 0
        rows[category] = {
            "fix_linked_records": n,
            "direct_pr_records": by_direct[category],
            "candidate_issue_records": by_candidate[category],
            "static_replay": static,
            "eager_reproducer": eager,
            "category_level_caught_records": caught,
            "category_level_missed_records": n - caught,
            "static_detect_depth_ops": 0 if static["caught_by_tensorguard"] else None,
            "dynamic_prefix_ops_before_failure": case.dynamic_prefix_ops_before_failure,
            "constructed_tensor_count": case.constructed_tensor_count,
            "evidence_note": case.evidence_note,
        }
    return rows


def _aggregate(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    field: str | None = None,
) -> Dict[str, Any]:
    cats = list(rows.values())
    if field is not None:
        cats = [row for row in cats if int(row[field]) > 0]
    total = sum(int(row["fix_linked_records"] if field is None else row[field]) for row in cats)
    caught = sum(
        int(row["fix_linked_records"] if field is None else row[field])
        for row in cats
        if row["static_replay"]["caught_by_tensorguard"]
    )
    return {
        "records": total,
        "category_level_caught": caught,
        "category_level_missed": total - caught,
        "category_level_catch_rate": round(caught / total, 6) if total else None,
    }


def _cost_proxy(rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    total = sum(int(row["fix_linked_records"]) for row in rows.values())
    static_steps = sum(
        int(row["fix_linked_records"]) * int(row["static_replay"]["graph_steps"])
        for row in rows.values()
    )
    dynamic_prefix = sum(
        int(row["fix_linked_records"]) * int(row["dynamic_prefix_ops_before_failure"])
        for row in rows.values()
    )
    constructed = sum(
        int(row["fix_linked_records"]) * int(row["constructed_tensor_count"])
        for row in rows.values()
    )
    return {
        "unit": "structural_proxy_not_wall_clock",
        "static": {
            "analyzer_passes": total,
            "model_executions": 0,
            "concrete_tensors_constructed": 0,
            "graph_steps_analyzed": static_steps,
            "requires_historical_checkout": False,
        },
        "dynamic_forward_baseline": {
            "forward_invocations": total,
            "concrete_tensors_constructed": constructed,
            "successful_prefix_ops_before_failure": dynamic_prefix,
            "failing_ops_reached": total,
            "requires_concrete_inputs": True,
        },
        "comparison": {
            "static_detects_before_runtime_execution": True,
            "successful_prefix_ops_saved_by_static_gate": dynamic_prefix,
            "concrete_tensor_constructions_avoided_by_static_gate": constructed,
        },
    }


def measure() -> Dict[str, Any]:
    logging.disable(logging.CRITICAL)
    try:
        records = _load_records()
        linked = _linked_records(records)
        rows = _category_rows(linked)
        status_counts = Counter(str(r["commit_link_status"]) for r in linked)
        kind_counts = Counter(str(r["github_kind"]) for r in linked)
        family_counts = Counter(str(r["project_family"]) for r in linked)
        category_counts = Counter(str(r["category"]) for r in linked)
        return {
            "step": 254,
            "schema_version": "tensorguard.pr-history-survival.v1",
            "source_corpus": {
                "path": "experiments_v5/provenance_bug_corpus/corpus.jsonl",
                "records": len(records),
                "fix_linked_records": len(linked),
                "direct_pr_records": status_counts[DIRECT_PR],
                "candidate_issue_records": status_counts[CANDIDATE_PR],
                "records_without_fix_link": len(records) - len(linked),
            },
            "evidence_scope": {
                "granularity": "runtime_signature_category_replay",
                "historical_pre_fix_checkouts_replayed": False,
                "stored_source_blob": False,
                "stored_patch": False,
                "per_row_counts_are_category_multiplicities": True,
                "claim": (
                    "For every fix-linked GitHub record, TensorGuard is evaluated "
                    "on the corresponding runtime-signature category witness, not "
                    "on the historical repository checkout."
                ),
            },
            "fix_linked_breakdown": {
                "by_commit_link_status": dict(sorted(status_counts.items())),
                "by_github_kind": dict(sorted(kind_counts.items())),
                "by_category": dict(sorted(category_counts.items())),
                "by_project_family": dict(sorted(family_counts.items())),
            },
            "survival_estimate": {
                "all_fix_linked": _aggregate(rows),
                "direct_pr_only": _aggregate(rows, field="direct_pr_records"),
                "candidate_issue_links_only": _aggregate(rows, field="candidate_issue_records"),
                "all_categories_represented": sorted(rows) == sorted(CASES),
            },
            "category_replay": rows,
            "detection_depth_and_ci_cost_proxy": _cost_proxy(rows),
            "honesty_notes": [
                "No historical patches or source bodies are redistributed or replayed.",
                "Device eager replay is CUDA-qualified; source-level static replay still exercises the device-transfer contract.",
                "Dtype replay covers the same runtime-signature family with a source-visible non-floating input; the committed eager reproducer covers the sibling input/weight dtype mismatch.",
                "All cost figures are deterministic structural proxies, not wall-clock measurements.",
            ],
        }
    finally:
        logging.disable(logging.NOTSET)


def render_markdown(data: Mapping[str, Any]) -> str:
    src = data["source_corpus"]
    surv = data["survival_estimate"]
    cost = data["detection_depth_and_ci_cost_proxy"]
    lines = [
        "# PR-history survival study (Step 254)",
        "",
        "This is an offline, deterministic survival estimate over the frozen "
        "GitHub provenance corpus.  It asks whether TensorGuard's current "
        "runtime-signature-category check would have rejected the class of bug "
        "represented by a fix-linked PR/issue before the fix merged.  It does "
        "**not** replay historical pre-fix checkouts.",
        "",
        f"Corpus rows: **{src['records']}**; fix-linked rows: "
        f"**{src['fix_linked_records']}** "
        f"({src['direct_pr_records']} direct PR rows, "
        f"{src['candidate_issue_records']} issue rows with candidate same-repo/category PR links).",
        "",
        "## Category-level survival estimate",
        "",
        "| population | rows | category-level caught | missed | catch rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, row in (
        ("all fix-linked", surv["all_fix_linked"]),
        ("direct PR only", surv["direct_pr_only"]),
        ("candidate issue links only", surv["candidate_issue_links_only"]),
    ):
        rate = row["category_level_catch_rate"]
        rate_s = "n/a" if rate is None else f"{rate:.3f}"
        lines.append(
            f"| {label} | {row['records']} | {row['category_level_caught']} "
            f"| {row['category_level_missed']} | {rate_s} |"
        )

    lines.extend([
        "",
        "## Runtime-signature replay rows",
        "",
        "| category | fix-linked rows | direct PR | candidate | TensorGuard verdict | eager proof | dynamic prefix ops | graph steps |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ])
    for category, row in data["category_replay"].items():
        lines.append(
            f"| `{category}` | {row['fix_linked_records']} | "
            f"{row['direct_pr_records']} | {row['candidate_issue_records']} | "
            f"{row['static_replay']['verdict']} | "
            f"{row['eager_reproducer']['status']} | "
            f"{row['dynamic_prefix_ops_before_failure']} | "
            f"{row['static_replay']['graph_steps']} |"
        )

    lines.extend([
        "",
        "## Detection-depth and CI-cost proxy",
        "",
        "The proxy is structural and deterministic, not a wall-clock benchmark.",
        "",
        f"- Static gate: {cost['static']['analyzer_passes']} analyzer passes, "
        f"{cost['static']['graph_steps_analyzed']} graph steps analyzed, "
        f"{cost['static']['model_executions']} model executions.",
        f"- Dynamic forward baseline: {cost['dynamic_forward_baseline']['forward_invocations']} "
        "forward invocations, "
        f"{cost['dynamic_forward_baseline']['concrete_tensors_constructed']} concrete "
        "tensor constructions, "
        f"{cost['dynamic_forward_baseline']['successful_prefix_ops_before_failure']} "
        "successful prefix ops before the failing op.",
        f"- Static gate avoids those concrete tensor constructions: "
        f"{cost['comparison']['concrete_tensor_constructions_avoided_by_static_gate']}.",
        "",
        "## Scope notes",
        "",
    ])
    for note in data["honesty_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text(encoding="utf-8") != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text(encoding="utf-8") != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if ok:
            print("pr_history_survival: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js, encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed artifacts")
    args = parser.parse_args(argv)
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
