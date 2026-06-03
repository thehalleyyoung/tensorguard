"""Step 256 -- cost/latency Pareto curves for verification quality.

This harness synthesizes three previously separate performance views into one
reviewer-facing trade-off artifact:

* model size: a source-level sweep of stacked ``nn.Linear`` models;
* operator coverage: AST-counted operator occurrences cross-referenced against
  TensorGuard's implemented operator census;
* solver budget: ``max_cegar_iterations`` swept over a budget-sensitive
  infeasible-contract witness plus clean and abstaining controls;
* abstention: sound-mode ``UNKNOWN`` rates and reasons.

The committed ``pareto_curves.json/.md`` files are byte-deterministic: they
contain only structural work, coverage, verdicts, CEGAR iteration counts, and the
non-dominated budget frontier.  Hardware-dependent latency is kept in a separate
volatile companion, ``pareto_latency.json``, where every point is normalized by
an anchor verification measured in the same run.  No raw seconds are committed.
"""

from __future__ import annotations

import ast
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.operator_frequency import implemented_operator_census  # noqa: E402
from src.api import BugCategory, verify_architecture  # noqa: E402

OUT_JSON = REPO / "evaluation" / "pareto_curves.json"
OUT_MD = REPO / "evaluation" / "pareto_curves.md"
OUT_LATENCY = REPO / "evaluation" / "pareto_latency.json"

SOUNDNESS_MODE = "sound"
SOLVER_BUDGETS = (0, 1, 3)
STACK_DEPTHS = (1, 4, 8, 16, 32)
WIDTH = 32
LATENCY_REPEATS = 3
ANCHOR_MODEL = "stack_1"
ANCHOR_BUDGET = 0


def _stack_source(depth: int, width: int = WIDTH) -> str:
    lines = [
        "import torch.nn as nn",
        "class M(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(depth):
        lines.append(f"        self.l{i} = nn.Linear({width}, {width})")
    lines.append("    def forward(self, x):")
    for i in range(depth):
        lines.append(f"        x = self.l{i}(x)")
        if i % 4 == 3:
            lines.append("        x = nn.functional.relu(x)")
    lines.append("        return x")
    return "\n".join(lines) + "\n"


_CEGAR_CONFLICT = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(768, 10)
        self.b = nn.Linear(512, 10)

    def forward(self, x):
        return self.a(x) + self.b(x)
"""


_HEURISTIC_LSTSQ = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        y = self.fc(x)
        _ = torch.linalg.lstsq(y, y).solution
        return y
"""


_DATA_DEPENDENT_ITEM = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        if x.sum().item() > 0:
            x = self.fc(x)
        return x
"""


def _params_for_stack(depth: int, width: int = WIDTH) -> int:
    return depth * (width * width + width)


def corpus() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for depth in STACK_DEPTHS:
        rows.append({
            "name": f"stack_{depth}",
            "family": "supported_stack",
            "depth": depth,
            "params": _params_for_stack(depth),
            "source": _stack_source(depth),
            "input_shapes": {"x": (2, WIDTH)},
        })
    rows.extend([
        {
            "name": "cegar_conflict",
            "family": "budget_sensitive_bug",
            "depth": 2,
            "params": 768 * 10 + 10 + 512 * 10 + 10,
            "source": _CEGAR_CONFLICT,
            "input_shapes": {"x": ("b", "f")},
        },
        {
            "name": "heuristic_lstsq",
            "family": "heuristic_operator_abstention",
            "depth": 1,
            "params": 8 * 8 + 8,
            "source": _HEURISTIC_LSTSQ,
            "input_shapes": {"x": (4, 8)},
        },
        {
            "name": "data_dependent_item",
            "family": "fragment_abstention",
            "depth": 1,
            "params": 8 * 8 + 8,
            "source": _DATA_DEPENDENT_ITEM,
            "input_shapes": {"x": (4, 8)},
        },
    ])
    return rows


def _qualified_name(node: ast.AST) -> str:
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def operator_counts(source: str) -> Dict[str, int]:
    """Count model-relevant operator occurrences from source syntax.

    Layer constructor calls count as the corresponding module operator.  Calls to
    ``self.layer(...)`` are skipped because the constructor already accounted for
    the layer occurrence.  Function/method calls such as ``nn.functional.relu``,
    ``torch.linalg.lstsq``, ``x.sum()``, and ``x.item()`` count by final
    attribute name.  Binary addition counts as ``add``.
    """

    tree = ast.parse(source)
    counts: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            name = func.attr
            if name.startswith("__"):
                continue
            qname = _qualified_name(func)
            if qname.startswith("self."):
                continue
            counts[name] += 1
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            counts["add"] += 1
    return dict(sorted(counts.items()))


def coverage_for_counts(counts: Dict[str, int]) -> Dict[str, object]:
    census = implemented_operator_census()
    total = sum(counts.values())
    covered = sum(freq for op, freq in counts.items() if op.lower() in census)
    uncovered = {
        op: freq for op, freq in counts.items() if op.lower() not in census
    }
    return {
        "operator_counts": counts,
        "operator_occurrences": total,
        "covered_operator_occurrences": covered,
        "uncovered_operator_occurrences": total - covered,
        "operator_coverage_ratio": round(covered / total, 4) if total else 1.0,
        "uncovered_operators": uncovered,
    }


def _has_refined_contract_bug(result) -> bool:
    return any(
        bug.category == BugCategory.CEGAR_REFINED_CONTRACT
        for bug in result.bugs
    )


def _verification_row(case: Dict[str, object], budget: int) -> Dict[str, object]:
    source = str(case["source"])
    coverage = coverage_for_counts(operator_counts(source))
    result = verify_architecture(
        source,
        input_shapes=case["input_shapes"],  # type: ignore[arg-type]
        max_cegar_iterations=budget,
        soundness_mode=SOUNDNESS_MODE,
    )
    verdict = str(result.verdict)
    return {
        "model": case["name"],
        "family": case["family"],
        "depth": case["depth"],
        "params": case["params"],
        "solver_budget": budget,
        "actual_cegar_iterations": int(getattr(result, "_cegar_iterations", 0)),
        "lines_analyzed": int(result.lines_analyzed),
        "functions_analyzed": int(result.functions_analyzed),
        "verdict": verdict,
        "decided": verdict != "UNKNOWN",
        "abstained": verdict == "UNKNOWN",
        "unknown_reasons": sorted(result.unknown_reasons),
        "bug_count": int(result.bug_count),
        "has_refined_contract_bug": _has_refined_contract_bug(result),
        **coverage,
    }


def _summaries(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for budget in SOLVER_BUDGETS:
        group = [row for row in rows if row["solver_budget"] == budget]
        n = len(group)
        abstentions = sum(1 for row in group if row["abstained"])
        refined = sum(1 for row in group if row["has_refined_contract_bug"])
        work_units = sum(
            int(row["lines_analyzed"]) * (1 + int(row["actual_cegar_iterations"]))
            for row in group
        )
        decided_params = [
            int(row["params"]) for row in group if row["decided"]
        ]
        out.append({
            "solver_budget": budget,
            "models": n,
            "safe": sum(1 for row in group if row["verdict"] == "SAFE"),
            "unsafe": sum(1 for row in group if row["verdict"] == "UNSAFE"),
            "unknown": abstentions,
            "abstention_rate": round(abstentions / n, 4),
            "mean_operator_coverage_ratio": round(
                sum(float(row["operator_coverage_ratio"]) for row in group) / n,
                4,
            ),
            "min_operator_coverage_ratio": min(
                float(row["operator_coverage_ratio"]) for row in group
            ),
            "refined_contract_diagnoses": refined,
            "total_actual_cegar_iterations": sum(
                int(row["actual_cegar_iterations"]) for row in group
            ),
            "structural_work_units": work_units,
            "max_model_params": max(int(row["params"]) for row in group),
            "max_decided_model_params": max(decided_params) if decided_params else 0,
        })
    return out


def _dominates(a: Dict[str, object], b: Dict[str, object]) -> bool:
    comparisons = [
        float(a["structural_work_units"]) <= float(b["structural_work_units"]),
        float(a["abstention_rate"]) <= float(b["abstention_rate"]),
        float(a["mean_operator_coverage_ratio"]) >= float(b["mean_operator_coverage_ratio"]),
        int(a["refined_contract_diagnoses"]) >= int(b["refined_contract_diagnoses"]),
    ]
    strict = [
        float(a["structural_work_units"]) < float(b["structural_work_units"]),
        float(a["abstention_rate"]) < float(b["abstention_rate"]),
        float(a["mean_operator_coverage_ratio"]) > float(b["mean_operator_coverage_ratio"]),
        int(a["refined_contract_diagnoses"]) > int(b["refined_contract_diagnoses"]),
    ]
    return all(comparisons) and any(strict)


def pareto_frontier(summaries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    frontier = []
    for row in summaries:
        if not any(_dominates(other, row) for other in summaries if other is not row):
            frontier.append(row)
    return sorted(frontier, key=lambda r: int(r["solver_budget"]))


def measure() -> Dict[str, object]:
    rows = [
        _verification_row(case, budget)
        for budget in SOLVER_BUDGETS
        for case in corpus()
    ]
    summaries = _summaries(rows)
    frontier = pareto_frontier(summaries)
    return {
        "step": 256,
        "generated_by": "evaluation/pareto_curves.py",
        "soundness_mode": SOUNDNESS_MODE,
        "solver_budgets": list(SOLVER_BUDGETS),
        "model_depths": list(STACK_DEPTHS),
        "width": WIDTH,
        "latency_companion": "evaluation/pareto_latency.json",
        "hardware_normalized_latency": {
            "reported_in": "evaluation/pareto_latency.json",
            "normalization": (
                "Each candidate median is divided by an anchor verification "
                "median measured in the same run; raw machine costs are not "
                "committed."
            ),
        },
        "rows": rows,
        "budget_summaries": summaries,
        "pareto_frontier": frontier,
        "invariants": {
            "budget_axis_is_load_bearing": any(
                int(row["solver_budget"]) > 0 and row["has_refined_contract_bug"]
                for row in rows
            ) and not any(
                int(row["solver_budget"]) == 0 and row["has_refined_contract_bug"]
                for row in rows
            ),
            "has_abstention_case": any(row["abstained"] for row in rows),
            "has_uncovered_operator_case": any(
                float(row["operator_coverage_ratio"]) < 1.0 for row in rows
            ),
            "frontier_excludes_dominated_budget": any(
                int(s["solver_budget"]) not in {
                    int(f["solver_budget"]) for f in frontier
                }
                for s in summaries
            ),
        },
    }


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def render_markdown(data: Dict[str, object]) -> str:
    lines = [
        "# Step 256 cost/latency Pareto curves",
        "",
        ("Sound-mode sweep over model size, operator coverage, solver budget, "
         "and abstention. The JSON/Markdown artifact is deterministic; latency "
         "is reported only in the hardware-normalized volatile companion "
         "`evaluation/pareto_latency.json`."),
        "",
        "## Budget frontier",
        "",
        "| budget | work units | mean coverage | abstention rate | refined diagnoses | frontier? |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    frontier_budgets = {
        int(row["solver_budget"]) for row in data["pareto_frontier"]  # type: ignore[index]
    }
    for row in data["budget_summaries"]:  # type: ignore[index]
        budget = int(row["solver_budget"])
        lines.append(
            f"| {budget} | {row['structural_work_units']} "
            f"| {row['mean_operator_coverage_ratio']:.4f} "
            f"| {row['abstention_rate']:.4f} "
            f"| {row['refined_contract_diagnoses']} "
            f"| {budget in frontier_budgets} |"
        )
    lines.extend([
        "",
        "## Model/budget points",
        "",
        "| model | family | budget | params | coverage | verdict | CEGAR iters | abstention reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in data["rows"]:  # type: ignore[index]
        reasons = "; ".join(row["unknown_reasons"]) if row["unknown_reasons"] else "-"
        lines.append(
            f"| `{row['model']}` | {row['family']} | {row['solver_budget']} "
            f"| {row['params']} | {row['operator_coverage_ratio']:.4f} "
            f"| {row['verdict']} | {row['actual_cegar_iterations']} "
            f"| {reasons} |"
        )
    lines.extend([
        "",
        "The non-dominated budget set keeps the zero-refinement point (lowest "
        "structural work) and the first refinement point (contract-level CEGAR "
        "diagnosis), while the higher budget is dominated once convergence has "
        "already been reached.",
        "",
    ])
    return "\n".join(lines)


def _verify_once(model_name: str, budget: int) -> None:
    by_name = {str(case["name"]): case for case in corpus()}
    case = by_name[model_name]
    verify_architecture(
        str(case["source"]),
        input_shapes=case["input_shapes"],  # type: ignore[arg-type]
        max_cegar_iterations=budget,
        soundness_mode=SOUNDNESS_MODE,
    )


def _median_cost(model_name: str, budget: int, repeats: int) -> float:
    _verify_once(model_name, budget)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _verify_once(model_name, budget)
        samples.append(time.perf_counter() - t0)
    return median(samples)


def measure_normalized_latency(
    repeats: int = LATENCY_REPEATS,
    model_names: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
) -> Dict[str, object]:
    names = list(model_names) if model_names is not None else [
        str(case["name"]) for case in corpus()
    ]
    budget_values = list(budgets) if budgets is not None else list(SOLVER_BUDGETS)
    anchor = _median_cost(ANCHOR_MODEL, ANCHOR_BUDGET, repeats)
    if anchor <= 0:
        anchor = 1e-12

    rows = []
    for budget in budget_values:
        for name in names:
            ratio = _median_cost(name, budget, repeats) / anchor
            rows.append({
                "model": name,
                "solver_budget": budget,
                "ratio_to_anchor": round(ratio, 4),
            })
    return {
        "step": 256,
        "generated_by": "evaluation/pareto_curves.py",
        "latency_reporting": "hardware_normalized",
        "normalization": {
            "anchor_model": ANCHOR_MODEL,
            "anchor_solver_budget": ANCHOR_BUDGET,
            "repeats_per_point": repeats,
            "ratio_definition": (
                "candidate median verification cost divided by anchor median "
                "verification cost in the same run"
            ),
            "raw_cost_committed": False,
        },
        "rows": rows,
    }


def gate() -> int:
    data = measure_normalized_latency(
        repeats=1, model_names=(ANCHOR_MODEL, "stack_16"), budgets=(0, 1)
    )
    ratios = [float(row["ratio_to_anchor"]) for row in data["rows"]]  # type: ignore[index]
    if not ratios or any((not math.isfinite(r)) or r <= 0.0 for r in ratios):
        print("pareto latency gate failed: non-positive normalized ratio")
        return 1
    print("pareto latency gate PASS: normalized ratios are positive")
    return 0


def run(check: bool = False, write: bool = True) -> int:
    data = measure()
    js = _dumps(data)
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
            print("pareto_curves: byte-identical")
        return 0 if ok else 1

    if write:
        OUT_JSON.write_text(js)
        OUT_MD.write_text(md)
        OUT_LATENCY.write_text(_dumps(measure_normalized_latency()))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_LATENCY} (volatile, hardware-normalized)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Check deterministic JSON/Markdown artifacts only.")
    ap.add_argument("--gate", action="store_true",
                    help="Run a small live normalized-latency sanity gate.")
    args = ap.parse_args()
    if args.gate:
        return gate()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
