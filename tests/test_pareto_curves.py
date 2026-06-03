"""Step 256 -- cost/latency Pareto curves."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation import pareto_curves as pc  # noqa: E402

_VOLATILE = (
    "time", "elapsed", "timestamp", "wall", "clock",
    "_ms", "seconds", "duration", "date",
)


def _walk_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_keys(value)


def _committed():
    return json.loads(pc.OUT_JSON.read_text())


def test_deterministic_artifact_has_no_wall_clock_fields():
    data = _committed()
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(token in low for token in _VOLATILE), f"volatile key: {key}"


def test_committed_pareto_artifact_is_byte_stable():
    data = pc.measure()
    assert json.dumps(data, indent=2, sort_keys=True) + "\n" == pc.OUT_JSON.read_text()
    assert pc.render_markdown(data) == pc.OUT_MD.read_text()
    assert pc.run(check=True) == 0


def test_solver_budget_axis_is_load_bearing_on_real_cegar_witness():
    rows = [
        row for row in _committed()["rows"]
        if row["model"] == "cegar_conflict"
    ]
    by_budget = {row["solver_budget"]: row for row in rows}
    assert by_budget[0]["actual_cegar_iterations"] == 0
    assert by_budget[0]["has_refined_contract_bug"] is False
    assert by_budget[1]["actual_cegar_iterations"] > 0
    assert by_budget[1]["has_refined_contract_bug"] is True
    assert _committed()["invariants"]["budget_axis_is_load_bearing"] is True


def test_abstention_and_operator_coverage_axes_are_exercised():
    data = _committed()
    assert data["invariants"]["has_abstention_case"] is True
    assert data["invariants"]["has_uncovered_operator_case"] is True
    unknown = [row for row in data["rows"] if row["verdict"] == "UNKNOWN"]
    assert unknown
    assert any(row["unknown_reasons"] for row in unknown)

    stacks = [
        row for row in data["rows"]
        if row["family"] == "supported_stack" and row["solver_budget"] == 0
    ]
    assert stacks
    assert all(row["operator_coverage_ratio"] == 1.0 for row in stacks)
    assert all(row["verdict"] == "SAFE" for row in stacks)


def test_pareto_frontier_is_non_dominated_and_excludes_saturated_budget():
    data = _committed()
    summaries = data["budget_summaries"]
    frontier = data["pareto_frontier"]
    assert frontier
    for point in frontier:
        assert not any(
            pc._dominates(other, point)
            for other in summaries
            if other is not point
        )
    assert data["invariants"]["frontier_excludes_dominated_budget"] is True
    frontier_budgets = {row["solver_budget"] for row in frontier}
    assert 3 not in frontier_budgets


def test_latency_companion_is_hardware_normalized_without_raw_costs():
    latency = json.loads(pc.OUT_LATENCY.read_text())
    assert latency["latency_reporting"] == "hardware_normalized"
    assert latency["normalization"]["raw_cost_committed"] is False
    assert latency["normalization"]["anchor_model"] == pc.ANCHOR_MODEL
    assert latency["rows"]
    assert all(row["ratio_to_anchor"] > 0 for row in latency["rows"])
    for key in _walk_keys(latency):
        assert "seconds" not in key.lower()
        assert "_ms" not in key.lower()


def test_live_normalized_latency_probe_is_positive_on_small_subset():
    live = pc.measure_normalized_latency(
        repeats=1, model_names=("stack_1", "stack_4"), budgets=(0,)
    )
    ratios = [row["ratio_to_anchor"] for row in live["rows"]]
    assert len(ratios) == 2
    assert all(ratio > 0 for ratio in ratios)


def test_make_and_reproduce_wiring_present():
    makefile = (REPO / "Makefile").read_text()
    assert "pareto-curves:" in makefile
    assert "pareto-curves-gate:" in makefile

    from reproducibility import reproduce_all as ra

    assert "evaluation/pareto_curves.json" in ra.GENERATED_DETERMINISTIC
    assert "evaluation/pareto_curves.md" in ra.GENERATED_DETERMINISTIC
    assert "evaluation/pareto_latency.json" in ra.VOLATILE_REGENERATED
