"""Tests for Step 252 broad same-case head-to-head benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproducibility import head_to_head_step252 as h2h  # noqa: E402


def _artifact():
    return json.loads(h2h.OUT_JSON.read_text())


def test_artifact_check_mode_accepts_current_environment():
    assert h2h.run(check=True) == 0


def test_all_requested_tool_families_are_present_on_same_cases():
    data = _artifact()
    assert data["tools"] == h2h.TOOLS
    assert data["corpus"]["n_cases"] == 20
    assert data["corpus"]["n_buggy"] == 10
    assert data["corpus"]["n_clean"] == 10

    for row in data["per_case"]:
        assert set(row["predictions"]) == set(data["tools"])
        assert row["name"] in h2h.ENTRY_SPECS
        assert row["entry"]["kind"] in {"function", "module"}


def test_frozen_llm_artifact_is_name_and_label_aligned():
    data = _artifact()
    assert data["llm_artifact"]["sha256"] == h2h._sha256(h2h.LLM_OUTPUT)

    llm = json.loads(h2h.LLM_OUTPUT.read_text())
    by_name = {row["name"]: row for row in llm["benchmarks"]}
    assert sorted(by_name) == sorted(row["name"] for row in data["per_case"])
    for row in data["per_case"]:
        assert bool(by_name[row["name"]]["expect_bug"]) == (row["label"] == h2h.BUGGY)
        pred = row["predictions"]["llm_gpt4_1_nano_frozen"]["pred"]
        assert pred == (h2h.BUGGY if by_name[row["name"]]["llm_found_bug"] else h2h.CLEAN)


def test_tensorguard_is_perfect_on_the_twenty_case_benchmark():
    c = _artifact()["metrics"]["tensorguard_unified"]["all"]
    assert (c["TP"], c["FP"], c["TN"], c["FN"], c["NA"]) == (10, 0, 10, 0, 0)
    assert c["precision"] == 1.0
    assert c["recall"] == 1.0


def test_runtime_smoke_misses_only_the_two_latent_branch_bugs():
    data = _artifact()
    misses = sorted(
        row["name"]
        for row in data["per_case"]
        if row["label"] == h2h.BUGGY
        and row["predictions"]["runtime_forward_smoke"]["pred"] != h2h.BUGGY
    )
    assert misses == ["cross_null_matmul", "null_optional_tensor_shape"]
    c = data["metrics"]["runtime_forward_smoke"]["all"]
    assert (c["TP"], c["FP"], c["TN"], c["FN"], c["NA"]) == (8, 0, 10, 2, 0)


def test_pyright_catches_optional_none_bugs_but_not_tensor_shape_bugs():
    data = _artifact()
    flagged = sorted(
        row["name"]
        for row in data["per_case"]
        if row["predictions"]["pyright"]["pred"] == h2h.BUGGY
    )
    assert flagged == ["cross_null_matmul", "null_optional_tensor_shape"]
    c = data["metrics"]["pyright"]["all"]
    assert (c["TP"], c["FP"], c["TN"], c["FN"], c["NA"]) == (2, 0, 10, 8, 0)


def test_export_guard_baseline_is_complete_on_module_subset_only():
    data = _artifact()
    module_cases = [
        row for row in data["per_case"] if row["entry"]["kind"] == "module"
    ]
    function_cases = [
        row for row in data["per_case"] if row["entry"]["kind"] == "function"
    ]
    assert len(module_cases) == 6
    assert all(
        row["predictions"]["torch_export_guards"]["pred"] != h2h.NA
        for row in module_cases
    )
    assert all(
        row["predictions"]["torch_export_guards"]["pred"] == h2h.NA
        for row in function_cases
    )
    c = data["metrics"]["torch_export_guards"]["module_subset"]
    assert (c["TP"], c["FP"], c["TN"], c["FN"], c["NA"]) == (3, 0, 3, 0, 0)


def test_runtime_annotation_tools_record_explicit_environment_qualification():
    data = _artifact()
    if not data["environment"]["jaxtyping_typeguard_compatible"]:
        assert data["metrics"]["jaxtyping_runtime"]["all"]["NA"] == 20
        assert {
            row["predictions"]["jaxtyping_runtime"]["detail"]
            for row in data["per_case"]
        } == {"jaxtyping_typeguard_incompatible"}
    if data["environment"]["torchtyping"] is None:
        assert data["metrics"]["torchtyping_runtime"]["all"]["NA"] == 20
        assert {
            row["predictions"]["torchtyping_runtime"]["detail"]
            for row in data["per_case"]
        } == {"torchtyping_unavailable"}


def test_artifact_has_no_volatile_result_fields():
    volatile = ("elapsed", "timestamp", "wall", "duration", "_ms", "seconds")

    def walk_keys(obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                yield key
                yield from walk_keys(val)
        elif isinstance(obj, list):
            for val in obj:
                yield from walk_keys(val)

    for key in walk_keys(_artifact()):
        assert not any(tok in str(key).lower() for tok in volatile), key
