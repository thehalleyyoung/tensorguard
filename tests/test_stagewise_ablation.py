"""Tests for the Step 253 stage-wise ablation harness."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

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


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module("reproducibility.stagewise_ablation")


@pytest.fixture(scope="module")
def data(mod):
    return mod.measure()


def test_no_volatile_keys(data):
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(token in low for token in _VOLATILE), f"volatile key: {key}"


def test_extraction_stage_uses_real_graph_and_catches_bug(data):
    extraction = data["extraction"]
    assert extraction["without_extraction_analyzable_modules"] == 0
    assert extraction["with_extraction_layers"] >= 2
    assert extraction["with_extraction_steps"] >= 2
    assert extraction["with_extraction_inputs"] == ["x"]
    assert extraction["caught_after_extraction"] is True
    assert "SHAPE-INCOMPATIBLE" in extraction["bug_tags"]


def test_each_abstract_domain_ablation_is_load_bearing_or_diagnostic(data):
    domains = data["abstract_domains"]
    assert domains["each_verification_domain_load_bearing"] is True
    assert domains["phase_is_diagnostic_only"] is True
    for domain in domains["verification_domains"]:
        row = domains["per_domain"][domain]
        assert row["full_caught"] is True, domain
        assert row["ablated_caught"] is False, domain
        assert row["delta"] == 1, domain
    assert domains["per_domain"]["phase"]["full_caught"] is False
    assert domains["per_domain"]["phase"]["ablated_caught"] is False


def test_every_registered_cross_domain_reduction_has_witness(data):
    from src.domains.product import Reduction

    reductions = data["cross_domain_reductions"]
    registered = sorted(cls.__name__ for cls in Reduction.__subclasses__())
    assert reductions["registered_class_names"] == registered
    assert reductions["all_registered_reductions_exercised"] is True
    assert reductions["all_reductions_are_refinements"] is True
    for cls_name in registered:
        row = reductions["per_reduction"][cls_name]
        assert row["changed"] is True, cls_name
        assert row["changed_components"], cls_name
        assert row["after_refines_before"] is True, cls_name


def test_cegar_stage_shows_refinement_knee_without_false_alarms(data):
    cegar = data["cegar"]
    assert cegar["recall_is_depth_invariant_full"] is True
    assert cegar["zero_false_alarms_all_depths"] is True
    assert cegar["precision_rises_then_plateaus"] is True
    assert cegar["work_saturates_at_convergence"] is True
    assert cegar["refined_diagnoses_at_depth_0"] == 0
    assert cegar["refined_diagnoses_at_knee"] == cegar["n_conflict_cases"]


def test_stub_stage_is_load_bearing(data):
    stubs = data["stubs"]
    assert stubs["without_stub_layer_kind"] != "STUB"
    assert stubs["with_stub_layer_kind"] == "STUB"
    assert stubs["valid_model_safe_without_stub"] is False
    assert stubs["valid_model_safe_with_stub"] is True
    assert stubs["bad_head_caught_with_stub"] is True
    assert stubs["bad_input_caught_with_stub"] is True
    assert stubs["stub_stage_load_bearing"] is True


def test_sound_mode_separates_proof_backed_and_heuristic_rules(data):
    proof = data["proof_rules"]
    assert proof["proof_backed_safe_without_abstention"] is True
    assert proof["heuristic_abstains_in_sound_mode"] is True
    assert proof["proof_backed_case"]["proof_status"] == "lean_theorem"
    assert proof["proof_backed_case"]["verdict"] == "SAFE"
    assert proof["heuristic_case"]["proof_status"] == "heuristic"
    assert proof["heuristic_case"]["verdict"] == "UNKNOWN"
    assert any(
        "heuristic-tagged operator" in reason
        for reason in proof["heuristic_case"]["unknown_reasons"]
    )
    assert proof["heuristic_rows_all_heuristic_footprints"] is True


def test_artifact_byte_deterministic(mod):
    a = mod.measure()
    b = mod.measure()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert mod.run(check=True) == 0
