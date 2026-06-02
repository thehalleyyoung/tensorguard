"""Step 35 -- tests for the published operator-coverage floor and release gate."""

from __future__ import annotations

import copy
import json
import os

from evaluation import operator_coverage as OC


def test_summary_publishes_percentage():
    m = OC.build_matrix()
    summ = m["summary"]
    expected = round(100.0 * summ["total_covered"]
                     / summ["total_public_operators"], 2)
    assert summ["overall_coverage_percent"] == expected
    # ratio and percent agree
    assert abs(summ["overall_coverage_percent"]
               - 100.0 * summ["overall_coverage_ratio"]) < 0.5


def test_floor_artifact_exists_and_is_consistent():
    assert os.path.exists(OC.FLOOR_PATH)
    floor = json.load(open(OC.FLOOR_PATH))
    assert "overall_coverage_ratio" in floor
    assert set(floor["namespaces"]) == set(OC.NAMESPACES)
    for v in floor["namespaces"].values():
        assert 0.0 <= v <= 1.0


def test_build_floor_matches_matrix():
    m = OC.build_matrix()
    floor = OC.build_floor(m)
    assert floor["overall_coverage_ratio"] == m["summary"]["overall_coverage_ratio"]
    assert floor["torch_version"] == m["meta"]["torch_version"]
    for ns in OC.NAMESPACES:
        assert floor["namespaces"][ns] == m["namespaces"][ns]["coverage_ratio"]


def test_gate_passes_against_committed_floor():
    # Only meaningful when the local torch matches the floor's; otherwise the
    # gate is QUALIFIED (skipped) and also returns 0.
    assert OC.gate() == 0


def test_gate_detects_regression(tmp_path, monkeypatch):
    m = OC.build_matrix()
    # Fabricate a floor that demands far more coverage than we have.
    inflated = OC.build_floor(m)
    inflated["overall_coverage_ratio"] = min(
        1.0, inflated["overall_coverage_ratio"] + 0.5)
    for ns in OC.NAMESPACES:
        inflated["namespaces"][ns] = min(1.0, inflated["namespaces"][ns] + 0.5)
    p = tmp_path / "floor.json"
    p.write_text(json.dumps(inflated))
    monkeypatch.setattr(OC, "FLOOR_PATH", str(p))
    assert OC.gate() == 1


def test_gate_qualified_on_version_mismatch(tmp_path, monkeypatch):
    m = OC.build_matrix()
    floor = OC.build_floor(m)
    floor["torch_version"] = "0.0.0-not-a-real-version"
    # Even with an impossible floor, a version mismatch must QUALIFY (skip), not fail.
    floor["overall_coverage_ratio"] = 1.0
    p = tmp_path / "floor.json"
    p.write_text(json.dumps(floor))
    monkeypatch.setattr(OC, "FLOOR_PATH", str(p))
    assert OC.gate() == 0


def test_gate_missing_floor_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(OC, "FLOOR_PATH", str(tmp_path / "nope.json"))
    assert OC.gate() == 1


def test_gate_tolerance_absorbs_tiny_slip(tmp_path, monkeypatch):
    m = OC.build_matrix()
    floor = OC.build_floor(m)
    # A slip strictly within tolerance must still pass.
    floor["overall_coverage_ratio"] = (
        m["summary"]["overall_coverage_ratio"] + OC.GATE_TOLERANCE / 2)
    p = tmp_path / "floor.json"
    p.write_text(json.dumps(floor))
    monkeypatch.setattr(OC, "FLOOR_PATH", str(p))
    assert OC.gate() == 0
