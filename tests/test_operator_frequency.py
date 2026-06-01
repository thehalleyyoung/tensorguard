"""Step 22 -- tests for the operator-frequency census harness."""

from __future__ import annotations

import json
import os

import pytest

from evaluation import operator_frequency as of

HERE = os.path.dirname(os.path.abspath(of.__file__))
JSON_PATH = os.path.join(HERE, "operator_frequency.json")
MD_PATH = os.path.join(HERE, "operator_frequency.md")


def _committed():
    with open(JSON_PATH) as fh:
        return json.load(fh)


def test_artifacts_exist():
    assert os.path.exists(JSON_PATH)
    assert os.path.exists(MD_PATH)


def test_json_is_deterministic_bytes():
    text = open(JSON_PATH).read()
    assert text == of._dumps(json.loads(text))


def test_census_includes_step22_ops():
    census = of.implemented_operator_census()
    for op in ("permute", "expand", "repeat"):
        assert op in census, "%s should be a recognised operator" % op


def test_step22_ops_marked_covered_in_artifact():
    rep = _committed()
    by_name = {o["operator"]: o for o in rep["operators"]}
    for op in rep["step22_implemented"]:
        if op in by_name:
            assert by_name[op]["covered"] is True


def test_summary_weights_consistent():
    rep = _committed()
    s = rep["summary"]
    assert s["covered_occurrences"] + s["uncovered_occurrences"] == \
        s["total_op_occurrences"]
    total = sum(o["frequency"] for o in rep["operators"])
    assert total == s["total_op_occurrences"]


def test_uncovered_ranked_matches_operators():
    rep = _committed()
    derived = [o for o in rep["operators"] if not o["covered"]]
    assert rep["uncovered_ranked"] == derived
    assert len(derived) == rep["summary"]["distinct_uncovered"]


def test_operators_sorted_by_descending_frequency():
    rep = _committed()
    freqs = [o["frequency"] for o in rep["operators"]]
    assert freqs == sorted(freqs, reverse=True)


def test_frequency_coverage_ratio_high():
    # The engine should reason about the overwhelming majority of real ops.
    rep = _committed()
    assert rep["summary"]["frequency_coverage_ratio"] >= 0.9


def test_check_passes_against_committed_artifact():
    rc = of.run(check=True, write=False)
    assert rc == 0  # 0 == up-to-date OR QUALIFIED version-skip


def test_markdown_renders_without_error():
    md = of.render_markdown(_committed())
    assert "Operator frequency census" in md
    assert "permute" in md
