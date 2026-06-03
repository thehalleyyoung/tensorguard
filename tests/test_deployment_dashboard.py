"""Step 228 -- tests for deployment release dashboard ratchets."""

from __future__ import annotations

import copy

import pytest

from evaluation import deployment_dashboard as dd


def _with_first_row(man, **updates):
    mutated = copy.deepcopy(man)
    mutated["releases"][0]["rows"][0].update(updates)
    if "key" in updates:
        mutated["releases"][0]["rows"][0]["key"] = updates["key"]
    mutated["releases"][0]["summary"] = dd.summarize(mutated["releases"][0]["rows"])
    return mutated


def test_manifest_is_deterministic_and_tracks_four_surfaces():
    m1 = dd.manifest()
    m2 = dd.manifest()
    assert m1 == m2
    rows = m1["releases"][0]["rows"]
    assert {row["surface"] for row in rows} == {
        "quant",
        "export",
        "compile",
        "distributed",
    }
    assert all(row["release"] == dd.CURRENT_RELEASE for row in rows)
    assert all(row["status"] == "passed" for row in rows)
    assert all(row["supported"] for row in rows)
    assert m1["releases"][0]["summary"]["supported_passed"] == len(rows)


def test_committed_artifacts_are_up_to_date():
    assert dd.run(check=True) == 0


def test_committed_baseline_passes_gate():
    result = dd.compare_to_baseline(dd.manifest(), dd.load_baseline())
    assert result.ok, result


def test_supported_backend_failure_or_skip_is_a_regression():
    base = dd.build_baseline(dd.manifest())
    failed = _with_first_row(dd.manifest(), status="failed")
    failed_result = dd.compare_to_baseline(failed, base)
    assert not failed_result.ok
    assert any("passed -> failed" in item for item in failed_result.regressions)

    skipped = _with_first_row(dd.manifest(), status="skipped")
    skipped_result = dd.compare_to_baseline(skipped, base)
    assert not skipped_result.ok
    assert any("passed -> skipped" in item for item in skipped_result.regressions)


def test_supported_backend_cannot_silently_become_unsupported():
    base = dd.build_baseline(dd.manifest())
    unsupported = _with_first_row(dd.manifest(), supported=False)
    result = dd.compare_to_baseline(unsupported, base)
    assert not result.ok
    assert any("supported -> unsupported" in item for item in result.regressions)


def test_removed_and_new_supported_rows_require_baseline_review():
    base = dd.build_baseline(dd.manifest())
    removed = copy.deepcopy(dd.manifest())
    removed["releases"][0]["rows"].pop()
    removed["releases"][0]["summary"] = dd.summarize(removed["releases"][0]["rows"])
    result = dd.compare_to_baseline(removed, base)
    assert not result.ok
    assert result.missing

    extra = copy.deepcopy(dd.manifest())
    row = copy.deepcopy(extra["releases"][0]["rows"][0])
    row["key"] = "0.1.0-dev|new|backend|gate"
    row["surface"] = "new"
    extra["releases"][0]["rows"].append(row)
    extra["releases"][0]["summary"] = dd.summarize(extra["releases"][0]["rows"])
    result = dd.compare_to_baseline(extra, base)
    assert not result.ok
    assert result.unregistered_supported == (row["key"],)


def test_unsupported_optional_skip_is_not_a_supported_backend_regression():
    man = _with_first_row(dd.manifest(), status="skipped", supported=False, required=False)
    base = dd.build_baseline(man)
    result = dd.compare_to_baseline(man, base)
    assert result.ok, result


def test_live_smoke_has_no_supported_failures_when_torch_is_available():
    pytest.importorskip("torch")
    rows = dd.measure()
    assert len(rows) == len(dd.GATE_SPECS)
    failures = [
        row for row in rows
        if row["supported"] and row["live_status"] == "failed"
    ]
    assert not failures
    passed_surfaces = {
        row["surface"] for row in rows
        if row["live_status"] == "passed"
    }
    assert {"quant", "distributed"} <= passed_surfaces
