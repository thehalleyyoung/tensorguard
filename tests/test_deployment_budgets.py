"""Step 226 -- tests for deployment latency/memory release gates."""

from evaluation import deployment_budgets as db


def test_manifest_is_deterministic_and_has_no_measurements():
    m1 = db.manifest()
    m2 = db.manifest()
    assert m1 == m2
    assert len(m1["budget_rows"]) == len(db.MODEL_SPECS) * len(db.STAGE_SPECS)
    for row in m1["budget_rows"]:
        assert "latency_s" not in row
        assert "memory_mb" not in row
        assert row["latency_budget_s"] > 0
        assert row["memory_budget_mb"] > 0


def test_pareto_curves_cover_each_backend_and_are_nondominated():
    curves = db.backend_pareto_curves()
    assert set(curves) == {stage.backend for stage in db.STAGE_SPECS}
    for backend, curve in curves.items():
        assert curve == sorted(curve, key=lambda p: (p["latency_budget_s"], p["memory_budget_mb"]))
        assert len(curve) >= 2
        for idx, point in enumerate(curve):
            assert point["latency_budget_s"] > 0
            assert point["memory_budget_mb"] > 0
            for other_idx, other in enumerate(curve):
                if idx != other_idx:
                    assert not db._dominates(other, point), (backend, other, point)


def test_gate_rows_reference_pareto_profiles():
    profiles_by_backend = {
        backend: {point["profile"] for point in points}
        for backend, points in db.backend_pareto_curves().items()
    }
    for row in db.budget_rows():
        assert row["profile"] in profiles_by_backend[row["backend"]]


def test_committed_manifest_is_up_to_date():
    assert db.run(check=True) == 0


def test_live_deployment_gate_passes_supported_backends():
    rows = db.measure()
    assert len(rows) == len(db.budget_rows())
    checked = [row for row in rows if row["status"] == "passed"]
    skipped = [row for row in rows if row["status"] == "skipped"]
    failed = [row for row in rows if row["status"] == "failed"]
    assert not failed
    if checked:
        assert all(row["within_budget"] for row in checked)
        assert {row["pipeline"] for row in checked} >= {"export", "compile"}
        assert any(row["phase"] == "after" for row in checked)
    else:
        assert skipped
    assert db.gate() == 0
