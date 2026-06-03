"""Step 267 — best-paper artifact package in Docker, conda, and source modes."""

from __future__ import annotations

import reproducibility.artifact_package as pkg


def test_manifest_has_exactly_the_three_required_modes():
    data = pkg.build_manifest()
    assert data["n_modes"] == 3
    assert {m["mode"] for m in data["modes"]} == {"docker", "conda", "source"}
    assert data["all_modes_passed"] is True


def test_source_mode_runs_real_bug_smoke():
    data = pkg.build_manifest()
    smoke = data["real_smoke"]
    assert smoke["example"] == "examples/shape_bug.py"
    assert smoke["observed_status"] == "UNSAFE"
    assert smoke["n_bugs"] >= 1
    assert smoke["passed"] is True


def test_docker_mode_has_fresh_image_barriers():
    docker = next(m for m in pkg.build_manifest()["modes"] if m["mode"] == "docker")
    assert docker["passed"] is True
    checks = docker["checks"]
    assert checks["multi_stage_wheel_build"]
    assert checks["runtime_is_wheel_only"]
    assert checks["non_root_runtime"]
    assert checks["roadmap_excluded"]
    assert "docker build" in docker["fresh_machine_command"]
    assert "Dockerfile" in docker["files"]


def test_conda_mode_is_noarch_and_isolated():
    conda = next(m for m in pkg.build_manifest()["modes"] if m["mode"] == "conda")
    assert conda["passed"] is True
    checks = conda["checks"]
    assert checks["noarch_python"]
    assert checks["isolated_test_command"]
    assert checks["z3_range_matches_pyproject"]
    assert any("isolated" in b for b in conda["freshness_barriers"])


def test_source_mode_excludes_hidden_local_state():
    source = next(m for m in pkg.build_manifest()["modes"] if m["mode"] == "source")
    assert source["passed"] is True
    assert source["checks"]["roadmap_not_shipped"]
    assert source["checks"]["real_smoke_passed"]
    assert "python -m venv" in source["fresh_machine_command"]
    assert "reproduce_all.py --check" in source["full_paper_evidence_command"]


def test_docs_and_reproduce_pipeline_are_wired():
    import reproducibility.reproduce_all as ra

    assert "reproducibility/artifact_package.json" in ra.GENERATED_DETERMINISTIC
    assert "reproducibility/artifact_package.md" in ra.GENERATED_DETERMINISTIC
    assert "docs/artifact/PACKAGING.md" in ra.GENERATED_DETERMINISTIC
    assert any("artifact_package.py" in " ".join(step[1]) for step in ra.STEPS)


def test_check_mode_byte_identical():
    assert pkg.run(check=True) == 0
