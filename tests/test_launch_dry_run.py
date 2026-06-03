"""Step 289 -- launch demo dry-run executes real TensorGuard code."""

from __future__ import annotations

import subprocess
import sys

import reproducibility.launch_dry_run as ldr


def test_demo_script_paths_are_extracted_and_exist():
    paths = ldr._demo_paths()
    assert "examples/quickstart.py" in paths
    assert "reproducibility/artifact_index.md" in paths
    assert all((ldr.REPO / path).exists() for path in paths)


def test_dry_run_executes_quickstart_and_bug_in_fresh_virtualenv():
    data = ldr.build_dry_run()
    assert data["summary"]["all_steps_passed"] is True
    assert data["summary"]["quickstart_verdict"] == "SAFE"
    assert data["summary"]["gallery_bug_verdict"] == "UNSAFE"
    steps = {step["name"]: step for step in data["steps"]}
    assert steps["create_fresh_virtualenv"]["passed"] is True
    assert steps["install_checkout_no_network_resolution"]["passed"] is True
    assert steps["run_gallery_bug_variant"]["bug_count"] >= 1
    assert steps["open_evidence_tour_files"]["missing_paths"] == []


def test_markdown_records_fresh_environment_and_verdicts():
    data = ldr.build_dry_run()
    md = ldr.render_markdown(data)
    assert "python -m venv --system-site-packages" in md
    assert "network dependency resolution: **False**" in md
    assert "run_quickstart_from_generated_demo" in md
    assert "run_gallery_bug_variant" in md
    assert "UNSAFE" in md


def test_cli_check_passes_against_committed_artifacts():
    proc = subprocess.run(
        [sys.executable, "reproducibility/launch_dry_run.py", "--check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_reproduce_all_and_makefile_own_launch_dry_run():
    import reproducibility.reproduce_all as ra

    assert "reproducibility/launch_dry_run.json" in ra.GENERATED_DETERMINISTIC
    assert "reproducibility/launch_dry_run.md" in ra.GENERATED_DETERMINISTIC
    assert any(step[1][-1] == "reproducibility/launch_dry_run.py" for step in ra.STEPS)
    makefile = (ldr.REPO / "Makefile").read_text(encoding="utf-8")
    assert "\nlaunch-dry-run:" in makefile
    assert "reproducibility/launch_dry_run.py --check" in makefile
