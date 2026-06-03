"""Step 126 -- reviewer-friendly one-command reproduction scripts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import reproducibility.reviewer_commands as rc

REPO = Path(__file__).resolve().parent.parent


def test_manifest_covers_main_result_groups():
    data = rc.measure()
    assert data["n_main_results"] >= 12
    ids = {row["id"] for row in data["results"]}
    for required in {
        "headline_60bug",
        "precision_recall",
        "significance",
        "sound_mode_fp",
        "hard_recall",
        "differential_fuzz",
        "negative_fuzz",
        "deployment_gallery",
        "pareto_curves",
        "paper_evidence",
    }:
        assert required in ids


def test_every_command_resolves_and_outputs_exist():
    data = rc.measure()
    assert data["all_commands_resolve"]
    assert data["all_outputs_present"]
    for row in data["results"]:
        assert row["command_resolves"], row
        assert row["all_outputs_present"], row


def test_shell_entrypoints_are_real_and_delegate_to_manifest():
    data = rc.measure()
    scripts = {row["path"]: row for row in data["scripts"]}
    assert set(scripts) == {
        "scripts/reproduce_main_results.sh",
        "scripts/check_main_results.sh",
    }
    for rel, row in scripts.items():
        path = REPO / rel
        assert row["present"]
        assert path.exists()
        assert "reproducibility.reviewer_commands" in path.read_text()


def test_make_targets_referenced_by_manifest_exist():
    makefile = (REPO / "Makefile").read_text()
    targets = set(re.findall(r"^([A-Za-z0-9_.-]+):", makefile, flags=re.MULTILINE))
    for row in rc.measure()["results"]:
        cmd = row["command"]
        if cmd.startswith("make "):
            assert cmd.split()[1] in targets


def test_dry_run_exercises_real_manifest():
    proc = subprocess.run(
        [sys.executable, "reproducibility/reviewer_commands.py", "--dry-run"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout
    assert "headline_60bug" in proc.stdout
    assert "bash scripts/reproduce_main_results.sh" in proc.stdout


def test_generated_manifest_is_byte_identical():
    assert rc.main(["--check"]) == 0


def test_reproduce_all_owns_reviewer_command_artifacts():
    import reproducibility.reproduce_all as ra

    assert "reproducibility/reviewer_commands.json" in ra.GENERATED_DETERMINISTIC
    assert "reproducibility/reviewer_commands.md" in ra.GENERATED_DETERMINISTIC
