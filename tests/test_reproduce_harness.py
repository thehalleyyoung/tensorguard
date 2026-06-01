"""Regression tests for the from-scratch reproduction harness (Step 10).

These tests call the Python orchestrator directly (not via ``make``) so the
pytest suite does not depend on a ``make`` binary. A separate, ``make``-guarded
integration test exercises the user-facing ``make reproduce-check`` target when
``make`` is available.
"""

import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from reproducibility import reproduce_all  # noqa: E402


@pytest.fixture(scope="module")
def regenerated():
    """Run the orchestrator once for the whole module (idempotent)."""
    rc = reproduce_all.run(check=False)
    return rc


def _git_show_head(path):
    """Return the committed (HEAD) bytes of a tracked path, or None if absent."""
    proc = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return proc.stdout if proc.returncode == 0 else None


def test_orchestrator_runs_end_to_end_and_audit_passes(regenerated):
    # Regenerates every CI-reproducible artifact in place and runs the numeric
    # audit. Idempotent, so this leaves the tree byte-identical to its prior
    # state for committed artifacts.
    assert regenerated == 0, "reproduce_all.run() failed (a regenerator or the numeric audit failed)"


def test_regeneration_is_byte_identical_to_committed(regenerated):
    # After regeneration, every byte-deterministic generated artifact must match
    # exactly what is committed at HEAD. Comparing against HEAD (not the working
    # tree) isolates 'regeneration reproduces the committed bytes' from unrelated
    # local edits. Proves the committed artifacts are not stale.
    paths = reproduce_all.GENERATED_DETERMINISTIC + reproduce_all._corpus_repro_paths()
    mismatches = []
    for rel in paths:
        committed = _git_show_head(rel)
        if committed is None:
            continue  # not yet committed; skip
        with open(os.path.join(REPO, rel), "rb") as fh:
            current = fh.read()
        if current != committed:
            mismatches.append(rel)
    assert not mismatches, (
        "regeneration does not reproduce committed bytes for: "
        + ", ".join(mismatches)
        + " (run `make reproduce` and commit the result)"
    )


def test_headline_json_excluded_from_determinism_check():
    # The headline JSON carries a volatile elapsed_s field, so it must NOT be in
    # the byte-diff set; its scientific content is validated by the numeric audit.
    assert "reproducibility/reproduce_headline_60bug.json" not in reproduce_all.GENERATED_DETERMINISTIC
    assert "reproducibility/reproduce_headline_60bug.json" in reproduce_all.VOLATILE_REGENERATED


def test_generated_paths_list_is_consistent(regenerated):
    # Every declared deterministic path actually exists on disk after a run.
    for rel in reproduce_all.GENERATED_DETERMINISTIC + reproduce_all._corpus_repro_paths():
        assert os.path.exists(os.path.join(REPO, rel)), f"missing generated artifact: {rel}"


@pytest.mark.skipif(shutil.which("make") is None, reason="make not available")
def test_make_reproduce_check_target():
    # Exercise the user-facing Makefile target end-to-end so it cannot rot.
    proc = subprocess.run(
        ["make", "reproduce-check"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert proc.returncode == 0, proc.stdout.decode("utf-8", "replace")[-3000:]
