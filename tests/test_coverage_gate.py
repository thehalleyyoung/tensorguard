"""Step 82 — pin the coverage gate's configuration and prove it is enforceable.

Asserts the gated module list is consistent between the gate script and
``pyproject.toml``, that the threshold is the documented value, and (as a real
end-to-end check) runs the gate script in a subprocess and asserts it passes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib

import pytest

from reproducibility.coverage_gate import (
    GATED_MODULES,
    GATING_TESTS,
    THRESHOLD,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_threshold_is_at_least_90():
    assert THRESHOLD >= 90.0


def test_gated_modules_match_pyproject_source():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as fh:
        data = tomllib.load(fh)
    source = data["tool"]["coverage"]["run"]["source"]
    assert set(source) == set(GATED_MODULES), (
        "coverage_gate.GATED_MODULES drifted from pyproject "
        "[tool.coverage.run] source"
    )
    assert data["tool"]["coverage"]["report"]["fail_under"] == int(THRESHOLD)


def test_gated_modules_and_tests_exist():
    for m in GATED_MODULES:
        assert os.path.exists(os.path.join(_REPO, m)), m
    for t in GATING_TESTS:
        assert os.path.exists(os.path.join(_REPO, t)), t


def test_coverage_config_enables_branch():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as fh:
        data = tomllib.load(fh)
    assert data["tool"]["coverage"]["run"].get("branch") is True


@pytest.mark.slow
def test_coverage_gate_passes_end_to_end():
    proc = subprocess.run(
        [sys.executable, os.path.join("reproducibility", "coverage_gate.py")],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "RESULT: PASS" in out, out
