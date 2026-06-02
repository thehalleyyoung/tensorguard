"""Step 83 — validate the compatibility-matrix CI workflow.

A broken matrix (an impossible python/torch pairing, a missing referenced test,
or malformed YAML) would only surface as a red CI run after merge. These tests
parse `.github/workflows/matrix.yml` and assert the matrix is internally
consistent before it can land.
"""

from __future__ import annotations

import os

import pytest

yaml = pytest.importorskip("yaml")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKFLOW = os.path.join(_REPO, ".github", "workflows", "matrix.yml")

# Minimum Python supported by each torch line (CPython, CPU wheels).
# Sources: PyTorch release notes / wheel availability.
_TORCH_PY_RANGE = {
    "2.2.2": ((3, 8), (3, 12)),
    "2.4.1": ((3, 8), (3, 12)),
    "2.6.0": ((3, 9), (3, 13)),
    "2.7.0": ((3, 9), (3, 13)),
}


def _load():
    with open(_WORKFLOW, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_workflow_parses():
    data = _load()
    assert "jobs" in data
    assert "matrix" in data["jobs"]
    assert "nightly-torch" in data["jobs"]


def _py_tuple(s: str):
    a, b = s.split(".")
    return (int(a), int(b))


def test_every_pairing_is_compatible():
    include = _load()["jobs"]["matrix"]["strategy"]["matrix"]["include"]
    assert include, "matrix include list is empty"
    for entry in include:
        py = _py_tuple(str(entry["python-version"]))
        torch_v = str(entry["torch-version"])
        assert torch_v in _TORCH_PY_RANGE, f"unknown torch {torch_v}"
        lo, hi = _TORCH_PY_RANGE[torch_v]
        assert lo <= py <= hi, (
            f"python {py} is outside torch {torch_v} support {lo}..{hi} "
            f"on {entry['os']}"
        )


def test_matrix_covers_three_oses():
    include = _load()["jobs"]["matrix"]["strategy"]["matrix"]["include"]
    oses = {e["os"] for e in include}
    assert {"ubuntu-latest", "macos-latest", "windows-latest"} <= oses


def test_matrix_covers_python_3_9_through_3_13():
    include = _load()["jobs"]["matrix"]["strategy"]["matrix"]["include"]
    pys = {str(e["python-version"]) for e in include}
    for v in ("3.9", "3.10", "3.11", "3.12", "3.13"):
        assert v in pys, f"python {v} not covered by the matrix"


def test_referenced_test_files_exist():
    data = _load()
    text = open(_WORKFLOW, "r", encoding="utf-8").read()
    for job in ("matrix", "nightly-torch"):
        assert data["jobs"][job]["steps"], job
    for token in text.split():
        if token.startswith("tests/test_") and token.endswith(".py"):
            assert os.path.exists(os.path.join(_REPO, token)), token


def test_nightly_is_non_blocking():
    job = _load()["jobs"]["nightly-torch"]
    assert job.get("continue-on-error") is True


def test_fail_fast_disabled():
    strat = _load()["jobs"]["matrix"]["strategy"]
    assert strat.get("fail-fast") is False
