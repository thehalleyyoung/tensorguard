"""Step 81 — public API, PEP 561 typing marker, and type-stub plumbing.

Proves the shipped package is typed (carries a ``py.typed`` marker that
setuptools is configured to include) and that a downstream type-checker actually
consumes TensorGuard's annotations (revealed types are concrete, not ``Any``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib

import pytest

import src

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_py_typed_marker_exists():
    assert os.path.exists(os.path.join(_REPO, "src", "py.typed")), (
        "PEP 561 py.typed marker missing"
    )


def test_pyproject_ships_py_typed_as_package_data():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as fh:
        data = tomllib.load(fh)
    pkg_data = data["tool"]["setuptools"]["package-data"]
    assert "py.typed" in pkg_data["src"], (
        "py.typed not declared in [tool.setuptools.package-data]"
    )


def test_public_api_includes_typed_entry_points():
    for name in (
        "analyze",
        "analyze_file",
        "verify_architecture",
        "verify_file_safely",
        "verify_source_safely",
        "is_static_only_source",
        "AnalysisResult",
    ):
        assert name in src.__all__, f"{name} missing from public API"
        assert hasattr(src, name)


def test_analysis_result_type_hints_resolve():
    # Regression for the unimported `Any` in api.py annotations.
    import typing

    import src.api as api

    hints = typing.get_type_hints(api.AnalysisResult)
    assert "diagnostics" in hints


def _have_mypy() -> bool:
    try:
        import mypy  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_mypy(), reason="mypy not installed")
def test_downstream_mypy_sees_concrete_types(tmp_path):
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from src import analyze, AnalysisResult\n"
        "r: AnalysisResult = analyze('x = 1')\n"
        "reveal_type(r)\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--ignore-missing-imports",
            "--no-error-summary",
            str(consumer),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=240,
    )
    out = proc.stdout + proc.stderr
    # The revealed type must be the concrete AnalysisResult, proving the
    # py.typed marker makes mypy consume our annotations (not treat as Any).
    assert "AnalysisResult" in out, out
    assert 'Revealed type is "Any"' not in out, out
