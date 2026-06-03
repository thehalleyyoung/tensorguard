"""Step 78 — pin the public API/CLI surface and the SemVer single source.

These tests fail if a stability-guaranteed symbol, keyword, or CLI subcommand is
removed or renamed without going through the deprecation process, and if the
declared version drifts from ``src.__version__`` or stops being valid SemVer.
"""

import inspect
import os
import tomllib

import src
from src.deprecation import parse_version

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The stability-guaranteed top-level exports (see DEPRECATION_POLICY.md).
_PUBLIC_API = {
    "analyze",
    "analyze_file",
    "analyze_directory",
    "analyze_function",
    "quick_check",
    "AnalysisResult",
    "Bug",
    "BugCategory",
    "SourceLocation",
    "checked",
    "__version__",
}

# Documented stable keyword arguments of verify_architecture.
_STABLE_VERIFY_KWARGS = {
    "source",
    "input_shapes",
    "check_devices",
    "check_phases",
    "check_gradients",
    "filename",
    "soundness_mode",
    "infer_inputs",
}

# Stability-guaranteed CLI subcommands.
_STABLE_SUBCOMMANDS = {
    "analyze",
    "analyze-package",
    "verify",
    "watch",
    "ci-check",
    "init",
    "report",
    "export",
    "diff",
    "server",
    "version",
    "config",
    "operator-confidence",
    "playground",
    "sarif-trends",
}


def test_public_exports_present():
    exported = set(src.__all__)
    missing = _PUBLIC_API - exported
    assert not missing, f"public API removed from src.__all__: {missing}"
    for name in _PUBLIC_API:
        assert hasattr(src, name), f"src.{name} not importable"


def test_verify_architecture_stable_kwargs():
    from src.api import verify_architecture

    params = set(inspect.signature(verify_architecture).parameters)
    missing = _STABLE_VERIFY_KWARGS - params
    assert not missing, f"stable kwargs removed: {missing}"


def test_phase7_integration_api_importable():
    from src.baseline import apply_baseline, write_baseline  # noqa: F401
    from src.framework_hooks import (  # noqa: F401
        TensorGuardCallback,
        TensorGuardTrainerCallback,
        verify_before_training,
    )
    from src.reporters import render, to_json, to_junit_xml, write_report  # noqa: F401
    from src.torch_integration import (  # noqa: F401
        TensorGuardViolation,
        guarded_compile,
        make_tensorguard_backend,
        verify_module,
    )


def test_version_is_valid_semver():
    assert parse_version(src.__version__) == (0, 1, 0)


def test_version_single_source_of_truth():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as fh:
        pyproject_version = tomllib.load(fh)["project"]["version"]
    assert pyproject_version == src.__version__
    # the CLI must not drift either
    from src.cli import main as cli_main

    assert cli_main._VERSION == src.__version__


def test_cli_subcommands_stable():
    from src.cli.main import ReftypeCliApp

    subcommands = set(ReftypeCliApp.COMMANDS.keys())
    missing = _STABLE_SUBCOMMANDS - subcommands
    assert not missing, f"stable CLI subcommands removed: {missing}"


def test_deprecation_policy_doc_exists():
    path = os.path.join(_REPO, "DEPRECATION_POLICY.md")
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()
    assert "Semantic Versioning" in text
    assert "DeprecationWarning" in text
    assert "at least one MINOR release" in text
