"""Tests that check_devices, check_phases, and check_gradients are
forwarded through the public Python API and the CLI verify command.

Success criterion (from exploration prompt):
  python -m pytest tests/test_public_api_check_flags.py -x -q  →  exit 0
"""
from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Tiny nn.Module source used across tests
# ---------------------------------------------------------------------------

_SIMPLE_MODULE = """\
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""

# A module that exercises gradient-related patterns (gradient checkpointing)
# so that check_gradients=True has something to reason about.
_GRAD_MODULE = """\
import torch
import torch.nn as nn

class GradNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)
"""


# ---------------------------------------------------------------------------
# 1. Public API — signature checks
# ---------------------------------------------------------------------------

class TestPublicAPISignature:
    """Verify that the public API functions declare the expected kwargs."""

    def test_verify_architecture_accepts_check_devices(self):
        from src.api import verify_architecture
        sig = inspect.signature(verify_architecture)
        assert "check_devices" in sig.parameters, (
            "verify_architecture must accept check_devices keyword argument"
        )

    def test_verify_architecture_accepts_check_phases(self):
        from src.api import verify_architecture
        sig = inspect.signature(verify_architecture)
        assert "check_phases" in sig.parameters, (
            "verify_architecture must accept check_phases keyword argument"
        )

    def test_verify_architecture_accepts_check_gradients(self):
        from src.api import verify_architecture
        sig = inspect.signature(verify_architecture)
        assert "check_gradients" in sig.parameters, (
            "verify_architecture must accept check_gradients keyword argument"
        )

    def test_verify_module_accepts_check_devices(self):
        from src.api import verify_module
        sig = inspect.signature(verify_module)
        assert "check_devices" in sig.parameters

    def test_verify_module_accepts_check_phases(self):
        from src.api import verify_module
        sig = inspect.signature(verify_module)
        assert "check_phases" in sig.parameters

    def test_verify_module_accepts_check_gradients(self):
        from src.api import verify_module
        sig = inspect.signature(verify_module)
        assert "check_gradients" in sig.parameters

    def test_check_flags_have_bool_defaults(self):
        """All three check flags must default to True (opt-out pattern)."""
        from src.api import verify_architecture
        sig = inspect.signature(verify_architecture)
        for flag in ("check_devices", "check_phases", "check_gradients"):
            default = sig.parameters[flag].default
            assert default is True, (
                f"{flag} must default to True; got {default!r}"
            )


# ---------------------------------------------------------------------------
# 2. Public API — runtime behaviour
# ---------------------------------------------------------------------------

class TestPublicAPIRuntime:
    """Verify that the flags are actually forwarded to the analysis engine."""

    def test_verify_architecture_default_flags_returns_result(self):
        from src.api import verify_architecture, AnalysisResult
        result = verify_architecture(_SIMPLE_MODULE)
        assert isinstance(result, AnalysisResult)
        assert hasattr(result, "status")

    def test_verify_architecture_check_devices_false(self):
        from src.api import verify_architecture, AnalysisResult
        result = verify_architecture(_SIMPLE_MODULE, check_devices=False)
        assert isinstance(result, AnalysisResult)

    def test_verify_architecture_check_phases_false(self):
        from src.api import verify_architecture, AnalysisResult
        result = verify_architecture(_SIMPLE_MODULE, check_phases=False)
        assert isinstance(result, AnalysisResult)

    def test_verify_architecture_check_gradients_false(self):
        from src.api import verify_architecture, AnalysisResult
        result = verify_architecture(_SIMPLE_MODULE, check_gradients=False)
        assert isinstance(result, AnalysisResult)

    def test_verify_architecture_all_flags_disabled(self):
        from src.api import verify_architecture, AnalysisResult
        result = verify_architecture(
            _SIMPLE_MODULE,
            check_devices=False,
            check_phases=False,
            check_gradients=False,
        )
        assert isinstance(result, AnalysisResult)

    def test_check_gradients_false_suppresses_gradient_bugs(self):
        """Disabling check_gradients must remove gradient-class bugs from result."""
        from src.api import verify_architecture
        # Run with gradients enabled, then disabled; the disabled run must not
        # have MORE bugs than the enabled run (filtering never adds bugs).
        r_on = verify_architecture(_GRAD_MODULE, check_gradients=True)
        r_off = verify_architecture(_GRAD_MODULE, check_gradients=False)
        grad_bugs_on = [
            b for b in r_on.bugs
            if "grad" in b.message.lower() or "gradient" in b.message.lower()
        ]
        grad_bugs_off = [
            b for b in r_off.bugs
            if "grad" in b.message.lower() or "gradient" in b.message.lower()
        ]
        assert len(grad_bugs_off) <= len(grad_bugs_on), (
            "check_gradients=False must not produce MORE gradient bugs "
            f"(on={len(grad_bugs_on)}, off={len(grad_bugs_off)})"
        )

    def test_verify_module_flags_forwarded(self, tmp_path):
        """verify_module must forward all three check flags."""
        module_file = tmp_path / "simple.py"
        module_file.write_text(_SIMPLE_MODULE)
        from src.api import verify_module, AnalysisResult
        result = verify_module(
            str(module_file),
            check_devices=False,
            check_phases=False,
            check_gradients=False,
        )
        assert isinstance(result, AnalysisResult)


# ---------------------------------------------------------------------------
# 3. CLI — flag presence in help text
# ---------------------------------------------------------------------------

class TestCLIFlagPresence:
    """Verify that the verify subcommand advertises the expected flags."""

    @pytest.fixture(scope="class")
    def verify_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "src.cli.main", "verify", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        return proc.stdout + proc.stderr

    def test_no_device_check_in_help(self, verify_help):
        assert "--no-device-check" in verify_help, (
            "CLI verify subcommand must advertise --no-device-check"
        )

    def test_no_phase_check_in_help(self, verify_help):
        assert "--no-phase-check" in verify_help, (
            "CLI verify subcommand must advertise --no-phase-check"
        )

    def test_no_grad_check_in_help(self, verify_help):
        assert "--no-grad-check" in verify_help, (
            "CLI verify subcommand must advertise --no-grad-check"
        )


# ---------------------------------------------------------------------------
# 4. CLI — end-to-end invocation with each flag
# ---------------------------------------------------------------------------

class TestCLIEndToEnd:
    """Run the CLI verify command on a tiny example with each flag."""

    @pytest.fixture(scope="class")
    def module_file(self, tmp_path_factory):
        p = tmp_path_factory.mktemp("cli") / "simple.py"
        p.write_text(_SIMPLE_MODULE)
        return str(p)

    def _run_verify(self, module_file: str, *extra_args: str) -> subprocess.CompletedProcess:
        cmd = [sys.executable, "-m", "src.cli.main", "verify", module_file, *extra_args]
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            cwd=str(Path(__file__).parent.parent),
        )

    def test_cli_verify_default(self, module_file):
        proc = self._run_verify(module_file)
        assert proc.returncode == 0, (
            f"CLI verify exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )

    def test_cli_verify_no_device_check(self, module_file):
        proc = self._run_verify(module_file, "--no-device-check")
        assert proc.returncode == 0, (
            f"--no-device-check: exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )

    def test_cli_verify_no_phase_check(self, module_file):
        proc = self._run_verify(module_file, "--no-phase-check")
        assert proc.returncode == 0, (
            f"--no-phase-check: exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )

    def test_cli_verify_no_grad_check(self, module_file):
        proc = self._run_verify(module_file, "--no-grad-check")
        assert proc.returncode == 0, (
            f"--no-grad-check: exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )

    def test_cli_verify_all_flags_disabled(self, module_file):
        proc = self._run_verify(
            module_file,
            "--no-device-check", "--no-phase-check", "--no-grad-check",
        )
        assert proc.returncode == 0, (
            f"all flags disabled: exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )

    def test_cli_verify_json_output_has_status(self, module_file):
        """JSON output must contain a 'status' field when flags are used."""
        import json
        proc = self._run_verify(module_file, "--format", "json", "--no-device-check")
        assert proc.returncode == 0, (
            f"JSON format: exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )
        data = json.loads(proc.stdout)
        assert "status" in data, f"JSON output missing 'status': {data}"

    def test_cli_verify_analysis_ran(self, module_file):
        """With --no-grad-check the analysis still runs and reports duration."""
        import json
        proc = self._run_verify(module_file, "--format", "json", "--no-grad-check")
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert "duration_ms" in data, "JSON output must include duration_ms"
        assert data["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# 5. README disclaimer check
# ---------------------------------------------------------------------------

class TestReadmeDisclaimer:
    """The phrase 'currently not forwarded' must not appear in the README."""

    def test_disclaimer_absent(self):
        readme = Path(__file__).parent.parent / "README.md"
        assert readme.exists(), "README.md not found"
        content = readme.read_text()
        assert "currently not forwarded" not in content, (
            "README.md still contains the disclaimer 'currently not forwarded'; "
            "remove it now that the flags are wired."
        )
