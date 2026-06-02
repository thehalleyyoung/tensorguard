#!/usr/bin/env python
"""Step 82 — enforceable coverage gate for TensorGuard's supported surface.

`src/` is large and includes research/experimental code paths; rather than a
misleading whole-tree number, this gate enforces >=90% **line coverage on the
public API and the Phase 7/8 integration modules** — the surface users actually
depend on and that ships in the wheel.

Run directly (`python reproducibility/coverage_gate.py`) or in CI
(`.github/workflows/coverage.yml`). Exits non-zero if any gated module falls
below the threshold, printing a per-module table.

The single source of truth for the gated modules and their tests lives here and
is asserted against `pyproject.toml`'s `[tool.coverage.run] source` by
`tests/test_coverage_gate.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys

THRESHOLD = 90.0

# Gated modules (relative to repo root) and the tests that exercise them.
GATED_MODULES = [
    "src/safe_loader.py",
    "src/reporters.py",
    "src/baseline.py",
    "src/deprecation.py",
    "src/torch_integration.py",
    "src/framework_hooks.py",
]

GATING_TESTS = [
    "tests/test_security.py",
    "tests/test_reporters.py",
    "tests/test_baseline.py",
    "tests/test_deprecation.py",
    "tests/test_torch_integration.py",
    "tests/test_framework_hooks.py",
    "tests/test_typing.py",
]

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    cov_args = []
    for m in GATED_MODULES:
        cov_args.append(f"--cov={m.replace('/', '.')[:-3]}")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *GATING_TESTS,
        *cov_args,
        f"--cov-fail-under={THRESHOLD:g}",
        "--cov-report=term-missing",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    print("coverage gate: threshold >= %.0f%% on:" % THRESHOLD)
    for m in GATED_MODULES:
        print("  -", m)
    proc = subprocess.run(cmd, cwd=_REPO)
    if proc.returncode == 0:
        print("\nRESULT: PASS (gated coverage >= %.0f%%)" % THRESHOLD)
    else:
        print("\nRESULT: FAIL (gated coverage < %.0f%%)" % THRESHOLD)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
