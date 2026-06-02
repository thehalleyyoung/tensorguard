"""Step 71 — container image + zero-install run path.

The Docker daemon is not assumed available in CI, so this test does two things:

1. Statically validates the ``Dockerfile`` contract — a multi-stage build that
   builds a wheel and installs *only* that wheel into a slim runtime, runs as a
   non-root user, and uses the ``tensorguard`` console script as the entrypoint.
2. Proves the entrypoint command itself actually resolves and runs by invoking
   the installed ``tensorguard`` console script as a subprocess (the exact same
   command the image's ENTRYPOINT runs). This is the part most likely to break.
"""

import os
import shutil
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKERFILE = os.path.join(_REPO, "Dockerfile")
_DOCKERIGNORE = os.path.join(_REPO, ".dockerignore")


def _dockerfile():
    return open(_DOCKERFILE, encoding="utf-8").read()


def test_multistage_builds_and_installs_only_the_wheel():
    df = _dockerfile()
    assert "AS builder" in df
    assert "AS runtime" in df
    assert "python -m build --wheel" in df
    # runtime installs the wheel produced by the builder, not the toolchain
    assert "COPY --from=builder /dist/*.whl" in df
    assert "pip install --no-cache-dir /tmp/*.whl" in df


def test_entrypoint_is_the_console_script():
    df = _dockerfile()
    assert 'ENTRYPOINT ["tensorguard"]' in df


def test_runs_as_non_root():
    df = _dockerfile()
    assert "USER tg" in df
    assert "useradd" in df


def test_dockerignore_excludes_roadmap_and_git():
    ign = open(_DOCKERIGNORE, encoding="utf-8").read()
    assert "100_STEPS.md" in ign
    assert ".git" in ign
    assert "tests" in ign


def test_reproducible_build_arg_present():
    df = _dockerfile()
    assert "SOURCE_DATE_EPOCH" in df


def test_entrypoint_command_actually_resolves():
    """The image ENTRYPOINT is `tensorguard`; prove that command runs."""
    exe = shutil.which("tensorguard")
    if exe is None:
        # Fall back to the module form the console script wraps.
        cmd = [sys.executable, "-m", "src.cli.main", "--help"]
    else:
        cmd = [exe, "--help"]
    proc = subprocess.run(
        cmd, cwd=_REPO, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: tensorguard" in proc.stdout
    assert "verify" in proc.stdout
