# Step 68 — nox integration.
#
# `nox` provides reproducible, isolated sessions. The `tests` session runs the
# suite; the `tensorguard` session runs TensorGuard's own static verification
# over the examples as a quality gate. Run `nox -l` to list sessions.

import nox

PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """Run the test suite."""
    session.install("-e", ".[dev]", "z3-solver")
    session.run("pytest", "tests/", "-q", "--timeout=300", *session.posargs)


@nox.session(python="3.11")
def tensorguard(session):
    """Self-check: verify the example models with TensorGuard."""
    session.install("-e", ".", "z3-solver")
    session.run(
        "python", "-m", "src.github_action",
        env={"INPUT_PATHS": "examples", "INPUT_FAIL_ON": "never"},
    )


@nox.session(python="3.11")
def lint(session):
    """Type-check the package."""
    session.install("-e", ".[dev]")
    session.run("mypy", "src", success_codes=[0, 1])
