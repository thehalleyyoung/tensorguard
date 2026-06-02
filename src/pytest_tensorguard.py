"""Step 68 — ``pytest --tensorguard`` plugin.

Adds a ``--tensorguard`` flag that turns a test session into a static
verification gate: after the normal tests finish, TensorGuard verifies the
configured paths and fails the session (non-zero exit) if any module has a
shape/device/dtype/phase/gradient bug.  This lets a project enforce
"no architecturally-broken models land on main" with the same `pytest` command
it already runs.

The check itself is the pure Step-66 :func:`src.github_action.run_action`; the
plugin only wires options and the session hook.  Register it via the
``pytest11`` entry point (see pyproject) or load explicitly with
``pytest -p src.pytest_tensorguard``.
"""

from __future__ import annotations

from typing import List


def pytest_addoption(parser):
    group = parser.getgroup("tensorguard")
    group.addoption(
        "--tensorguard",
        action="store_true",
        default=False,
        help="Run TensorGuard static verification as a session gate.",
    )
    group.addoption(
        "--tensorguard-path",
        action="append",
        default=[],
        dest="tensorguard_paths",
        help="Path(s) to verify (repeatable). Defaults to the rootdir.",
    )
    group.addoption(
        "--tensorguard-input-shapes",
        action="store",
        default="",
        dest="tensorguard_input_shapes",
        help='Semicolon-separated input-shape specs, e.g. "x=batch,3,32,32".',
    )
    group.addoption(
        "--tensorguard-soundness-mode",
        action="store",
        default="balanced",
        dest="tensorguard_soundness_mode",
        choices=["sound", "balanced", "heuristic"],
    )


def _resolve_paths(config) -> List[str]:
    paths = config.getoption("tensorguard_paths")
    if paths:
        return list(paths)
    return [str(config.rootpath)]


def run_session_check(config):
    """Run verification for the session; return (failed, report_lines)."""
    from src.github_action import run_action, _parse_shapes

    shapes = _parse_shapes(config.getoption("tensorguard_input_shapes"))
    result = run_action(
        _resolve_paths(config),
        soundness_mode=config.getoption("tensorguard_soundness_mode"),
        input_shapes=shapes or None,
        fail_on="any",
    )
    lines: List[str] = []
    if result.total_issues == 0:
        lines.append(
            f"TensorGuard: verified {result.files_checked} file(s); no issues."
        )
    else:
        lines.append(
            f"TensorGuard: {result.total_issues} issue(s) in "
            f"{result.files_with_issues} of {result.files_checked} file(s):"
        )
        for ann in result.annotations:
            loc = f"{ann.file}:{ann.line}"
            if ann.col:
                loc += f":{ann.col}"
            lines.append(f"  {loc}: {ann.message}")
    return result.failed, lines


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if not config.getoption("tensorguard"):
        return
    failed, lines = run_session_check(config)
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("")
        for line in lines:
            reporter.write_line(line)
    else:  # pragma: no cover - terminal reporter always present under pytest
        print("\n".join(lines))
    if failed:
        session.exitstatus = 1
