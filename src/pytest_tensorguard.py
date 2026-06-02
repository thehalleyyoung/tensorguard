"""Step 68/163 — ``pytest --tensorguard`` plugin.

Adds a ``--tensorguard`` flag that turns a test session into a static
verification gate: after the normal tests finish, TensorGuard verifies the code
the suite *exercises* and fails the session (non-zero exit) if any module has a
shape/device/dtype/phase/gradient bug.  This lets a project enforce
"no architecturally-broken models land on main" with the same `pytest` command
it already runs.

Step 163 — *modules under test*.  By default the plugin no longer guesses at the
rootdir; it verifies the **project modules actually imported during the test
session** (the code the tests pull in), discovered by snapshotting ``sys.modules``
at session start and collecting the modules first imported afterwards whose
source lives under the project root (excluding the virtualenv, site-packages, the
test files themselves, and TensorGuard's own package).  An explicit
``--tensorguard-path`` still pins an exact set, and ``--tensorguard-rootdir``
restores the old whole-tree scan.

The check itself is the pure Step-66 :func:`src.github_action.run_action`; the
plugin only wires options, import discovery, and the session hook.  Register it
via the ``pytest11`` entry point (see pyproject) or load explicitly with
``pytest -p src.pytest_tensorguard``.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Set

# Populated at session start with the module names already imported, so we can
# tell which project modules the *tests* dragged in.
_MODULES_AT_START: Optional[Set[str]] = None


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
        help="Path(s) to verify (repeatable). Overrides import discovery.",
    )
    group.addoption(
        "--tensorguard-rootdir",
        action="store_true",
        default=False,
        dest="tensorguard_rootdir",
        help="Verify the whole rootdir instead of just the modules under test.",
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


def pytest_sessionstart(session):
    global _MODULES_AT_START
    if session.config.getoption("tensorguard"):
        _MODULES_AT_START = set(sys.modules)


def _under(path: str, root: str) -> bool:
    try:
        ap, ar = os.path.abspath(path), os.path.abspath(root)
        return ap == ar or ap.startswith(os.path.join(ar, ""))
    except Exception:
        return False


def _is_excluded(file_path: str) -> bool:
    parts = os.path.abspath(file_path).split(os.sep)
    # Skip virtualenvs, site/dist-packages, caches, and TensorGuard itself; the
    # test files are excluded separately by the caller.
    bad = {"site-packages", "dist-packages", ".venv", "venv", "__pycache__", ".tox"}
    if bad.intersection(parts):
        return True
    base = os.path.basename(file_path)
    return base.startswith("conftest")


def discover_modules_under_test(
    config, modules_at_start: Optional[Set[str]]
) -> List[str]:
    """Source files of project modules first imported during the session.

    A module qualifies when (1) it was imported *after* session start, (2) its
    ``__file__`` is a real ``.py`` under the project root, (3) it is not the
    plugin/TensorGuard package, a test module, or under an excluded tree.
    """
    root = str(config.rootpath)
    start = modules_at_start or set()
    test_files = set()
    for item in getattr(config, "_tg_collected_files", []) or []:
        test_files.add(os.path.abspath(item))

    out: List[str] = []
    seen: set = set()
    for name, mod in list(sys.modules.items()):
        if name in start:
            continue
        if name.startswith(("src", "tensorguard", "_pytest", "pytest", "py")):
            continue
        f = getattr(mod, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        af = os.path.abspath(f)
        if af in seen or af in test_files:
            continue
        if not _under(af, root) or _is_excluded(af):
            continue
        # A test module (file named test_*.py / *_test.py) is the harness, not
        # the code under test.
        base = os.path.basename(af)
        if base.startswith("test_") or base.endswith("_test.py"):
            continue
        seen.add(af)
        out.append(af)
    return sorted(out)


def pytest_collection_modifyitems(session, config, items):
    # Record the test files so they can be excluded from "modules under test".
    files = set()
    for it in items:
        f = getattr(it, "fspath", None)
        if f is not None:
            files.add(str(f))
    config._tg_collected_files = files


def _resolve_paths(config) -> List[str]:
    explicit = config.getoption("tensorguard_paths")
    if explicit:
        return list(explicit)
    if config.getoption("tensorguard_rootdir"):
        return [str(config.rootpath)]
    discovered = discover_modules_under_test(config, _MODULES_AT_START)
    if discovered:
        return discovered
    # Nothing identifiable was imported (e.g. a suite that imports no project
    # code) — fall back to the rootdir so the gate is never silently a no-op.
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
