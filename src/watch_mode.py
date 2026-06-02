"""A small, testable watch engine for live architecture verification.

Step 60.  ``tensorguard verify --watch FILE`` re-verifies an ``nn.Module`` file
every time it (or a sibling it imports from the same directory) changes, giving
instant feedback while editing.

The interesting logic — change detection and a single verification pass — lives
here as pure, side-effect-free functions so it can be unit-tested without a real
filesystem watcher or an infinite loop.  The CLI is a thin wrapper that polls
:func:`poll_once`, runs :func:`run_verification` on the changed files, prints the
results, and sleeps.

``verify_fn`` is injected (``verify_fn(path: str) -> AnalysisResult``-like), so
tests can drive the engine with a stub and no torch involvement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class WatchResult:
    """Outcome of verifying a single file during a watch pass."""

    path: str
    ok: bool  # True iff no error-severity bugs were reported
    bug_count: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None  # set if verification itself raised
    result: Any = None  # the underlying AnalysisResult (for rich rendering)


def snapshot_mtimes(
    paths: List[str], mtime_fn: Callable[[str], float] = os.path.getmtime
) -> Dict[str, float]:
    """Return the current modification time of each existing path."""
    out: Dict[str, float] = {}
    for p in paths:
        try:
            out[p] = mtime_fn(p)
        except OSError:
            continue
    return out


def poll_once(
    paths: List[str],
    prev_mtimes: Dict[str, float],
    mtime_fn: Callable[[str], float] = os.path.getmtime,
) -> Tuple[List[str], Dict[str, float]]:
    """One change-detection poll.

    Returns ``(changed_paths, new_mtimes)``.  A path is reported changed only if
    it was seen before (``prev_mtimes`` had an entry) and its mtime increased —
    newly-appearing files are recorded but not reported, so the first sighting
    of a file does not spuriously trigger a pass.  Pure apart from ``mtime_fn``.
    """
    new_mtimes = dict(prev_mtimes)
    changed: List[str] = []
    for p in paths:
        try:
            mt = mtime_fn(p)
        except OSError:
            continue
        prev = prev_mtimes.get(p)
        if prev is None or mt > prev:
            new_mtimes[p] = mt
            if prev is not None:
                changed.append(p)
    return changed, new_mtimes


def _bug_count(result: Any) -> int:
    # Mirror the verify command's reported count: prefer the de-duplicated
    # source-mapped diagnostics when present, else error-severity bugs.
    diagnostics = getattr(result, "diagnostics", None)
    if diagnostics:
        return len(diagnostics)
    bugs = getattr(result, "bugs", None) or []
    return sum(1 for b in bugs if getattr(b, "severity", "error") == "error")


def run_verification(
    path: str, verify_fn: Callable[[str], Any]
) -> WatchResult:
    """Verify one file, capturing any exception as a failed-but-survivable pass."""
    try:
        result = verify_fn(path)
    except Exception as e:  # a syntax error mid-edit must not kill the watcher
        return WatchResult(path=path, ok=False, error=str(e))
    n = _bug_count(result)
    return WatchResult(
        path=path,
        ok=(n == 0),
        bug_count=n,
        duration_ms=float(getattr(result, "duration_ms", 0.0) or 0.0),
        result=result,
    )


def run_pass(
    paths: List[str], verify_fn: Callable[[str], Any]
) -> List[WatchResult]:
    """Verify every path in *paths*, returning one :class:`WatchResult` each."""
    return [run_verification(p, verify_fn) for p in paths]


def format_watch_result(wr: WatchResult, use_color: bool = False) -> str:
    """Render a one-line status headline for a single file's result."""
    name = os.path.basename(wr.path)
    if wr.error is not None:
        body = f"! {name}: could not verify ({wr.error})"
        color = "31"  # red
    elif wr.ok:
        body = f"\u2713 {name}: verified safe ({wr.duration_ms:.1f}ms)"
        color = "32"  # green
    else:
        noun = "issue" if wr.bug_count == 1 else "issues"
        body = f"\u2717 {name}: {wr.bug_count} {noun} ({wr.duration_ms:.1f}ms)"
        color = "31"  # red
    if use_color:
        return f"\033[{color}m{body}\033[0m"
    return body
