"""Parallel analysis driver (roadmap Step 85).

Whole-package analysis (Step 82) analyses every module independently: each file is
reasoned about over its own import-augmented module in a *fresh* interpreter, with
no shared mutable state and no ordering dependency between files (imported symbols
are inlined, not analysed across a live boundary).  That makes package analysis
**embarrassingly parallel** — the unit of work is one module, and the per-module
:class:`~src.symexec.engine.SymResult` is a deterministic function of that module
alone, so the merged output is byte-identical regardless of how the work is
scheduled.

This module spreads that work across CPU cores.  The default ``"process"``
backend runs each module in a separate process (real parallelism for this
CPU-bound, GIL-bound pure-Python engine), shipping the parsed
:class:`~src.symexec.package.PackageIndex` to each worker once via a pool
initializer.  A ``"thread"`` backend (shared index, still one interpreter per
module — thread-safe because workers never mutate shared state) and a ``"serial"``
backend are also available; the process backend transparently falls back to serial
if a worker pool cannot be created (restricted sandboxes, unpicklable state).

Correctness: the result is identical to a serial
:func:`~src.symexec.package.analyze_package` run — same files, same per-file
fingerprints — because each module's analysis is self-contained and the merge is
order-independent (results are keyed by path and the report list is canonically
sorted).  The module is torch-free.

Why *module*-level and not *function*-level parallelism: a file's fingerprint folds
in the per-run abstain-coverage accumulated across all of its analysis passes in
one shared interpreter (see :mod:`.incremental`), so functions within a file are
not independently analysable without changing the result.  The file is therefore
the smallest sound parallel unit — the same unit incremental analysis re-runs.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Optional

from .engine import SymResult
from .package import PackageIndex, PackageResult, _analyze_one

__all__ = ["analyze_package_parallel"]


# --------------------------------------------------------------------------- #
# Process-pool worker plumbing (module-level so it is picklable under spawn)   #
# --------------------------------------------------------------------------- #

_WORKER_INDEX: Optional[PackageIndex] = None


def _worker_init(index: PackageIndex) -> None:
    """Pool initializer: stash the shared, read-only index in each worker once."""
    global _WORKER_INDEX
    _WORKER_INDEX = index


def _worker_analyze(args):
    """Analyze one module in a worker process; returns ``(path, SymResult)``."""
    qual, budget_ms = args
    assert _WORKER_INDEX is not None
    info = _WORKER_INDEX.modules[qual]
    return info.path, _analyze_one(_WORKER_INDEX, qual, budget_ms=budget_ms)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def _serial(index: PackageIndex, quals: List[str], budget_ms) -> Dict[str, SymResult]:
    out: Dict[str, SymResult] = {}
    for qual in quals:
        out[index.modules[qual].path] = _analyze_one(index, qual, budget_ms=budget_ms)
    return out


def analyze_package_parallel(
    root: str,
    *,
    workers: Optional[int] = None,
    backend: str = "process",
    budget_ms: Optional[float] = None,
) -> PackageResult:
    """Analyze every module under ``root`` in parallel.

    The result is identical to :func:`~src.symexec.package.analyze_package`; only
    the scheduling differs.

    Parameters
    ----------
    workers:
        Maximum worker count.  Defaults to ``min(os.cpu_count(), #modules)``.
    backend:
        ``"process"`` (default — real parallelism, isolated workers),
        ``"thread"`` (shared index, GIL-bound) or ``"serial"`` (no pool).
    budget_ms:
        Optional per-module wall-clock guard, as in :func:`analyze_source`.
    """
    index = PackageIndex.build(root)
    quals = sorted(index.modules)
    n = len(quals)

    if workers is None:
        workers = min(os.cpu_count() or 1, n) if n else 1
    workers = max(1, workers)

    if backend == "serial" or workers == 1 or n <= 1:
        results = _serial(index, quals, budget_ms)
        return PackageResult(root=root, results=results, index=index)

    if backend == "thread":
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_analyze_one, index, q, budget_ms=budget_ms): q
                for q in quals
            }
            for fut in futs:
                q = futs[fut]
                results[index.modules[q].path] = fut.result()
        return PackageResult(root=root, results=results, index=index)

    if backend != "process":
        raise ValueError(f"unknown backend {backend!r}")

    # Process backend: ship the parsed index to each worker once, fall back to
    # serial if a pool cannot be created (sandboxed env / unpicklable state).
    try:
        results = {}
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init, initargs=(index,)
        ) as ex:
            for path, res in ex.map(
                _worker_analyze, [(q, budget_ms) for q in quals]
            ):
                results[path] = res
        return PackageResult(root=root, results=results, index=index)
    except Exception:
        results = _serial(index, quals, budget_ms)
        return PackageResult(root=root, results=results, index=index)
