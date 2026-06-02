"""Step 50 -- deterministic parallel per-module verification.

Verifying many ``nn.Module`` sources (a package, a model zoo, a CI batch) is
embarrassingly parallel: each module is verified independently.  This module
runs those jobs across CPU cores while guaranteeing **bit-for-bit deterministic
output**: the returned verdicts are always in the caller's input order and are
identical to a sequential run, regardless of how many workers are used or in
what order they happen to finish.

Determinism rests on three facts:

  * ``verify_model`` is a pure function of (source, input shapes, options) -- it
    performs no I/O and uses no wall-clock or RNG state that affects the
    verdict -- so each job's result is independent of scheduling;
  * results are reassembled by the job's input index, never by completion
    order;
  * a fixed multiprocessing start method (``spawn`` by default) gives a clean,
    reproducible worker interpreter.

Workers return a small, picklable :class:`ParallelVerdict` summary rather than
the heavyweight Z3-derived result object, so the same verdicts cross the process
boundary cheaply.
"""
from __future__ import annotations

import concurrent.futures as _cf
import multiprocessing as _mp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.model_checker import verify_model

__all__ = [
    "VerificationJob",
    "ParallelVerdict",
    "verify_one",
    "verify_sequential",
    "verify_parallel",
    "verify_parallel_map",
]


@dataclass(frozen=True)
class VerificationJob:
    """A single verification task.

    Parameters
    ----------
    name : str
        Caller-chosen identifier (e.g. a file path or model name).
    source : str
        Python source containing the ``nn.Module`` to verify.
    input_shapes : dict, optional
        Forward-parameter shapes (ints or symbolic-dim strings).
    options : dict, optional
        Extra keyword arguments forwarded to ``verify_model`` (``check_*``
        flags, ``default_device``, etc.).  Must be picklable.
    """
    name: str
    source: str
    input_shapes: Optional[Tuple[Tuple[str, tuple], ...]] = None
    options: Tuple[Tuple[str, object], ...] = ()

    @classmethod
    def make(cls, name: str, source: str,
             input_shapes: Optional[Dict[str, tuple]] = None,
             **options: object) -> "VerificationJob":
        shp = (tuple(sorted(input_shapes.items()))
               if input_shapes else None)
        return cls(name=name, source=source, input_shapes=shp,
                   options=tuple(sorted(options.items())))

    def shapes_dict(self) -> Optional[Dict[str, tuple]]:
        return dict(self.input_shapes) if self.input_shapes else None

    def options_dict(self) -> Dict[str, object]:
        return dict(self.options)


@dataclass
class ParallelVerdict:
    """A small, picklable summary of one verification."""
    name: str
    safe: bool
    num_violations: int
    violation_kinds: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def key(self) -> Tuple:
        """A hashable, order-independent fingerprint for equality checks."""
        return (self.name, self.safe, self.num_violations,
                tuple(self.violation_kinds), self.error)


def verify_one(job: VerificationJob) -> ParallelVerdict:
    """Verify a single job; never raises (errors are captured in the verdict).

    Defined at module top level so it is picklable for ``spawn`` workers.
    """
    try:
        result = verify_model(
            job.source, input_shapes=job.shapes_dict(),
            **job.options_dict())
    except Exception as exc:  # pragma: no cover - defensive
        return ParallelVerdict(
            name=job.name, safe=False, num_violations=0,
            violation_kinds=[], error="%s: %s" % (
                type(exc).__name__, str(exc)[:200]))
    viols = list(getattr(result, "violations", None) or [])
    if not viols:
        # Unsafe results carry their violations on the counterexample trace.
        ce = getattr(result, "counterexample", None)
        if ce is not None:
            viols = list(getattr(ce, "violations", None) or [])
    # Sort kinds so the summary is canonical regardless of internal ordering.
    kinds = sorted(str(getattr(v, "kind", "")) for v in viols)
    # Surface extraction/parse errors recorded on the result, if any.
    errors = list(getattr(result, "errors", None) or [])
    err = "; ".join(str(e) for e in errors)[:200] if errors else None
    return ParallelVerdict(
        name=job.name, safe=bool(result.safe),
        num_violations=len(viols), violation_kinds=kinds, error=err)


def verify_sequential(jobs: List[VerificationJob]) -> List[ParallelVerdict]:
    """Reference implementation: verify jobs one at a time, in order."""
    return [verify_one(job) for job in jobs]


def verify_parallel(
    jobs: List[VerificationJob],
    max_workers: Optional[int] = None,
    start_method: str = "spawn",
) -> List[ParallelVerdict]:
    """Verify *jobs* across processes, returning verdicts in input order.

    The output is guaranteed identical to :func:`verify_sequential` (same
    verdicts, same order) for any ``max_workers``.  With a single job, or when
    ``max_workers == 1``, the work is done in-process to avoid pool overhead.
    """
    if not jobs:
        return []
    if max_workers is not None and max_workers <= 1:
        return verify_sequential(jobs)
    if len(jobs) == 1:
        return [verify_one(jobs[0])]

    ctx = _mp.get_context(start_method)
    n_workers = max_workers or min(len(jobs), ctx.cpu_count() or 1)

    results: List[Optional[ParallelVerdict]] = [None] * len(jobs)
    with _cf.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx
    ) as pool:
        # Submit with the input index so completion order is irrelevant.
        future_to_idx = {
            pool.submit(verify_one, job): idx
            for idx, job in enumerate(jobs)
        }
        for fut in _cf.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()

    # By construction every slot is filled; the cast keeps type-checkers happy.
    return [r for r in results if r is not None]


def verify_parallel_map(
    jobs: List[VerificationJob],
    max_workers: Optional[int] = None,
    start_method: str = "spawn",
) -> Dict[str, ParallelVerdict]:
    """Like :func:`verify_parallel` but keyed by job name (names must be unique)."""
    verdicts = verify_parallel(jobs, max_workers=max_workers,
                               start_method=start_method)
    out: Dict[str, ParallelVerdict] = {}
    for v in verdicts:
        if v.name in out:
            raise ValueError("duplicate job name: %r" % v.name)
        out[v.name] = v
    return out
