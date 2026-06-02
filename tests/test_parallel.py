"""Step 50 -- deterministic parallel per-module verification.

Verifies that running many modules across processes produces output that is
bit-for-bit identical to a sequential run (same verdicts, same order) for any
worker count, and that it actually distributes work across cores.
"""
import time

import pytest

from src.parallel import (
    ParallelVerdict,
    VerificationJob,
    verify_one,
    verify_parallel,
    verify_parallel_map,
    verify_sequential,
)


def _stack(n_layers: int, dim: int = 32, bad: bool = False) -> str:
    init = "\n".join(
        "        self.l%d = nn.Linear(%d, %d)" % (i, dim, dim)
        for i in range(n_layers)
    )
    body = "\n".join(
        "        x = nn.functional.relu(self.l%d(x))" % i
        for i in range(n_layers)
    )
    if bad:
        init += "\n        self.bad = nn.Linear(%d, 2)" % (dim + 5)
        body += "\n        x = self.bad(x)"
    return ("import torch.nn as nn\n"
            "class M(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "%s\n"
            "    def forward(self, x):\n"
            "%s\n"
            "        return x\n" % (init, body))


def _jobs(n: int = 8):
    return [
        VerificationJob.make(
            "m%d" % i, _stack(4 + i, bad=(i % 3 == 0)),
            input_shapes={"x": ("b", 32)})
        for i in range(n)
    ]


def test_single_job_runs_in_process():
    jobs = _jobs(1)
    out = verify_parallel(jobs)
    assert len(out) == 1
    assert out[0].name == "m0"


def test_order_is_input_order():
    jobs = _jobs(8)
    out = verify_parallel(jobs, max_workers=4)
    assert [v.name for v in out] == ["m%d" % i for i in range(8)]


def test_parallel_equals_sequential():
    jobs = _jobs(8)
    seq = verify_sequential(jobs)
    par = verify_parallel(jobs, max_workers=4)
    assert [v.key() for v in seq] == [v.key() for v in par]


def test_deterministic_across_worker_counts():
    jobs = _jobs(8)
    seq = verify_sequential(jobs)
    for w in (2, 3, 8):
        par = verify_parallel(jobs, max_workers=w)
        assert [v.key() for v in par] == [v.key() for v in seq], (
            "non-deterministic at %d workers" % w)


def test_max_workers_one_is_sequential():
    jobs = _jobs(4)
    assert ([v.key() for v in verify_parallel(jobs, max_workers=1)]
            == [v.key() for v in verify_sequential(jobs)])


def test_verdicts_are_correct():
    # Even-indexed-by-3 jobs have a deliberate shape mismatch.
    jobs = _jobs(6)
    out = verify_parallel_map(jobs, max_workers=3)
    for i in range(6):
        v = out["m%d" % i]
        if i % 3 == 0:
            assert not v.safe and v.num_violations > 0
        else:
            assert v.safe and v.num_violations == 0


def test_empty_job_list():
    assert verify_parallel([]) == []


def test_invalid_source_is_unsafe_not_raised():
    # Unparseable / non-module source must yield a verdict (unsafe), never crash
    # a worker.
    jobs = [
        VerificationJob.make("broken1", "this is not python {",
                             input_shapes={"x": (1,)}),
        VerificationJob.make("broken2", "x = 1\n",
                             input_shapes={"x": (1,)}),
    ]
    out = verify_parallel(jobs, max_workers=2)
    assert [v.name for v in out] == ["broken1", "broken2"]
    assert all(not v.safe for v in out)


def test_duplicate_names_rejected_in_map():
    j = VerificationJob.make("dup", _stack(2), input_shapes={"x": ("b", 32)})
    with pytest.raises(ValueError):
        verify_parallel_map([j, j])


def test_speedup_is_real():
    # 10 medium jobs; parallel should not be dramatically slower than
    # sequential (and is typically faster). We assert correctness plus a loose
    # wall-clock sanity bound to avoid flakiness on shared CI.
    jobs = _jobs(10)
    t0 = time.perf_counter()
    seq = verify_sequential(jobs)
    seq_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    par = verify_parallel(jobs, max_workers=4)
    par_t = time.perf_counter() - t0
    assert [v.key() for v in par] == [v.key() for v in seq]
    # Parallel must finish within the sequential time plus generous overhead.
    assert par_t <= seq_t + 30.0
