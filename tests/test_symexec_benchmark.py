"""Performance-benchmark harness for the symexec engine (roadmap Step 78).

Two guarantees are pinned here:

* **Per-file latency budget** — every file in the curated corpus, and even
  adversarially-shaped inputs (deep recursion, deeply-nested loops, wide
  branching), analyses well under a generous wall-clock budget.  This is the CI
  gate that catches an accidental super-linear regression.
* **Iteration caps bound cost deterministically** — the engine's recursion /
  loop / disjunction caps mean the *same* pathological file takes the *same*
  bounded work every run, so the latency budget is enforceable rather than
  hopeful.  The optional ``budget_ms`` guard is a sound, defence-in-depth coarse
  stop: when it trips, analysis abstains on the rest and never invents a report.
"""

from __future__ import annotations

import textwrap

import pytest

from src.symexec import (
    ITERATION_CAPS,
    analyze_source,
    benchmark_paths,
    benchmark_source,
    summarise,
)

# A generous ceiling: real corpus files measure well under a millisecond, so a
# 2-second budget is a ~1000x cushion that only a genuine blow-up would breach.
PER_FILE_BUDGET_MS = 2000.0


def _corpus_records():
    return benchmark_paths(
        ["tests/symexec_corpus/wild", "tests/symexec_corpus/correct"], repeats=1
    )


def test_every_corpus_file_is_within_latency_budget():
    records = _corpus_records()
    assert records, "no corpus files were benchmarked"
    slow = [(r.filename, r.wall_ms) for r in records if r.wall_ms > PER_FILE_BUDGET_MS]
    assert not slow, f"files over {PER_FILE_BUDGET_MS}ms budget: {slow}"
    assert all(r.error is None for r in records), [r for r in records if r.error]


def test_summary_profile_is_well_formed():
    s = summarise(_corpus_records())
    assert s["files"] > 0
    assert s["errors"] == 0
    assert s["mean_ms"] <= s["max_ms"] + 1e-9
    assert s["p95_ms"] <= s["max_ms"] + 1e-9
    assert s["max_ms"] <= PER_FILE_BUDGET_MS
    assert s["iteration_caps"] == dict(ITERATION_CAPS)


def test_iteration_caps_are_exposed():
    for key in (
        "max_call_depth",
        "loop_unroll",
        "loop_fixpoint_max",
        "loop_narrow_max",
        "disjunction_width",
    ):
        assert key in ITERATION_CAPS
        assert ITERATION_CAPS[key] >= 1


def test_deep_recursion_terminates_under_budget():
    # Mutual/self recursion is capped by max_call_depth, so this returns fast
    # rather than diverging.
    src = textwrap.dedent(
        """
        def f(x):
            return f(x) + 1

        def g(x, y):
            a = f(x)
            return a
        """
    )
    rec = benchmark_source(src, repeats=1)
    assert rec.error is None
    assert rec.wall_ms < PER_FILE_BUDGET_MS


def test_nested_loops_terminate_under_budget():
    # Loop fixpoints are bounded by the unroll/fixpoint/narrow caps; nesting them
    # multiplies a *constant* number of passes, not an unbounded one.
    src = textwrap.dedent(
        """
        def f(n):
            t = 0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        t = t + i + j + k
            return t
        """
    )
    rec = benchmark_source(src, repeats=1)
    assert rec.error is None
    assert rec.wall_ms < PER_FILE_BUDGET_MS


def test_wide_branching_terminates_under_budget():
    # Disjunctive state is capped at ITERATION_CAPS['disjunction_width']; a long
    # if/elif chain collapses to a sound join rather than exploding 2^n.
    branches = "\n".join(
        f"    {'if' if i == 0 else 'elif'} x == {i}:\n        y = {i}" for i in range(40)
    )
    src = "def f(x):\n    y = 0\n" + branches + "\n    return y\n"
    rec = benchmark_source(src, repeats=1)
    assert rec.error is None
    assert rec.wall_ms < PER_FILE_BUDGET_MS


# -- budget guard (sound coarse stop) -----------------------------------


def _many_functions(n: int) -> str:
    return "\n".join(f"def f{i}(x):\n    y = x + {i}\n    return y\n" for i in range(n))


def test_budget_guard_stops_early_and_records_abstain():
    src = _many_functions(60)
    rec = benchmark_source(src, repeats=1, budget_ms=0.0)
    assert rec.budget_exceeded
    # Stopping early is sound: it never invents a bug.
    assert rec.bugs == 0


def test_budget_none_is_byte_identical_to_default():
    # The opt-in budget must not perturb the historic unbounded behaviour.
    src = textwrap.dedent(
        """
        import torch

        def monte_carlo_rollout(model, x):
            output = x
            output = output[-1, :, :]
            return output

        if __name__ == "__main__":
            x = torch.randn(10, 32)
            monte_carlo_rollout(None, x)
        """
    )
    default = analyze_source(src)
    unbounded = analyze_source(src, budget_ms=None)
    generous = analyze_source(src, budget_ms=PER_FILE_BUDGET_MS)
    assert default.fingerprint() == unbounded.fingerprint() == generous.fingerprint()
    assert [b.kind.name for b in default.bugs] == [b.kind.name for b in generous.bugs]


def test_generous_budget_does_not_trip_on_corpus():
    records = benchmark_paths(
        ["tests/symexec_corpus/wild"], repeats=1, budget_ms=PER_FILE_BUDGET_MS
    )
    assert records
    assert not any(r.budget_exceeded for r in records)


def test_benchmark_record_to_dict_is_json_ready():
    rec = benchmark_source("def f():\n    return 1\n", repeats=1)
    d = rec.to_dict()
    for key in ("filename", "wall_ms", "statements", "bugs", "coverage", "abstentions"):
        assert key in d
    assert d["wall_ms"] >= 0.0
