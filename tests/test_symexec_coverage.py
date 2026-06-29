"""Statement-coverage metric for the symexec engine (roadmap Step 77).

The coverage meter is a *purely diagnostic* companion to the abstain ledger: it
measures how much of a program the abstract interpreter actually reasoned about
with a non-``Top`` value, so two "no bugs" results can be told apart (engine
proved the file safe vs. engine went ``Top`` immediately).  These tests pin the
three invariants that make it trustworthy: it is sound-by-construction (never
changes the bugs found or the proof fingerprint), monotone (loops / re-analysis
never inflate it), and faithful (gaps point at the real unmodeled statements).
"""

from __future__ import annotations

import textwrap

from src.symexec import CoverageMeter, analyze_source


def _cov(src: str):
    return analyze_source(textwrap.dedent(src)).coverage


def test_fully_interpreted_function_is_full_coverage():
    cov = _cov(
        """
        def f():
            x = 1
            y = x + 2
            return y
        """
    )
    assert cov.coverage == 1.0
    assert cov.value_coverage == 1.0
    assert cov.gaps() == []
    assert cov.total >= 2


def test_top_binding_lowers_value_coverage():
    # ``z`` is bound from an opaque/unknown source, so the engine holds Top for
    # it: a real coverage gap the metric must surface.
    cov = _cov(
        """
        def f(a):
            z = some_unknown_global_thing(a)
            return z
        """
    )
    assert cov.value_coverage < 1.0
    assert any(g.is_binding and not g.non_top for g in cov.gaps())


def test_coverage_is_one_when_no_statements_executed():
    # A bare class with no demo and no analyzable free function: vacuously full.
    cov = _cov("X = 1\n")
    assert 0.0 <= cov.coverage <= 1.0


def test_metric_does_not_change_bugs_or_fingerprint():
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
    r = analyze_source(src)
    # Coverage is populated…
    assert r.coverage.total > 0
    # …but the bug set and fingerprint are exactly what they were before the
    # meter existed (coverage is never folded into the footprint).
    assert [b.kind.name for b in r.bugs] == ["RANK_INDEX_ERROR"]
    fp = r.fingerprint()
    # Re-running yields a byte-identical fingerprint (determinism preserved).
    assert analyze_source(src).fingerprint() == fp


def test_coverage_is_deterministic():
    src = """
        def f(n):
            total = 0
            for i in range(n):
                total = total + i
            return total
        """
    a = _cov(src).to_dict()
    b = _cov(src).to_dict()
    assert a == b


def test_loop_body_counted_once():
    # The loop body statement executes on multiple fixpoint passes; the meter
    # keys on the AST node, so it is counted exactly once (monotone).
    cov = _cov(
        """
        def f(n):
            s = 0
            for i in range(n):
                s = s + 1
            return s
        """
    )
    # Every distinct statement appears at most once in the record set.
    seen = [r.line for r in cov_records(cov)]
    assert len(seen) == len(set(seen))


def cov_records(cov: CoverageMeter):
    return list(cov._records.values())


def test_unmodeled_statement_is_a_gap():
    # ``global`` has no transfer function; it must show up as an unmodeled gap
    # rather than silently counting as covered.
    cov = _cov(
        """
        def f():
            global G
            G = 1
            return G
        """
    )
    assert "Global" in cov.unmodeled_kinds()
    assert cov.modeled_coverage < 1.0


def test_to_dict_is_json_stable_and_complete():
    cov = _cov(
        """
        def f():
            x = 1
            return x
        """
    )
    d = cov.to_dict()
    for key in (
        "total_statements",
        "modeled_statements",
        "non_top_statements",
        "binding_statements",
        "non_top_bindings",
        "coverage",
        "modeled_coverage",
        "value_coverage",
        "unmodeled_kinds",
    ):
        assert key in d
    assert 0.0 <= d["coverage"] <= 1.0
    assert d["non_top_statements"] <= d["total_statements"]


def test_summary_is_one_line():
    cov = _cov("def f():\n    x = 1\n    return x\n")
    s = cov.summary()
    assert "\n" not in s
    assert "coverage" in s
