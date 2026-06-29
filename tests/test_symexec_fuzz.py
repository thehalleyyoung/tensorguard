"""AST-mutation fuzzing for the symexec engine (roadmap Step 79).

The fuzzer mutates a seed corpus at the AST level and re-runs the engine over
each mutant to assert the properties a trustworthy analyser must hold for *any*
syntactically-valid input: it never crashes, never hangs, is deterministic, and
— for semantics-preserving mutations — never changes the bugs it reports.  These
tests pin those guarantees and lock the operators that exercise them.
"""

from __future__ import annotations

import ast
import glob
import textwrap

import pytest

from src.symexec import (
    analyze_source,
    fuzz,
    mutate,
    rename_local,
)
from src.symexec.fuzz import FuzzCrash, _OPERATORS


def _corpus_sources():
    files = sorted(glob.glob("tests/symexec_corpus/wild/*.py")) + sorted(
        glob.glob("tests/symexec_corpus/correct/*.py")
    )
    return [open(f, encoding="utf-8").read() for f in files]


# -- campaign: robustness + determinism + metamorphic soundness ----------


def test_fuzz_campaign_finds_no_failures():
    report = fuzz(_corpus_sources(), iterations=60, base_seed=2024)
    assert report.mutants > 100, "fuzzer did not produce enough effective mutants"
    assert report.ok, "\n".join(
        f"{c.kind}/{c.operator}@{c.seed}: {c.detail}\n{c.source}"
        for c in report.crashes[:5]
    )


def test_fuzz_report_summary_is_one_line():
    report = fuzz(_corpus_sources()[:3], iterations=10, base_seed=1)
    s = report.summary()
    assert "\n" not in s and "fuzz:" in s


# -- mutate(): validity + determinism ------------------------------------


@pytest.mark.parametrize("seed", list(range(40)))
def test_every_mutant_parses(seed):
    corpus = _corpus_sources()
    src = corpus[seed % len(corpus)]
    try:
        ast.parse(src)
    except SyntaxError:
        pytest.skip("seed is a deliberately-unparseable repro")
    res = mutate(src, seed=seed)
    # A mutation never turns valid Python into invalid Python.
    ast.parse(res.source)


def test_mutate_is_deterministic():
    src = _corpus_sources()[0]
    a = mutate(src, seed=123)
    b = mutate(src, seed=123)
    assert a.source == b.source
    assert a.operator == b.operator
    assert a.changed == b.changed


def test_all_operators_can_fire():
    # Over a spread of seeds every registered operator (plus rename_local) fires
    # at least once — so the campaign genuinely exercises them all.
    fired = set()
    corpus = _corpus_sources()
    for i in range(2000):
        res = mutate(corpus[i % len(corpus)], seed=i)
        if res.changed:
            fired.add(res.operator)
    expected = {name for name, _, _ in _OPERATORS} | {"rename_local"}
    assert expected <= fired, f"never fired: {expected - fired}"


# -- rename_local is genuinely semantics-preserving ----------------------


def test_rename_local_preserves_bug_kinds():
    # Renaming a purely-local variable must not change the reported bugs.
    src = textwrap.dedent(
        """
        import torch

        def f(x):
            tmp = x[-1, :, :]
            out = tmp
            return out

        if __name__ == "__main__":
            f(torch.randn(10, 32))
        """
    )
    before = sorted(b.kind.name for b in analyze_source(src).bugs)
    renamed = rename_local(src, seed=3)
    assert renamed is not None and renamed != src
    after = sorted(b.kind.name for b in analyze_source(renamed).bugs)
    assert before == after


def test_rename_local_does_not_touch_parameters():
    # Parameters are part of the call interface (callers may pass them by
    # keyword), so they must never be renamed — only local assignments are.
    src = textwrap.dedent(
        """
        def forward(x, flag=False):
            y = x
            return y
        """
    )
    renamed = rename_local(src, seed=0)
    # Whatever happens, the parameter names survive verbatim.
    assert renamed is None or ("flag" in renamed and "def forward(x, flag" in renamed)


def test_rename_local_returns_none_when_nothing_local():
    # No local assignments → nothing safe to rename.
    src = "def f(a, b):\n    return a\n"
    assert rename_local(src, seed=0) is None


# -- robustness on adversarial hand-written inputs -----------------------


@pytest.mark.parametrize(
    "src",
    [
        "x, y, z = f()\n",
        "def f():\n    return\n    return 1, 2\n",
        "a = b = c = d[0][1][2]\n",
        "def f(*a, **k):\n    return a @ k\n",
        "class C:\n    def forward(self):\n        return self.x.y.z\n",
        "for i in range(0):\n    break\nelse:\n    pass\n",
        "lambda q: (q for q in q)\n",
    ],
)
def test_engine_never_crashes_on_odd_inputs(src):
    # analyze_source must always return a result, never raise.
    r1 = analyze_source(src)
    r2 = analyze_source(src)
    assert r1.fingerprint() == r2.fingerprint()


def test_fuzz_crash_record_carries_repro_source():
    c = FuzzCrash("crash", "mutate_operator", 7, "Boom: x", "def f():\n    pass\n")
    assert c.source and c.seed == 7 and c.kind == "crash"
