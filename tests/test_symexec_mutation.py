"""Tests for the Step-80 mutation-testing harness (``src/symexec/mutation_testing.py``).

These are *meta-tests*: they mutate the engine's own source and verify the
harness correctly distinguishes mutants the suite **kills** from those that
**survive**, and that running a faithful pytest campaign is safe (always
restores the target file) and able to catch a deliberately broken engine.
"""

import ast
import os

import pytest

import src.symexec as symexec
from src.symexec import mutation_testing as mt


CORE_LATTICE = ["_join2", "_meet2", "_leq2", "join", "meet", "leq"]

_SAMPLE_SRC = """
def f(a, b):
    if a < b and a == 0:
        return a + 1
    return a or b
"""


# --------------------------------------------------------------------------- #
# Mutant generation                                                           #
# --------------------------------------------------------------------------- #

def test_all_operators_fire():
    mutants = mt.generate_mutants(_SAMPLE_SRC)
    kinds = {m.operator for m in mutants}
    assert {"comparison", "arithmetic", "boolean", "constant"} <= kinds


def test_mutants_are_valid_python_and_changed():
    mutants = mt.generate_mutants(_SAMPLE_SRC)
    assert mutants
    for m in mutants:
        # every mutant parses ...
        ast.parse(m.source)
        # ... and differs from the original.
        assert m.source.strip() != _SAMPLE_SRC.strip()


def test_generation_is_deterministic():
    a = mt.generate_mutants(_SAMPLE_SRC)
    b = mt.generate_mutants(_SAMPLE_SRC)
    assert [m.describe() for m in a] == [m.describe() for m in b]


def test_func_name_scoping_restricts_mutants():
    src = (
        "def outer(a, b):\n"
        "    return a < b\n"
        "\n"
        "def inner(c, d):\n"
        "    return c == d\n"
    )
    only_inner = mt.generate_mutants(src, func_names=["inner"])
    assert only_inner
    # The `a < b` comparison lives in `outer`; scoping to `inner` must exclude it.
    details = {m.detail for m in only_inner}
    assert "Lt->GtE" not in details
    assert any("Eq->NotEq" == m.detail for m in only_inner)


def test_seeded_subsample_is_reproducible():
    src = open(symexec.__file__.replace("__init__.py", "values.py")).read()
    full = mt.generate_mutants(src, func_names=CORE_LATTICE)
    assert len(full) > 10
    import random

    def sample(seed):
        rng = random.Random(seed)
        return sorted(
            (m.describe() for m in rng.sample(full, 5)),
        )

    assert sample(7) == sample(7)


# --------------------------------------------------------------------------- #
# In-process mutant loading                                                   #
# --------------------------------------------------------------------------- #

def test_load_mutant_module_resolves_relative_imports():
    import src.symexec.values as values

    source = open(values.__file__).read()
    mod = mt.load_mutant_module("src.symexec.values", source)
    # Relative imports (.symdim, src.domains.intervals) resolved to real siblings.
    assert hasattr(mod, "join") and hasattr(mod, "TOP")
    # It is a *fresh* object, not the cached one.
    assert mod is not values


# --------------------------------------------------------------------------- #
# Lattice oracle                                                              #
# --------------------------------------------------------------------------- #

def test_real_values_module_passes_lattice_oracle():
    import src.symexec.values as values

    assert mt.lattice_oracle(values) is True


def _broken_leq_source() -> str:
    """The real ``values.py`` with top-level ``leq`` forced to always return False."""
    import src.symexec.values as values

    tree = ast.parse(open(values.__file__).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "leq":
            node.body = [ast.Return(value=ast.Constant(value=False))]
    return ast.unparse(tree)


def test_lattice_oracle_kills_broken_module():
    broken = mt.load_mutant_module("src.symexec.values", _broken_leq_source())
    # leq(a, a) is now False -> reflexivity violated -> oracle reports failure.
    assert mt.lattice_oracle(broken) is False


def test_lattice_campaign_distinguishes_kill_from_survive():
    rep = mt.run_mutation_testing(
        "src.symexec.values",
        mt.lattice_oracle,
        func_names=CORE_LATTICE,
        seed=1,
    )
    assert rep.total > 0
    assert rep.killed > 0  # the suite-mirroring oracle catches real mutations
    assert 0.0 <= rep.score <= 1.0
    assert rep.killed + rep.survived == rep.total
    d = rep.to_dict()
    assert d["target"] == "src.symexec.values"
    assert d["killed"] == rep.killed
    assert "score" in d and "survivors" in d


def test_injected_mutant_list_is_scored_directly():
    killable = mt.Mutant(
        operator="manual", lineno=0, detail="leq->False", source=_broken_leq_source()
    )
    rep = mt.run_mutation_testing(
        "src.symexec.values", mt.lattice_oracle, mutants=[killable]
    )
    assert rep.total == 1 and rep.killed == 1 and rep.survived == 0
    assert rep.score == 1.0


def test_import_breaking_mutant_counts_as_killed():
    bad = mt.Mutant(
        operator="manual", lineno=0, detail="syntax", source="def (:\n"
    )
    rep = mt.run_mutation_testing(
        "src.symexec.values", mt.lattice_oracle, mutants=[bad]
    )
    assert rep.killed == 1


# --------------------------------------------------------------------------- #
# End-to-end corpus oracle (mutant interpreter)                               #
# --------------------------------------------------------------------------- #

def _corpus_cases():
    base = os.path.join("tests", "symexec_corpus")
    return [
        (
            open(os.path.join(base, "wild", "matmul_dim_mismatch.py")).read(),
            ("matmul_dim_mismatch",),
        ),
        (
            open(os.path.join(base, "correct", "good_matmul.py")).read(),
            (),
        ),
    ]


def test_corpus_oracle_passes_on_real_interpreter():
    import src.symexec.interpreter as interp

    oracle = mt.make_corpus_oracle(_corpus_cases())
    assert oracle(interp) is True


def test_corpus_oracle_detects_wrong_verdict():
    import src.symexec.interpreter as interp

    # A wrong expectation stands in for a mutant that changed the verdict: the
    # oracle must report failure (i.e. it would *kill* such a mutant).
    wrong = [(_corpus_cases()[0][0], ("reshape_size_mismatch",))]
    oracle = mt.make_corpus_oracle(wrong)
    assert oracle(interp) is False


def _matmul_disabled_interpreter_source() -> str:
    """Real ``interpreter.py`` with ``_check_matmul`` neutered to ``return None``.

    Stands in for a single mutant that silences the matmul detector; the wild
    matmul corpus case must then change verdict (no bug) and be *killed*.
    """
    import src.symexec.interpreter as interp

    tree = ast.parse(open(interp.__file__).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_check_matmul":
            node.body = [ast.Return(value=ast.Constant(value=None))]
    return ast.unparse(tree)


def test_end_to_end_corpus_oracle_kills_disabled_detector():
    oracle = mt.make_corpus_oracle(_corpus_cases())
    mutant = mt.Mutant(
        operator="manual",
        lineno=0,
        detail="_check_matmul->None",
        source=_matmul_disabled_interpreter_source(),
    )
    rep = mt.run_mutation_testing(
        "src.symexec.interpreter", oracle, mutants=[mutant]
    )
    # Disabling the matmul detector changes the wild corpus verdict => killed.
    assert rep.total == 1 and rep.killed == 1 and rep.survived == 0


# --------------------------------------------------------------------------- #
# Faithful pytest campaign (subprocess) — safety + real-suite kill            #
# --------------------------------------------------------------------------- #

def test_pytest_campaign_restores_file_and_kills_broken():
    import src.symexec.values as values

    target_file = values.__file__
    before = open(target_file).read()

    killable = mt.Mutant(
        operator="manual", lineno=0, detail="leq->False", source=_broken_leq_source()
    )
    rep = mt.run_pytest_mutation_testing(
        "src.symexec.values",
        ["tests/test_symexec_lattice.py"],
        mutants=[killable],
        timeout=120.0,
    )
    # The committed lattice suite catches a broken `leq` (reflexivity test).
    assert rep.total == 1 and rep.killed == 1

    after = open(target_file).read()
    assert after == before  # file restored byte-identical
