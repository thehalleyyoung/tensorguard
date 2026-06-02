"""Tests for the Step 118 reduced-product vs independent-domains ablation."""

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_VOLATILE = ("time", "elapsed", "timestamp", "wall", "clock",
             "_ms", "seconds", "duration", "date")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module("reproducibility.reduced_product_ablation")


@pytest.fixture(scope="module")
def data(mod):
    return mod.measure()


def test_no_volatile_keys(data):
    for k in _walk_keys(data):
        assert not any(s in k.lower() for s in _VOLATILE), f"volatile key: {k}"


def test_reduced_product_strictly_more_precise(data):
    # The reduced product eliminates every spurious warning the independent
    # product raises, with at least one genuine elimination.
    assert data["independent_false_positives"] > 0
    assert data["reduced_false_positives"] == 0
    assert data["false_positives_eliminated"] == data["independent_false_positives"]
    assert data["precision_gain_cases"] >= 7
    assert data["reduced_product_strictly_more_precise"] is True


def test_no_recall_loss_vs_real_execution(data):
    # On every program where CPython actually null-derefs, the reduced product
    # must still warn -- the reduction may never hide a real bug.
    assert data["reduced_misses_real_null"] == 0
    assert data["no_recall_loss"] is True
    for s in data["per_scenario"]:
        if s["oracle_null_deref_reachable"]:
            assert s["reduced_warns"], s["id"]


def test_lattice_refinement_holds_everywhere(data):
    assert data["n_refinements_checked"] == data["n_scenarios"]
    assert data["lattice_refinement_holds_all"] is True
    assert data["reduced_is_sound_refinement"] is True
    for s in data["per_scenario"]:
        assert s["leq_refinement_holds"], s["id"]


def test_oracle_matches_scenario_families(data):
    # Guarded-precise programs are genuinely null-safe; genuine/definite-null
    # programs genuinely null-deref. This validates the CPython oracle itself.
    for s in data["per_scenario"]:
        if s["family"] == "guarded_precise":
            assert s["oracle_null_deref_reachable"] is False, s["id"]
            assert s["is_precision_gain"] is True, s["id"]
        else:
            assert s["oracle_null_deref_reachable"] is True, s["id"]


def test_only_reductions_differ_live(mod):
    # Re-run the live interpreter on a guarded program with reductions off vs
    # on; only the reduction set differs, yet the verdict flips warn -> safe.
    scen = next(s for s in mod._build_scenarios() if s.sid == "guard_int")
    v_ind, _ = mod._run_product(scen, reduced=False)
    v_red, _ = mod._run_product(scen, reduced=True)
    assert v_ind is not None  # independent product false-alarms
    assert v_red is None  # reduced product is precise


def test_artifact_byte_deterministic(mod):
    a = mod.measure()
    b = mod.measure()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert mod.run(check=True) == 0  # committed artifact is byte-identical
