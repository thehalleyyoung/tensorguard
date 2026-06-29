"""Roadmap **step 9 — coverage metric for derived contracts**.

A :class:`~src.symexec.model_contract.ModelContract` is deliberately *partial*:
it emits only the parameters it can prove, and abstains on the rest.  This suite
turns "how much does it prove?" into a **measured, regression-gated** number:

``coverage = soundly-emitted state_dict params / params PyTorch registers``.

Two layers of testing:

* **Pure** unit tests of :meth:`ModelContract.coverage` /
  :class:`CoverageReport` — torch-free, exercising the soundness gate, the
  partial/missing accounting, and the empty-oracle edge.
* A **torch-gated dashboard + baseline** test that builds every real fixture
  module, measures coverage against its true ``state_dict()``, prints a table,
  and asserts (a) **every** emission is sound (zero false positives) and (b)
  coverage never regresses below ``tests/data/coverage_baseline.json``.

The torch oracle lives only in :mod:`tests._torch_oracle`; torch never touches
TensorGuard's trust path.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from src.symexec import CoverageReport, derive_model_contract
from src.symexec.model_contract import ModelContract

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _torch_oracle import FIXTURES, state_dict_shapes  # noqa: E402

BASELINE_PATH = pathlib.Path(__file__).parent / "data" / "coverage_baseline.json"


def _baseline() -> dict:
    data = json.loads(BASELINE_PATH.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _contract(source: str, construction: str) -> ModelContract:
    return derive_model_contract(source, construction)


# --------------------------------------------------------------------------- #
# Pure (torch-free) unit tests of the coverage computation.                     #
# --------------------------------------------------------------------------- #
_LINEAR = """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 8)
"""


def test_full_coverage_when_oracle_matches_exactly():
    mc = _contract(_LINEAR, "M()")
    # The true torch state_dict for one Linear(4, 8).
    oracle = {"a.weight": (8, 4), "a.bias": (8,)}
    r = mc.coverage(oracle)
    assert isinstance(r, CoverageReport)
    assert r.emitted == 2
    assert r.registered == 2
    assert r.num_correct == 2
    assert r.unsound == ()
    assert r.missing == ()
    assert r.fraction == 1.0
    assert r.is_sound


def test_partial_coverage_counts_missing_not_unsound():
    mc = _contract(_LINEAR, "M()")
    # Oracle registers an extra param the (partial) contract did not emit.
    oracle = {"a.weight": (8, 4), "a.bias": (8,), "extra.weight": (3, 3)}
    r = mc.coverage(oracle)
    assert r.num_correct == 2
    assert r.registered == 3
    assert r.missing == ("extra.weight",)
    assert r.unsound == ()            # missing != unsound
    assert r.is_sound                 # a partial contract is still sound
    assert r.fraction == pytest.approx(2 / 3)


def test_wrong_shape_emission_is_unsound():
    mc = _contract(_LINEAR, "M()")
    oracle = {"a.weight": (999, 4), "a.bias": (8,)}  # weight shape disagrees
    r = mc.coverage(oracle)
    assert "a.weight" in r.unsound
    assert not r.is_sound
    # The wrong emission is NOT counted as correct.
    assert r.num_correct == 1


def test_emitted_name_absent_from_oracle_is_unsound():
    mc = _contract(_LINEAR, "M()")
    oracle = {"a.weight": (8, 4)}  # contract also emits a.bias, oracle lacks it
    r = mc.coverage(oracle)
    assert "a.bias" in r.unsound
    assert not r.is_sound


def test_empty_oracle_is_vacuously_full_when_nothing_emitted():
    src = """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()   # registers no params
"""
    mc = _contract(src, "M()")
    r = mc.coverage({})
    assert r.emitted == 0
    assert r.registered == 0
    assert r.fraction == 1.0       # vacuous full coverage, no division by zero
    assert r.is_sound


def test_coverage_report_is_frozen_and_pure():
    mc = _contract(_LINEAR, "M()")
    oracle = {"a.weight": (8, 4), "a.bias": (8,)}
    r = mc.coverage(oracle)
    with pytest.raises(Exception):
        r.emitted = 99  # frozen dataclass
    # Calling twice with the same oracle yields identical reports (pure).
    assert mc.coverage(oracle) == r


def test_coverage_normalises_list_shapes_in_oracle():
    # An oracle that hands shapes as lists (e.g. parsed JSON) still compares
    # equal to the contract's tuple shapes.
    mc = _contract(_LINEAR, "M()")
    oracle = {"a.weight": [8, 4], "a.bias": [8]}
    r = mc.coverage(oracle)
    assert r.is_sound and r.fraction == 1.0


# --------------------------------------------------------------------------- #
# Torch-gated dashboard + non-regression baseline.                              #
# --------------------------------------------------------------------------- #
def test_baseline_file_covers_every_fixture():
    base = _baseline()
    names = {f.name for f in FIXTURES}
    assert set(base) == names, (
        "coverage_baseline.json must have exactly one entry per fixture; "
        f"missing={names - set(base)} extra={set(base) - names}"
    )
    assert all(0.0 <= v <= 1.0 for v in base.values())


def test_coverage_dashboard_sound_and_no_regression():
    pytest.importorskip("torch")
    base = _baseline()

    rows = []
    failures = []
    for fx in FIXTURES:
        oracle = state_dict_shapes(fx.source, fx.construction)
        mc = derive_model_contract(fx.source, fx.construction)
        r = mc.coverage(oracle)
        rows.append((fx.name, r))

        # (a) HARD soundness gate: a partial contract may emit fewer params,
        # but every param it DOES emit must match torch exactly.
        if not r.is_sound:
            failures.append(f"{fx.name}: UNSOUND emissions {r.unsound}")

        # (b) Non-regression: coverage must not drop below the recorded baseline.
        want = base[fx.name]
        if r.fraction + 1e-9 < want:
            failures.append(
                f"{fx.name}: coverage regressed {r.fraction:.4f} < baseline {want:.4f}"
            )

    # Dashboard (visible with `pytest -s`).
    print("\n=== model-contract coverage dashboard ===")
    print(f"{'fixture':24} {'emit':>5} {'reg':>5} {'ok':>5} {'cover':>7} sound")
    for name, r in rows:
        print(
            f"{name:24} {r.emitted:5d} {r.registered:5d} {r.num_correct:5d} "
            f"{r.fraction:7.3f} {r.is_sound}"
        )
    overall_ok = sum(r.num_correct for _, r in rows)
    overall_reg = sum(r.registered for _, r in rows)
    print(f"{'TOTAL':24} {'':>5} {overall_reg:5d} {overall_ok:5d} "
          f"{overall_ok / overall_reg:7.3f}")

    assert not failures, "\n".join(failures)


def test_every_partial_contract_is_sound_against_torch():
    # Independent of the baseline: across the whole fixture corpus, the deriver
    # must NEVER emit a parameter torch doesn't register with that exact shape.
    pytest.importorskip("torch")
    for fx in FIXTURES:
        oracle = state_dict_shapes(fx.source, fx.construction)
        r = derive_model_contract(fx.source, fx.construction).coverage(oracle)
        assert r.is_sound, f"{fx.name}: unsound emissions {r.unsound}"


def test_baseline_is_achievable_not_stale():
    # Guard against a stale baseline that is *above* achievable coverage (which
    # would make the regression gate impossible to satisfy). Every baseline
    # value must be <= the currently-measured coverage.
    pytest.importorskip("torch")
    base = _baseline()
    for fx in FIXTURES:
        oracle = state_dict_shapes(fx.source, fx.construction)
        r = derive_model_contract(fx.source, fx.construction).coverage(oracle)
        assert base[fx.name] <= r.fraction + 1e-9, (
            f"{fx.name}: baseline {base[fx.name]} exceeds achievable "
            f"coverage {r.fraction}"
        )
