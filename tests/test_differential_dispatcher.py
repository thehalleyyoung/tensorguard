"""Regression tests for the differential-vs-dispatcher harness (Step 113)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproducibility import differential_dispatcher as dd  # noqa: E402

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
def data():
    return dd.measure()


def test_no_volatile_fields(data):
    for key in _walk_keys(data):
        low = str(key).lower()
        assert not any(tok in low for tok in _VOLATILE), f"volatile key: {key}"


def test_byte_deterministic(data):
    assert dd.run(check=True) == 0


def test_scale_at_least_1000(data):
    assert data["n_modules"] >= 1000
    assert data["scale_at_least_1000"] is True


def test_generators_produce_a_real_mix():
    # Each family must, under the committed seed, yield both clean and raising
    # modules -- otherwise the differential test would be vacuous.
    rng = random.Random(dd.SEED)
    seen = {fam: {"clean": 0, "raises": 0} for fam in dd.FAMILIES}
    for _ in range(60):
        for fam, gen in dd.FAMILIES.items():
            src, shapes = gen(rng)
            key = "clean" if dd._torch_runs_clean(src, shapes) else "raises"
            seen[fam][key] += 1
    for fam, counts in seen.items():
        assert counts["clean"] > 0, f"{fam}: never clean"
        assert counts["raises"] > 0, f"{fam}: never raises"


def test_zero_soundness_violations(data):
    # The load-bearing soundness property: no random module is ever proved SAFE
    # while the live dispatcher rejects it.
    assert data["n_soundness_violations"] == 0
    assert data["zero_soundness_violations"] is True
    assert data["soundness_violation_examples"] == []


def test_zero_false_alarms(data):
    assert data["n_false_alarms"] == 0
    assert data["zero_false_alarms"] is True
    assert data["false_alarm_examples"] == []


def test_decided_agreement_is_perfect(data):
    assert data["n_agree_on_decided"] == data["n_decided"]
    assert data["decided_agreement_perfect"] is True


def test_agreement_matrix_is_consistent(data):
    m = data["agreement_matrix"]
    total = sum(m.values())
    assert total == data["n_modules"]
    gt = data["ground_truth"]
    assert gt["n_clean"] + gt["n_raises"] == data["n_modules"]
