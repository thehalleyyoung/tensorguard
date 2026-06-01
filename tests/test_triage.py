"""Step 18 regression tests -- the frozen minimal-reproducer suite.

These tests *are* the converted disagreements: each frozen entry is replayed as
a regression. A buggy reproducer must stay caught (runtime raises AND
TensorGuard refutes); its clean sibling must stay clean (runtime clean AND
TensorGuard accepts). If TensorGuard ever regresses on any pattern, the matching
case fails.
"""

from __future__ import annotations

import json
import os

import pytest

from evaluation import triage


with open(triage.OUT_JSON, "r", encoding="utf-8") as _fh:
    _ARTIFACT = json.load(_fh)

_ENTRIES = _ARTIFACT["regression_suite"]["entries"]
_IDS = [e["id"] for e in _ENTRIES]


def test_suite_has_fifty_reproducers_across_many_categories():
    assert _ARTIFACT["regression_suite"]["count"] == triage.N_REGRESSIONS
    assert len(_ENTRIES) == triage.N_REGRESSIONS
    assert len(_ARTIFACT["regression_suite"]["by_category"]) >= 8


def test_triage_found_no_disagreements():
    t = _ARTIFACT["disagreement_triage"]
    assert t["population_total"] == t["clean_models_examined"] + t["faulty_models_examined"]
    assert t["population_total"] >= 400
    assert t["total_disagreements"] == 0
    assert t["false_positives"] == 0
    assert t["false_negatives"] == 0


@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_buggy_reproducer_stays_caught(entry):
    shape = tuple(entry["input_shapes"]["x"])
    assert triage.runtime_raises(entry["buggy_source"], shape), \
        "%s: buggy reproducer no longer raises at runtime" % entry["id"]
    assert triage.tensorguard_refutes(entry["buggy_source"], shape), \
        "%s: TensorGuard regressed -- no longer refutes the bug" % entry["id"]


@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_clean_sibling_stays_clean(entry):
    shape = tuple(entry["input_shapes"]["x"])
    assert not triage.runtime_raises(entry["clean_source"], shape), \
        "%s: clean sibling now raises at runtime" % entry["id"]
    assert not triage.tensorguard_refutes(entry["clean_source"], shape), \
        "%s: TensorGuard regressed -- false positive on the clean sibling" % entry["id"]


def test_catalogue_is_deterministic():
    a = triage.build_catalogue()
    b = triage.build_catalogue()
    assert [e["id"] for e in a] == [e["id"] for e in b]
    assert [e["buggy_source"] for e in a] == [e["buggy_source"] for e in b]


def test_committed_artifact_is_up_to_date():
    assert os.path.exists(triage.OUT_JSON)
    triage.run(check=True)
