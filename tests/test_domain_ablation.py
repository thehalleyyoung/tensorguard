"""Tests for the Step 117 per-domain ablation harness."""

import importlib
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
def data():
    mod = importlib.import_module("reproducibility.domain_ablation")
    return mod.measure()


def test_no_volatile_keys(data):
    for k in _walk_keys(data):
        low = k.lower()
        assert not any(s in low for s in _VOLATILE), f"volatile key: {k}"


def test_full_recall_per_verification_domain(data):
    for d in data["verification_domains"]:
        c = data["contributions"][d]
        assert c["full_recall_is_one"], d
        rf = c["recall_full"]
        assert rf["k"] == rf["n"] and rf["n"] > 0


def test_lodo_drops_own_domain_recall_to_zero(data):
    for d in data["verification_domains"]:
        c = data["contributions"][d]
        assert c["lodo_recall_is_zero"], d
        assert c["marginal_contribution"] == 1.0, d


def test_domains_orthogonal(data):
    assert data["domains_orthogonal"] is True
    m = data["recall_matrix"]
    for a in data["verification_domains"]:
        for d in data["verification_domains"]:
            if a == d:
                continue
            cell = m[a][d]
            assert cell["k"] == cell["n"], (a, d)


def test_toggle_report_crosscheck_agrees(data):
    cc = data["toggle_report_crosscheck"]
    assert cc["pairs_compared"] > 0
    assert cc["disagreements"] == 0
    assert cc["agree"] is True


def test_phase_is_diagnostic_only(data):
    assert data["phase_diagnostic"]["is_diagnostic_only"] is True
    assert data["phase_diagnostic"]["recall_full"]["k"] == 0


def test_every_domain_necessary(data):
    assert data["every_domain_necessary"] is True
    assert data["all_verification_domains_full_recall"] is True


def test_byte_determinism():
    mod = importlib.import_module("reproducibility.domain_ablation")
    assert mod.run(check=True) == 0
