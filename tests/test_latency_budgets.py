"""Step 46 -- end-to-end verification latency budgets.

Validates the latency-budget harness: deterministic manifest, byte-reproducible
artifact, live budget enforcement, and that every tier currently meets its
budget on this machine.
"""
import pytest

from evaluation import latency_budgets as lb


def test_corpus_covers_all_tiers():
    tiers = {tier for _n, tier, _s, _sh, _b in lb.corpus()}
    assert tiers == {"small", "medium", "large"}


def test_manifest_is_deterministic_and_has_no_timings():
    m1 = lb.manifest()
    m2 = lb.manifest()
    assert m1 == m2
    for r in m1["models"]:
        assert "latency_s" not in r
        assert r["steps"] > 0
        assert r["budget_s"] > 0


def test_committed_manifest_is_up_to_date():
    assert lb.run(check=True) == 0


def test_step_counts_increase_with_tier():
    by_tier = {}
    for r in lb.manifest()["models"]:
        by_tier.setdefault(r["tier"], []).append(r["steps"])
    assert max(by_tier["small"]) < min(by_tier["medium"])
    assert max(by_tier["medium"]) < min(by_tier["large"])


def test_gate_passes_within_budget():
    # Measures live latency; should comfortably pass given the generous budgets.
    rows = lb.measure()
    assert all(r["within_budget"] for r in rows), [
        (r["model"], r["latency_s"], r["budget_s"]) for r in rows
        if not r["within_budget"]
    ]
    assert lb.gate() == 0
