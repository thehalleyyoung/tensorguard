"""Step 45 -- AST-frontend parse-success SLA harness.

Stress-tests the source/AST frontend (`extract_computation_graph`) across a
curated corpus of real-world-style PyTorch model sources, asserting the
published parse-success floor holds and that the committed artifact stays
byte-reproducible.
"""
import importlib

import pytest

from evaluation import frontend_parse_sla as sla
from evaluation.frontend_corpus import CORPUS


def test_corpus_is_substantial_and_unique():
    assert len(CORPUS) >= 30
    # No duplicate sources masquerading as distinct models.
    assert len(set(CORPUS.values())) == len(CORPUS)


def test_every_corpus_model_parses():
    failures = []
    for name, src in CORPUS.items():
        rec = sla._eval_one(name, src)
        if not rec["parsed"]:
            failures.append((name, rec["error"]))
    assert not failures, "frontend failed to parse: %r" % failures


def test_summary_meets_floor():
    rep = sla.build_report()
    summ = rep["summary"]
    assert summ["n_models"] == len(CORPUS)
    assert summ["parsed"] == summ["n_models"]
    assert summ["parse_success_rate"] >= sla.PARSE_FLOOR
    assert summ["total_steps"] > 0


def test_gate_passes():
    assert sla.gate() == 0


def test_committed_artifact_is_up_to_date():
    # The committed JSON/MD must match a freshly-built report (the AST frontend
    # is torch-version-independent, so this is byte-reproducible everywhere).
    assert sla.run(check=True) == 0


def test_records_isolated_and_unsupported_fields():
    rep = sla.build_report()
    for r in rep["models"]:
        assert "isolated" in r and "unsupported" in r
        assert isinstance(r["steps"], int)
