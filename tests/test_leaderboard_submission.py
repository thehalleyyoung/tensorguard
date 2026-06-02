"""Step 174 — community leaderboard submission validator + honest re-scoring.

A public leaderboard that accepts external tool submissions via CI must (a)
reject malformed submissions before they reach the scoreboard and (b) never
trust a submitter's self-reported metrics.  We prove both against the real
frozen corpus and the real scorer:

* a well-formed entry over real corpus ids validates;
* unknown case ids, bad verdict tokens, missing ``tool``/``verdicts`` and
  self-reported metric fields are each rejected with a specific message;
* the committed example entry validates;
* the leaderboard recomputes a *lying* entry's metrics from its raw verdicts
  (an always-SAFE tool that claims recall 1.0 is scored recall 0.0).
"""

from __future__ import annotations

import pytest

from real_benchmarks.load import load_items
from reproducibility.leaderboard import _score
from reproducibility.validate_entry import (
    REPO,
    validate_entry,
    valid_case_ids,
)


@pytest.fixture(scope="module")
def corpus_ids():
    return valid_case_ids()


@pytest.fixture(scope="module")
def labels():
    return {it["id"]: it["label"] for it in load_items(verify=True)}


def _some_ids(corpus_ids, n=2):
    return sorted(corpus_ids)[:n]


def test_well_formed_entry_validates(corpus_ids):
    a, b = _some_ids(corpus_ids)
    entry = {"tool": "demo", "verdicts": {a: "UNSAFE", b: "safe"}}
    assert validate_entry(entry, corpus_ids) == []


def test_unknown_case_id_rejected(corpus_ids):
    entry = {"tool": "demo", "verdicts": {"definitely_not_a_real_id": "SAFE"}}
    problems = validate_entry(entry, corpus_ids)
    assert any("unknown corpus case id" in p for p in problems)


def test_invalid_verdict_token_rejected(corpus_ids):
    a = _some_ids(corpus_ids, 1)[0]
    problems = validate_entry({"tool": "demo", "verdicts": {a: "MAYBE"}}, corpus_ids)
    assert any("invalid verdict" in p for p in problems)


def test_missing_tool_and_verdicts_rejected(corpus_ids):
    assert any("tool" in p for p in validate_entry({"verdicts": {}}, corpus_ids))
    assert any("verdicts" in p for p in validate_entry({"tool": "x"}, corpus_ids))


def test_self_reported_metrics_rejected(corpus_ids):
    a = _some_ids(corpus_ids, 1)[0]
    entry = {"tool": "liar", "recall": 1.0, "f1": 1.0, "verdicts": {a: "UNSAFE"}}
    problems = validate_entry(entry, corpus_ids)
    assert any("self-reported metric" in p for p in problems)


def test_committed_example_entry_validates(corpus_ids):
    import json

    path = REPO / "benchmarks" / "leaderboard_entries" / "trivial-always-safe.json"
    assert validate_entry(json.loads(path.read_text()), corpus_ids) == []


def test_lying_entry_is_rescored_to_truth(labels):
    # An always-SAFE tool: on a corpus with >=1 buggy case its true recall is 0,
    # no matter what it claims. The scorer ignores claims and recomputes.
    verdicts = {cid: "SAFE" for cid in labels}
    score = _score(verdicts, labels)
    assert any(v == "buggy" for v in labels.values())  # corpus has real bugs
    assert score["recall"] == 0.0
    assert score["fp"] == 0  # always-SAFE never raises a false alarm
