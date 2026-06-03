"""Tests for the open benchmark leaderboard harness (Step 95)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import reproducibility.leaderboard as lb  # noqa: E402

VOLATILE_TOKENS = ("time", "elapsed", "timestamp", "wall", "clock", "_ms",
                   "seconds", "duration", "date")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_score_confusion_and_metrics():
    labels = {"a": "buggy", "b": "buggy", "c": "clean", "d": "clean"}
    verdicts = {"a": "UNSAFE", "b": "SAFE", "c": "SAFE", "d": "UNSAFE"}
    s = lb._score(verdicts, labels)
    assert (s["tp"], s["fn"], s["tn"], s["fp"]) == (1, 1, 1, 1)
    assert s["recall"] == 0.5
    assert s["precision"] == 0.5
    assert s["accuracy"] == 0.5


def test_unknown_on_clean_is_not_a_false_positive():
    labels = {"c": "clean", "b": "buggy"}
    verdicts = {"c": "UNKNOWN", "b": "UNKNOWN"}
    s = lb._score(verdicts, labels)
    assert s["fp"] == 0          # abstain on clean is not a false alarm
    assert s["fn"] == 1          # abstain on buggy is a miss
    assert s["abstain"] == 2


def test_tensorguard_is_reference_and_perfect_precision():
    data = lb.measure()
    ref = next(e for e in data["entries"] if e["source"] == "reference")
    assert ref["name"] == "TensorGuard"
    assert ref["scorecard"]["precision"] == 1.0
    assert ref["scorecard"]["fp"] == 0
    assert ref["scorecard"]["recall"] == 1.0


def test_entries_ranked_by_f1():
    data = lb.measure()
    f1s = [e["scorecard"]["f1"] for e in data["entries"]]
    assert f1s == sorted(f1s, reverse=True)
    assert data["entries"][0]["rank"] == 1


def test_community_entry_is_rescored_not_trusted():
    """The committed trivial-always-safe baseline must score recall 0 here,
    regardless of what its file claims."""
    data = lb.measure()
    names = {e["name"] for e in data["entries"]}
    assert "trivial-always-safe" in names
    triv = next(e for e in data["entries"]
                if e["name"] == "trivial-always-safe")
    assert triv["source"] == "community"
    assert triv["scorecard"]["recall"] == 0.0
    assert triv["scorecard"]["tp"] == 0


def test_corpus_fingerprint_is_stable_64hex():
    data = lb.measure()
    fp = data["corpus"]["fingerprint_sha256"]
    assert len(fp) == 64 and all(c in "0123456789abcdef" for c in fp)


def test_artifact_is_byte_deterministic():
    assert lb.run(check=True) == 0


def test_signed_submission_policy_and_cadence_are_published():
    data = lb.measure()
    assert data["scorer_version"] == "leaderboard-v2-signed-submissions"
    assert data["refresh_policy"]["cadence"] == "monthly"
    assert data["refresh_policy"]["workflow"] == ".github/workflows/leaderboard.yml"
    assert data["submission_policy"]["signature_namespace"] == (
        "tensorguard-leaderboard-v1"
    )
    assert data["submission_policy"]["trust_anchor"].endswith("allowed_signers")
    assert any("self-reported metrics" in r for r in data["anti_overfitting_rules"])


def test_rendered_markdown_documents_cadence_and_overfitting_rules():
    md = lb.render_markdown(lb.measure())
    assert "Refresh cadence" in md
    assert "SSH-signed" in md
    assert "Anti-overfitting rules" in md


def test_artifact_has_no_volatile_fields():
    data = json.loads(lb.OUT_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        for tok in VOLATILE_TOKENS:
            assert tok not in low, f"volatile key token {tok!r} in {key!r}"
