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
import shutil
import subprocess

from real_benchmarks.load import load_items
from reproducibility.leaderboard import _score
from reproducibility.validate_entry import (
    REPO,
    SIGNATURE_NAMESPACE,
    canonical_signed_payload,
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


@pytest.fixture()
def signed_entry(tmp_path):
    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen is required for SSH signature validation")

    def _make(raw, identity="demo"):
        key = tmp_path / f"{identity}_key"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", identity, "-f", str(key)],
            check=True,
        )
        payload = tmp_path / f"{identity}_payload.json"
        payload.write_bytes(canonical_signed_payload(raw))
        subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-q",
                "-f",
                str(key),
                "-n",
                SIGNATURE_NAMESPACE,
                str(payload),
            ],
            check=True,
        )
        allowed = tmp_path / f"{identity}_allowed_signers"
        public_key = " ".join((key.with_suffix(".pub")).read_text().split()[:2])
        allowed.write_text(f"{identity} {public_key}\n")
        signed = dict(raw)
        signed["signature"] = {
            "identity": identity,
            "namespace": SIGNATURE_NAMESPACE,
            "value": (tmp_path / f"{identity}_payload.json.sig").read_text(),
        }
        return signed, allowed

    return _make


def test_well_formed_signed_entry_validates(corpus_ids, signed_entry):
    a, b = _some_ids(corpus_ids)
    entry, allowed = signed_entry({"tool": "demo", "verdicts": {a: "UNSAFE", b: "SAFE"}})
    assert validate_entry(entry, corpus_ids, allowed_signers=allowed) == []


def test_unsigned_entry_rejected(corpus_ids):
    a = _some_ids(corpus_ids, 1)[0]
    problems = validate_entry({"tool": "demo", "verdicts": {a: "SAFE"}}, corpus_ids)
    assert any("signature" in p for p in problems)


def test_unknown_case_id_rejected(corpus_ids):
    entry = {"tool": "demo", "verdicts": {"definitely_not_a_real_id": "SAFE"}}
    problems = validate_entry(entry, corpus_ids, require_signature=False)
    assert any("unknown corpus case id" in p for p in problems)


def test_invalid_verdict_token_rejected(corpus_ids):
    a = _some_ids(corpus_ids, 1)[0]
    problems = validate_entry(
        {"tool": "demo", "verdicts": {a: "MAYBE"}},
        corpus_ids,
        require_signature=False,
    )
    assert any("invalid verdict" in p for p in problems)


def test_lowercase_verdict_rejected_before_signing(corpus_ids):
    a = _some_ids(corpus_ids, 1)[0]
    problems = validate_entry(
        {"tool": "demo", "verdicts": {a: "safe"}},
        corpus_ids,
        require_signature=False,
    )
    assert any("must be uppercase" in p for p in problems)


def test_missing_tool_and_verdicts_rejected(corpus_ids):
    assert any(
        "tool" in p
        for p in validate_entry({"verdicts": {}}, corpus_ids, require_signature=False)
    )
    assert any(
        "verdicts" in p
        for p in validate_entry({"tool": "x"}, corpus_ids, require_signature=False)
    )


def test_self_reported_metrics_rejected(corpus_ids):
    a = _some_ids(corpus_ids, 1)[0]
    entry = {"tool": "liar", "recall": 1.0, "f1": 1.0, "verdicts": {a: "UNSAFE"}}
    problems = validate_entry(entry, corpus_ids, require_signature=False)
    assert any("self-reported metric" in p for p in problems)


def test_committed_example_entry_validates(corpus_ids):
    import json

    path = REPO / "benchmarks" / "leaderboard_entries" / "trivial-always-safe.json"
    assert validate_entry(json.loads(path.read_text()), corpus_ids) == []


def test_signature_detects_tampered_verdict(corpus_ids, signed_entry):
    a, b = _some_ids(corpus_ids)
    entry, allowed = signed_entry({"tool": "demo", "verdicts": {a: "SAFE", b: "SAFE"}})
    entry["verdicts"][a] = "UNSAFE"
    problems = validate_entry(entry, corpus_ids, allowed_signers=allowed)
    assert any("invalid leaderboard signature" in p for p in problems)


def test_signature_requires_trusted_allowed_signer(corpus_ids, signed_entry, tmp_path):
    a = _some_ids(corpus_ids, 1)[0]
    entry, _allowed = signed_entry({"tool": "demo", "verdicts": {a: "SAFE"}})
    empty_allowed = tmp_path / "allowed_signers"
    empty_allowed.write_text("")
    problems = validate_entry(entry, corpus_ids, allowed_signers=empty_allowed)
    assert any("invalid leaderboard signature" in p for p in problems)


def test_workflow_and_issue_template_publish_refresh_process():
    workflow = (REPO / ".github" / "workflows" / "leaderboard.yml").read_text()
    template = (
        REPO / ".github" / "ISSUE_TEMPLATE" / "leaderboard_submission.md"
    ).read_text()
    assert "cron: \"17 9 1 * *\"" in workflow
    assert "Validate submitted entries" in workflow
    assert "Anti-overfitting attestation" in template
    assert "allowed_signers" in template


def test_lying_entry_is_rescored_to_truth(labels):
    # An always-SAFE tool: on a corpus with >=1 buggy case its true recall is 0,
    # no matter what it claims. The scorer ignores claims and recomputes.
    verdicts = {cid: "SAFE" for cid in labels}
    score = _score(verdicts, labels)
    assert any(v == "buggy" for v in labels.values())  # corpus has real bugs
    assert score["recall"] == 0.0
    assert score["fp"] == 0  # always-SAFE never raises a false alarm
