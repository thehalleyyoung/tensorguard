"""Tests for the offline issue miner (Step 103)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from corpus_extended.issue_miner import mine_all, mine_fixture  # noqa: E402

DEMO_JSON = REPO / "reproducibility" / "issue_miner_demo.json"

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


def _by_id():
    return {c.issue_id: c for c in mine_all()}


def test_corroborated_buggy_in_allowlist_is_accepted():
    c = _by_id()["fixture-0001"]
    assert c.status == "accepted"
    assert c.label == "buggy"
    assert "corroborated" in c.reason


def test_corroborated_but_not_in_allowlist_is_only_proposed():
    c = _by_id()["fixture-0002"]
    assert c.status == "proposed"
    assert c.label == "buggy"


def test_feature_request_without_code_is_rejected():
    c = _by_id()["fixture-0003"]
    assert c.status == "rejected"
    assert "no python code" in c.reason


def test_unreproducible_claim_is_rejected():
    c = _by_id()["fixture-0004"]
    assert c.status == "rejected"
    assert "not reproduced" in c.reason


def test_nothing_buggy_is_accepted_without_reproduction():
    for c in mine_all():
        if c.status in ("accepted", "proposed"):
            # Must have been corroborated against real torch.
            assert c.reason.startswith("corroborated") or c.reason.endswith(
                "runs cleanly"
            )


def test_acceptance_requires_allowlist_membership():
    # With an empty allowlist, even corroborated candidates are only proposed.
    cands = mine_all(accepted=set())
    for c in cands:
        assert c.status != "accepted"


def test_acceptance_respects_custom_allowlist():
    cands = {c.issue_id: c for c in mine_all(accepted={"fixture-0002"})}
    assert cands["fixture-0002"].status == "accepted"
    # 0001 no longer in the (custom) allowlist -> demoted to proposed.
    assert cands["fixture-0001"].status == "proposed"


def test_demo_artifact_no_volatile_fields_and_deterministic():
    data = json.loads(DEMO_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"
    from reproducibility import issue_miner_demo as imd

    assert imd.run(check=True) == 0


def test_demo_artifact_safety_properties():
    data = json.loads(DEMO_JSON.read_text())
    assert data["all_corroborated_are_gated"] is True
    assert data["all_accepted_are_corroborated"] is True
    assert data["miner_is_offline"] is True
