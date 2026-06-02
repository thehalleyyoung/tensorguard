"""Step 85 — validate the PyTorch RFC / governance proposal is complete.

The RFC is a deliverable that argues for incubation; these tests guard that it
keeps the sections an RFC reviewer expects (motivation, soundness framing,
governance + a concrete maintenance plan) and does not regress into a stub.
"""

from __future__ import annotations

import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RFC = os.path.join(_REPO, "docs", "RFC_pytorch_companion.md")


def _read() -> str:
    with open(_RFC, "r", encoding="utf-8") as fh:
        return fh.read()


def test_rfc_exists_and_is_substantial():
    assert os.path.exists(_RFC)
    assert os.path.getsize(_RFC) > 3000


def test_rfc_has_required_sections():
    text = _read().lower()
    for section in (
        "summary",
        "motivation",
        "why this belongs with pytorch",
        "proposed scope",
        "design overview",
        "governance & maintenance plan",
        "compatibility & risks",
        "adoption plan",
        "alternatives considered",
    ):
        assert section in text, f"RFC missing section: {section}"


def test_rfc_frames_soundness_and_abstain():
    text = _read().lower()
    assert "sound" in text
    assert "abstain" in text


def test_rfc_maintenance_plan_references_governance_artifacts():
    text = _read()
    for ref in ("GOVERNANCE.md", "MAINTAINERS.md", "DEPRECATION_POLICY.md",
                "SECURITY.md"):
        assert ref in text, f"maintenance plan should cite {ref}"


def test_rfc_targets_pytorch_labs_incubation():
    text = _read().lower()
    assert "pytorch-labs" in text
    assert "torch.compile" in text and "torch.export" in text
