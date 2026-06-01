"""Regression tests for the soundness contract (100_STEPS.md Step 5).

These pin three things:
  1. SOUNDNESS_CONTRACT.md is in sync with src/soundness_contract.py;
  2. the contract is internally complete (every UnsupportedCategory has a
     SKIPPED clause; every clause has a real evidence string);
  3. the contract's empirical claims hold against real code — in particular
     the verifiable-fragment boundary and the documented KNOWN_UNSOUNDNESS
     gap U1 (out-of-fragment modules currently receive a silent SAFE through
     verify_architecture). Pinning the *current* (gap) behaviour keeps the
     contract honest; when Step 8 fixes U1 this test is updated in lockstep.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src import soundness_contract as sc
from src.api import verify_architecture
from src.verifiable_fragment import UnsupportedCategory, check_traceability

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "SOUNDNESS_CONTRACT.md"

PRE = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"


def test_doc_in_sync_with_module():
    assert DOC.exists(), "SOUNDNESS_CONTRACT.md must be committed"
    expected = sc.render_markdown()
    actual = DOC.read_text()
    assert actual.strip() == expected.strip(), (
        "SOUNDNESS_CONTRACT.md is stale; regenerate with "
        "`python -m src.soundness_contract > SOUNDNESS_CONTRACT.md`"
    )


def test_every_unsupported_category_has_a_skipped_clause():
    skipped = [c for c in sc.OUT_OF_FRAGMENT_CLAUSES
               if c.soundness_class is sc.SoundnessClass.SKIPPED]
    covered = " ".join(c.construct + c.evidence for c in skipped)
    for cat in UnsupportedCategory:
        assert cat.name in covered, f"no SKIPPED clause for {cat.name}"


def test_every_clause_has_rationale_and_evidence():
    for c in sc.all_clauses():
        assert c.construct and c.rationale and c.evidence
        assert c.soundness_class in sc.SoundnessClass


def test_guarantee_mentions_both_directions():
    g = sc.SOUNDNESS_GUARANTEE
    assert "Refutation soundness" in g
    assert "Verification soundness" in g
    assert "V_TG" in g


def test_fragment_boundary_is_real():
    """A clean MLP is in-fragment; a data-dependent branch is out."""
    class Clean(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(10, 10)

        def forward(self, x):
            return self.lin(x)

    class DDCF(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(10, 10)

        def forward(self, x):
            if x.sum() > 0:
                return self.lin(x)
            return x

    assert check_traceability(Clean()).in_verifiable_fragment is True
    ddcf = check_traceability(DDCF())
    assert ddcf.in_verifiable_fragment is False
    assert UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW in {
        c.category for c in ddcf.blocking_issues
    }


def test_known_unsoundness_U1_is_currently_real():
    """KNOWN_UNSOUNDNESS U1: verify_architecture does NOT yet gate on the
    fragment, so an out-of-fragment module gets a silent SAFE. This asserts
    the gap is honestly documented (and not already silently fixed)."""
    u1 = next(g for g in sc.KNOWN_UNSOUNDNESS if g.id == "U1")
    assert "abstain" in u1.description.lower()
    assert "silent pass" in u1.remediation.lower()

    ddcf = PRE + (
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.lin = nn.Linear(10, 10)\n"
        "    def forward(self, x):\n"
        "        if x.sum() > 0:\n"
        "            return self.lin(x)\n"
        "        return x\n"
    )
    res = verify_architecture(ddcf, input_shapes={"x": (4, 10)},
                              filename="<contract>")
    # Documented current behaviour: silent SAFE, no abstain (the U1 gap).
    assert res.status == "SAFE"
    assert res.bug_count == 0


def test_refutation_soundness_probe():
    """A real in-fragment shape bug is reported (refutation direction)."""
    buggy = PRE + (
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.lin = nn.Linear(512, 10)\n"
        "    def forward(self, x):\n"
        "        return self.lin(x)\n"
    )
    res = verify_architecture(buggy, input_shapes={"x": (4, 768)},
                              filename="<contract>")
    assert res.status == "UNSAFE"
    assert res.bug_count > 0
