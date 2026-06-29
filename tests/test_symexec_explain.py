"""Step 65 — ``--explain`` derivations.

The engine accumulates a provenance chain (source→…→sink), a concrete
counterexample, an optional algebraic certificate, and the 1-minimal failing
conditions into ``SymBug.evidence``.  These tests verify the on-demand
``--explain`` renderer that segments that evidence into a structured,
human-readable derivation — presentation-only, no analysis re-run.
"""

from __future__ import annotations

from src.symexec.bugs import SymBug, SymBugKind
from src.symexec.explain import Explanation, explain_bug, explain_bugs
from src.symexec.engine import analyze_source


def _bug(evidence=None, **kw):
    base = dict(
        kind=SymBugKind.BROADCAST_MISMATCH,
        message="tensors cannot broadcast",
        line=5,
        col=11,
        function="f",
        confidence=0.94,
        fix_suggestion="align the operand shapes",
        evidence=evidence,
    )
    base.update(kw)
    return SymBug(**base)


# -- evidence segmentation ----------------------------------------------


def test_derivation_chain_is_parsed():
    ev = "L3: torch.zeros(3) → L4: torch.zeros(2) → L5: add"
    exp = explain_bug(_bug(evidence=ev))
    assert exp.derivation == [
        "L3: torch.zeros(3)",
        "L4: torch.zeros(2)",
        "L5: add",
    ]


def test_counterexample_section_is_parsed():
    ev = "concrete counterexample: shapes (3) and (2) cannot broadcast"
    exp = explain_bug(_bug(evidence=ev))
    assert exp.counterexample == "shapes (3) and (2) cannot broadcast"
    assert exp.certificate is None


def test_certificate_section_is_parsed():
    ev = "certified counterexample: a.shape=(2, 3) @ b.shape=(4, 5) (inner 3 ≠ 4)"
    exp = explain_bug(_bug(kind=SymBugKind.MATMUL_DIM_MISMATCH, evidence=ev))
    assert exp.certificate == "a.shape=(2, 3) @ b.shape=(4, 5) (inner 3 ≠ 4)"
    assert exp.counterexample is None


def test_minimal_conditions_section_is_parsed():
    ev = "concrete counterexample: shapes (a) and (b); minimal failing conditions: a != b ∧ a != 1 ∧ b != 1"
    exp = explain_bug(_bug(evidence=ev))
    assert exp.counterexample == "shapes (a) and (b)"
    assert exp.minimal_conditions == "a != b ∧ a != 1 ∧ b != 1"


def test_certificate_and_provenance_combined():
    ev = "L2: torch.zeros(2, 3) → L3: torch.zeros(4, 5) | certified counterexample: inner 3 ≠ 4"
    exp = explain_bug(_bug(kind=SymBugKind.MATMUL_DIM_MISMATCH, evidence=ev))
    assert exp.derivation == ["L2: torch.zeros(2, 3)", "L3: torch.zeros(4, 5)"]
    assert exp.certificate == "inner 3 ≠ 4"


def test_unrecognized_fragment_becomes_a_note():
    exp = explain_bug(_bug(evidence="some freeform evidence"))
    assert exp.notes == ["some freeform evidence"]
    assert exp.derivation == []


def test_no_evidence_yields_empty_sections():
    exp = explain_bug(_bug(evidence=None))
    assert exp.derivation == []
    assert exp.counterexample is None
    assert exp.certificate is None
    assert exp.minimal_conditions is None
    assert exp.notes == []


# -- rendering ----------------------------------------------------------


def test_render_includes_all_sections():
    ev = (
        "L3: torch.zeros(3) → L5: add; "
        "concrete counterexample: shapes (3) and (2); "
        "minimal failing conditions: a != b"
    )
    text = explain_bug(_bug(evidence=ev), filename="m.py").render()
    assert "[BROADCAST_MISMATCH] tensors cannot broadcast" in text
    assert "at m.py:5:11 in f" in text
    assert "confidence: 0.94" in text
    assert "derivation:" in text
    assert "    L3: torch.zeros(3)" in text
    assert "counterexample: shapes (3) and (2)" in text
    assert "minimal failing conditions: a != b" in text
    assert "fix: align the operand shapes" in text


def test_render_minimal_bug_no_evidence():
    text = explain_bug(_bug(evidence=None), filename="m.py").render()
    assert "[BROADCAST_MISMATCH]" in text
    assert "derivation:" not in text
    assert "counterexample:" not in text


def test_explain_bugs_empty():
    assert explain_bugs([]) == "no bugs found."


def test_explain_bugs_joins_blocks():
    out = explain_bugs([_bug(line=1), _bug(line=2)], filename="m.py")
    assert out.count("[BROADCAST_MISMATCH]") == 2
    assert "\n\n" in out


# -- end-to-end ---------------------------------------------------------


def test_result_explain_renders_real_bug():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(3)\n"
        "    b = torch.zeros(2)\n"
        "    return a + b\n"
    )
    res = analyze_source(src, filename="demo.py")
    text = res.explain(filename="demo.py")
    assert "BROADCAST_MISMATCH" in text
    assert "demo.py:" in text
    assert "confidence:" in text


def test_result_explain_no_bugs():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(3, 4)\n"
        "    return a @ b\n"
    )
    res = analyze_source(src)
    assert res.explain() == "no bugs found."
