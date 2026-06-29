"""Positive *"why is this safe?"* reports (even_more Tier 5 #14).

The safety report only re-presents facts the analysis already computed: the
verdict (was any *sound* forced-failure bug provable), the coverage profile, the
relative-completeness guarantee (the kinds whose absence-of-report is a positive
guarantee on the covered fragment), and the abstain ledger marking where that
guarantee stops.  These tests pin those invariants without asserting torch
behaviour.
"""

from src.symexec import (
    SafetyReport,
    explain_safety,
    render_safety_report,
    safety_report,
)
from src.symexec.completeness_contract import COMPLETE_FOR
from src.symexec.engine import analyze_source

_SAFE = "def f(a, b):\n return a + b\n"
_MATMUL_BUG = (
    "import torch\n"
    "if __name__=='__main__':\n"
    " a=torch.randn(2,3); b=torch.randn(4,5); c=a@b\n"
)


def test_clean_source_is_proven_safe():
    r = analyze_source(_SAFE)
    rep = safety_report(r, filename="clean.py")
    assert isinstance(rep, SafetyReport)
    assert rep.proven_safe is True
    assert rep.sound_bug_count == 0


def test_forced_failure_is_not_safe():
    r = analyze_source(_MATMUL_BUG)
    rep = safety_report(r, filename="bug.py")
    assert rep.proven_safe is False
    assert rep.sound_bug_count >= 1


def test_report_lists_all_complete_for_kinds():
    rep = safety_report(analyze_source(_SAFE))
    expected = {c.kind for c in COMPLETE_FOR}
    assert set(rep.complete_for_kinds) == expected
    # No duplicates in the rendered list.
    assert len(rep.complete_for_kinds) == len(set(rep.complete_for_kinds))


def test_fingerprint_matches_result():
    r = analyze_source(_SAFE)
    rep = safety_report(r)
    assert rep.fingerprint == r.fingerprint()


def test_render_safe_verdict_markdown():
    rep = safety_report(analyze_source(_SAFE), filename="clean.py")
    md = render_safety_report(rep)
    assert md.startswith("# Why is `clean.py` safe?")
    assert "No forced-failure bug was provable" in md
    assert "Soundness contract" in md
    assert md.endswith("\n")


def test_render_unsafe_verdict_markdown():
    md = explain_safety(analyze_source(_MATMUL_BUG), filename="bug.py")
    assert "Not safe" in md
    assert "sound forced-failure" in md


def test_result_safety_method_matches_helper():
    r = analyze_source(_SAFE)
    assert r.safety(filename="clean.py") == explain_safety(r, filename="clean.py")


def test_heuristic_only_finding_still_proven_safe():
    # A nn.Module missing super().__init__() is an intent/heuristic warning, not
    # a sound forced-failure: the safety verdict must stay "proven safe" while
    # surfacing the heuristic count.
    src = (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.x = 1\n"
        "    def forward(self, x):\n"
        "        return x\n"
    )
    from src.symexec.config import SymConfig

    cfg = SymConfig(enable_heuristics=True)
    r = analyze_source(src, config=cfg)
    rep = safety_report(r)
    if rep.heuristic_bug_count:
        assert rep.proven_safe is True
        md = render_safety_report(rep)
        assert "heuristic suspicion" in md


def test_abstain_table_present_when_abstained():
    rep = safety_report(analyze_source(_SAFE))
    md = render_safety_report(rep)
    if rep.abstain_total:
        assert "Where the guarantee stops" in md
        assert "| Abstain category | Count |" in md
    else:
        assert "did not abstain anywhere" in md


def test_to_dict_is_json_ready():
    import json

    rep = safety_report(analyze_source(_MATMUL_BUG), filename="bug.py")
    d = rep.to_dict()
    json.dumps(d)  # must not raise
    assert d["proven_safe"] is False
    assert d["filename"] == "bug.py"
    assert isinstance(d["complete_for_kinds"], list)
