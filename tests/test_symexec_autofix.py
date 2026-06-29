"""Tests for verified auto-repair (even_more.md Tier 1, idea #2)."""

from __future__ import annotations

import ast

from src.symexec import (
    FixCandidate,
    VerifiedFix,
    propose_fix,
    repair,
    verify_fix,
)
from src.symexec.bugs import SymBug, SymBugKind
from src.symexec.engine import analyze_source

RESHAPE = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    x = torch.randn(2, 3)\n"
    "    y = x.reshape(5, 5)\n"
)
NEGDIM = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    x = torch.randn(2, -3)\n"
)
MATMUL = (  # no repair strategy registered
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    a = torch.randn(2, 3); b = torch.randn(4, 5); c = a @ b\n"
)


# --------------------------------------------------------------------------- #
# Proposal.                                                                    #
# --------------------------------------------------------------------------- #
def test_propose_reshape_flatten():
    bug = analyze_source(RESHAPE).bugs[0]
    cand = propose_fix(bug, RESHAPE)
    assert cand is not None
    assert cand.strategy == "reshape-flatten"
    assert ".reshape(-1)" in cand.patched_source
    ast.parse(cand.patched_source)  # still valid Python


def test_propose_negdim_abs():
    bug = analyze_source(NEGDIM).bugs[0]
    cand = propose_fix(bug, NEGDIM)
    assert cand is not None
    assert cand.strategy == "negdim-abs"
    assert "randn(2, 3)" in cand.patched_source
    assert "-3" not in cand.patched_source


def test_no_strategy_returns_none():
    bug = analyze_source(MATMUL).bugs[0]
    assert propose_fix(bug, MATMUL) is None


def test_propose_edit_is_line_local():
    bug = analyze_source(RESHAPE).bugs[0]
    cand = propose_fix(bug, RESHAPE)
    # only the offending line changed; line count preserved.
    before = RESHAPE.splitlines()
    after = cand.patched_source.splitlines()
    assert len(before) == len(after)
    changed = [i for i in range(len(before)) if before[i] != after[i]]
    assert changed == [bug.line - 1]


# --------------------------------------------------------------------------- #
# End-to-end repair (proposal + re-verification + diff).                       #
# --------------------------------------------------------------------------- #
def test_repair_reshape_is_verified():
    fixes = repair(RESHAPE, filename="m.py")
    assert len(fixes) == 1
    f = fixes[0]
    assert isinstance(f, VerifiedFix)
    assert f.verified
    assert f.kind == "reshape_size_mismatch"
    assert "reshape(-1)" in f.patched_source
    assert f.diff and "--- a/m.py" in f.diff and "+++ b/m.py" in f.diff
    # the patched source really is clean per the engine.
    assert analyze_source(f.patched_source).bugs == []


def test_repair_negdim_is_verified():
    fixes = repair(NEGDIM)
    assert len(fixes) == 1 and fixes[0].verified
    assert analyze_source(fixes[0].patched_source).bugs == []


def test_repair_returns_nothing_without_strategy():
    assert repair(MATMUL) == []


def test_repair_clean_source_is_empty():
    clean = (
        "import torch\n"
        "if __name__ == '__main__':\n"
        "    x = torch.randn(2, 3); y = x.reshape(6)\n"
    )
    assert repair(clean) == []


# --------------------------------------------------------------------------- #
# Re-verification gating (the core guarantee): reject bad candidates.          #
# --------------------------------------------------------------------------- #
def _bug(kind_value, line):
    return SymBug(kind=SymBugKind(kind_value), message="m", line=line, col=0,
                  function="")


def test_verify_rejects_when_target_still_fires():
    # patched source still contains the same reshape bug on the same line.
    cand = FixCandidate(
        kind="reshape_size_mismatch", line=4, strategy="noop",
        description="does nothing", patched_source=RESHAPE,
    )
    original = analyze_source(RESHAPE).bugs
    vf = verify_fix(cand, original, filename="m.py")
    assert not vf.verified
    assert "still fires" in vf.detail


def test_verify_rejects_when_new_bug_introduced():
    # original program has only a reshape bug; the "fix" removes it but the
    # patched source introduces a brand-new matmul bug.
    patched = (
        "import torch\n"
        "if __name__ == '__main__':\n"
        "    x = torch.randn(2, 3)\n"
        "    a = torch.randn(2, 3); b = torch.randn(4, 5); c = a @ b\n"
    )
    cand = FixCandidate(
        kind="reshape_size_mismatch", line=4, strategy="bad",
        description="introduces a new bug", patched_source=patched,
    )
    original = [_bug("reshape_size_mismatch", 4)]
    vf = verify_fix(cand, original, filename="m.py")
    assert not vf.verified
    assert "new bug kind" in vf.detail


def test_repair_unverified_only_flag_surfaces_rejections():
    # A source where the reshape flatten is fine, but we also show that
    # verified_only=False would include unverified candidates if any existed.
    fixes_all = repair(RESHAPE, verified_only=False)
    fixes_verified = repair(RESHAPE, verified_only=True)
    assert [f.verified for f in fixes_verified] == [True]
    assert len(fixes_all) >= len(fixes_verified)
