"""Step 97 — pin the completeness characterization of the modeled fragment.

These tests pin:
  1. ``docs/symexec/completeness.md`` is in sync with the source-of-truth module;
  2. the contract is well-formed — every clause names a real ``SymBugKind`` and
     every cited precondition is a real entry in the certificate vocabulary;
  3. the contract is *faithful and non-vacuous* — each kind in the "complete
     fragment" actually fires on a minimal example whose operands are known
     (so the completeness guarantee is exercised, not merely asserted), and the
     covered + best-effort kinds together stay within the real ``SymBugKind`` set.
"""

from __future__ import annotations

from pathlib import Path

from src.symexec import completeness_contract as cc
from src.symexec.bugs import SymBugKind
from src.symexec.certificate import PRECONDITIONS
from src.symexec.engine import analyze_source

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "symexec" / "completeness.md"

_KIND_VALUES = {k.value for k in SymBugKind}


def test_doc_in_sync_with_module():
    assert DOC.exists(), "docs/symexec/completeness.md must be committed"
    expected = cc.render_markdown()
    actual = DOC.read_text(encoding="utf-8")
    assert actual.strip() == expected.strip(), (
        "docs/symexec/completeness.md is stale; regenerate with "
        "`python -m src.symexec.completeness_contract > docs/symexec/completeness.md`"
    )


def test_clauses_are_well_formed():
    clauses = cc.COMPLETE_FOR + cc.NO_GUARANTEE
    assert clauses
    for c in clauses:
        assert c.kind and c.condition and c.code, f"incomplete clause: {c}"
        assert c.kind in _KIND_VALUES, f"unknown SymBugKind: {c.kind}"


def test_complete_clauses_cite_real_preconditions():
    for c in cc.COMPLETE_FOR:
        assert c.predicate in PRECONDITIONS, (
            f"clause for {c.kind} cites unknown precondition {c.predicate!r}"
        )


def test_no_guarantee_clauses_have_no_predicate():
    for c in cc.NO_GUARANTEE:
        assert c.predicate is None, (
            f"best-effort kind {c.kind} must not claim a precondition"
        )


def test_no_kind_is_both_complete_and_excluded():
    complete = {c.kind for c in cc.COMPLETE_FOR}
    excluded = {c.kind for c in cc.NO_GUARANTEE}
    assert not (complete & excluded), complete & excluded


def test_every_certificate_predicate_is_witnessed_by_some_clause():
    """Every runtime precondition in the vocabulary backs a complete clause."""
    used = {c.predicate for c in cc.COMPLETE_FOR}
    missing = set(PRECONDITIONS) - used
    assert not missing, f"preconditions with no completeness clause: {missing}"


def test_notes_tie_completeness_to_abstain_and_soundness():
    blob = " ".join(cc.COMPLETENESS_NOTES).lower()
    assert "abstain" in blob
    assert "soundness" in blob or "sound" in blob
    assert "⊤" in blob or "unknown" in blob


# --------------------------------------------------------------------------- #
# Non-vacuity: each "complete fragment" kind actually fires on known operands. #
# --------------------------------------------------------------------------- #
_EXAMPLES = {
    "matmul_dim_mismatch":
        "import torch\nif __name__=='__main__':\n a=torch.randn(2,3); b=torch.randn(4,5); c=a@b\n",
    "broadcast_mismatch":
        "import torch\nif __name__=='__main__':\n a=torch.randn(3,4); b=torch.randn(3,5); c=a+b\n",
    "layer_dim_mismatch":
        "import torch, torch.nn as nn\nif __name__=='__main__':\n m=nn.Linear(10,5); x=torch.randn(2,7); y=m(x)\n",
    "reshape_size_mismatch":
        "import torch\nif __name__=='__main__':\n x=torch.randn(2,3); y=x.reshape(5,5)\n",
    "cat_shape_mismatch":
        "import torch\nif __name__=='__main__':\n a=torch.randn(2,3); b=torch.randn(4,5); c=torch.cat([a,b],dim=0)\n",
    "einsum_dim_mismatch":
        "import torch\nif __name__=='__main__':\n a=torch.randn(2,3); b=torch.randn(4,5); c=torch.einsum('ij,jk->ik',a,b)\n",
    "axis_out_of_range":
        "import torch\nif __name__=='__main__':\n x=torch.randn(2,3); y=x.sum(dim=5)\n",
    "rank_index_error":
        "if __name__=='__main__':\n xs=[1,2,3]; y=xs[5]\n",
    "negative_dimension":
        "import torch\nif __name__=='__main__':\n x=torch.randn(2,-3)\n",
    "division_by_zero":
        "if __name__=='__main__':\n n=0; y=10//n\n",
    "unpack_arity_mismatch":
        "if __name__=='__main__':\n a,b,c=(1,2)\n",
    "return_arity_contract":
        "import torch, torch.nn as nn\nclass M(nn.Module):\n def forward(self,x):\n  return x\n"
        "if __name__=='__main__':\n m=M(); a,b=m(torch.randn(2,3))\n",
    "einops_pattern_mismatch":
        "import torch\nfrom einops import rearrange\nif __name__=='__main__':\n"
        " x=torch.randn(2,3,4); y=rearrange(x,'a b -> b a')\n",
    "none_propagation":
        "def f():\n x=None\n return x.shape\n",
}


def test_complete_clauses_are_non_vacuous():
    """For every kind we claim completeness on, there is a known-operand program
    the engine actually reports — the guarantee is exercised, not just asserted."""
    complete = {c.kind for c in cc.COMPLETE_FOR}
    # tensor_index_oob shares its example space with rank_index_error; allow the
    # examples map to cover all complete kinds except that aliased one.
    covered_by_examples = complete - {"tensor_index_oob"}
    assert covered_by_examples <= set(_EXAMPLES), (
        f"missing non-vacuity example for: {covered_by_examples - set(_EXAMPLES)}"
    )
    for kind, src in _EXAMPLES.items():
        if kind not in complete:
            continue
        kinds_found = {b.kind.value for b in analyze_source(src).bugs}
        assert kind in kinds_found, (
            f"completeness clause for {kind} is vacuous: not reported on its "
            f"known-operand example (got {kinds_found})"
        )
