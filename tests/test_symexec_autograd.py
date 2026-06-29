"""Step (even_more #5) — autograd in-place-on-leaf hazard.

An in-place op (``add_``, ``mul_``, …) applied directly to a tensor that is an
autograd *leaf* requiring grad raises ``RuntimeError`` ("a leaf Variable that
requires grad is being used in an in-place operation").  The detector is sound:
it fires only when both ``requires_grad`` and ``is_leaf`` are positively known to
be ``True``; it abstains otherwise, and never flags the permitted in-place ops
``requires_grad_`` / ``detach_``.

The allowlist of flagged ops was verified empirically against torch (every op in
``_INPLACE_AUTOGRAD_OPS`` raises on a leaf-grad tensor).
"""

from src.symexec.bugs import SymBugKind
from src.symexec.engine import analyze_source, analyze_file


def _kinds(src):
    return [b.kind.value for b in analyze_source(src).bugs]


def _main(body):
    return 'import torch\nif __name__ == "__main__":\n' + "".join(
        "    " + line + "\n" for line in body
    )


# ---- fires on an in-place op applied to a leaf requiring grad ----------------

def test_fires_on_add_inplace():
    src = _main(["x = torch.randn(3, requires_grad=True)", "x.add_(1)"])
    assert "inplace_on_leaf" in _kinds(src)


def test_fires_on_mul_inplace():
    src = _main(["x = torch.zeros(3, requires_grad=True)", "x.mul_(2)"])
    assert "inplace_on_leaf" in _kinds(src)


def test_fires_on_zero_inplace():
    src = _main(["x = torch.ones(2, 2, requires_grad=True)", "x.zero_()"])
    assert "inplace_on_leaf" in _kinds(src)


# ---- soundness: clean when the tensor is not a leaf -------------------------

def test_clean_on_non_leaf():
    # y is the output of an op, hence not a leaf -> in-place is allowed.
    src = _main([
        "x = torch.randn(3, requires_grad=True)",
        "y = x + 1",
        "y.add_(1)",
    ])
    assert "inplace_on_leaf" not in _kinds(src)


# ---- soundness: clean when the leaf does not require grad -------------------

def test_clean_on_plain_leaf():
    src = _main(["x = torch.randn(3)", "x.add_(1)"])
    assert "inplace_on_leaf" not in _kinds(src)


def test_clean_requires_grad_false():
    src = _main(["x = torch.randn(3, requires_grad=False)", "x.add_(1)"])
    assert "inplace_on_leaf" not in _kinds(src)


# ---- permitted in-place ops are never flagged -------------------------------

def test_detach_inplace_allowed():
    src = _main(["x = torch.randn(3, requires_grad=True)", "x.detach_()"])
    assert "inplace_on_leaf" not in _kinds(src)


def test_requires_grad_inplace_allowed():
    src = _main(["x = torch.randn(3, requires_grad=True)", "x.requires_grad_()"])
    assert "inplace_on_leaf" not in _kinds(src)


# ---- out-of-place form is fine ----------------------------------------------

def test_out_of_place_clean():
    src = _main(["x = torch.randn(3, requires_grad=True)", "x = x + 1"])
    assert "inplace_on_leaf" not in _kinds(src)


# ---- soundness: abstain when requires_grad is unknown -----------------------

def test_abstains_on_unknown_grad():
    src = (
        "import torch\n"
        "def f(x):\n"
        "    x.add_(1)\n"  # x is a parameter of unknown leaf/grad status
    )
    assert "inplace_on_leaf" not in _kinds(src)


# ---- message / category ------------------------------------------------------

def test_message_and_fix_present():
    src = _main(["x = torch.randn(3, requires_grad=True)", "x.add_(1)"])
    bugs = [b for b in analyze_source(src).bugs if b.kind is SymBugKind.INPLACE_ON_LEAF]
    assert bugs
    assert "add_" in bugs[0].message
    assert bugs[0].fix_suggestion


def test_kind_category():
    from src.symexec.bugs import _API_CATEGORY

    assert _API_CATEGORY[SymBugKind.INPLACE_ON_LEAF] == "TYPE_ERROR"


# ---- corpus fingerprint is unaffected by the new kind ------------------------

def test_corpus_fingerprint_unchanged():
    fp = analyze_file("tests/symexec_corpus/wild/matmul_dim_mismatch.py").fingerprint()
    assert fp == (
        "de466b6f54018384cb5b3c27b5b3f7be178001535d59bb785c46fdba83ead9e0"
    )
