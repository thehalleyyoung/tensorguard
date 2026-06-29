"""Step (even_more #4, dtype/device follow-on) — ``.item()`` on a non-scalar.

``tensor.item()`` requires the tensor to hold exactly one element; otherwise
PyTorch raises ``RuntimeError`` ("a Tensor with N elements cannot be converted
to Scalar").  The detector is sound: it fires only when the element count is
statically known and is not 1, and abstains on any unknown/symbolic dim.

The empirical truth table (verified against torch): shapes ``(2,3)`` and ``(0,)``
raise; ``(1,1)`` and ``()`` succeed — exactly what the detector predicts.
"""

from src.symexec.bugs import SymBugKind
from src.symexec.engine import analyze_source, analyze_file


def _kinds(src):
    return [b.kind.value for b in analyze_source(src).bugs]


def _main(body):
    return 'import torch\nif __name__ == "__main__":\n' + "".join(
        "    " + line + "\n" for line in body
    )


# ---- fires when numel is known and != 1 --------------------------------------

def test_fires_on_multi_element():
    src = _main(["a = torch.randn(2, 3)", "x = a.item()"])
    assert "item_on_nonscalar" in _kinds(src)


def test_fires_on_vector():
    src = _main(["a = torch.zeros(5)", "x = a.item()"])
    assert "item_on_nonscalar" in _kinds(src)


def test_fires_on_empty_tensor():
    # 0 elements also raises ("a Tensor with 0 elements ...")
    src = _main(["a = torch.zeros(0)", "x = a.item()"])
    assert "item_on_nonscalar" in _kinds(src)


# ---- clean when numel is exactly 1 -------------------------------------------

def test_clean_scalar():
    src = _main(["a = torch.randn(())", "x = a.item()"])
    assert "item_on_nonscalar" not in _kinds(src)


def test_clean_singleton_vector():
    src = _main(["a = torch.zeros(1)", "x = a.item()"])
    assert "item_on_nonscalar" not in _kinds(src)


def test_clean_singleton_matrix():
    src = _main(["a = torch.zeros(1, 1)", "x = a.item()"])
    assert "item_on_nonscalar" not in _kinds(src)


def test_clean_after_full_reduction():
    src = _main(["a = torch.randn(2, 3)", "x = a.sum().item()"])
    assert "item_on_nonscalar" not in _kinds(src)


# ---- soundness: abstain when the element count is not fully known ------------

def test_abstains_on_unknown_shape():
    src = (
        "import torch\n"
        "def f(a):\n"
        "    return a.item()\n"
    )
    assert "item_on_nonscalar" not in _kinds(src)


# ---- only the no-arg item() is modeled (item(*index) is a different op) -------

def test_indexed_item_not_flagged():
    # tensor.item(...) with an index argument is a distinct overload; we only
    # model the zero-arg scalar conversion, so this must not fire.
    src = _main(["a = torch.randn(2, 3)", "x = a.item(0)"])
    assert "item_on_nonscalar" not in _kinds(src)


# ---- message / category ------------------------------------------------------

def test_message_mentions_count_and_fix():
    src = _main(["a = torch.randn(2, 3)", "x = a.item()"])
    bugs = [b for b in analyze_source(src).bugs if b.kind is SymBugKind.ITEM_ON_NONSCALAR]
    assert bugs
    assert "6" in bugs[0].message
    assert bugs[0].fix_suggestion


def test_kind_category():
    from src.symexec.bugs import _API_CATEGORY

    assert _API_CATEGORY[SymBugKind.ITEM_ON_NONSCALAR] == "TYPE_ERROR"


# ---- corpus fingerprint is unaffected by the new kind ------------------------

def test_corpus_fingerprint_unchanged():
    fp = analyze_file("tests/symexec_corpus/wild/matmul_dim_mismatch.py").fingerprint()
    assert fp == (
        "de466b6f54018384cb5b3c27b5b3f7be178001535d59bb785c46fdba83ead9e0"
    )
