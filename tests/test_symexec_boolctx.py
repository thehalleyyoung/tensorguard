"""Sound detector — tensor used in a boolean context.

Using a tensor with more than one element (or zero elements) in a boolean
context — ``if t:`` / ``while t:`` / ``not t`` — raises ``RuntimeError``
("Boolean value of Tensor with more than one value is ambiguous").  The detector
reuses ``_known_numel`` and is sound: it fires only when the element count is
statically known and != 1, and abstains otherwise.

Empirically matched against torch: shapes ``(2,3)`` and ``(0,)`` raise; ``(1,)``
and ``()`` succeed — exactly the detector's predictions.
"""

from src.symexec.bugs import SymBugKind
from src.symexec.engine import analyze_source, analyze_file


def _kinds(src):
    return [b.kind.value for b in analyze_source(src).bugs]


def _main(body):
    return 'import torch\nif __name__ == "__main__":\n' + "".join(
        "    " + line + "\n" for line in body
    )


def test_if_multi_element_flagged():
    src = _main(["x = torch.randn(2, 3)", "if x:", "    pass"])
    assert "bool_on_nonscalar" in _kinds(src)


def test_while_multi_element_flagged():
    src = _main(["x = torch.ones(2, 2)", "while x:", "    break"])
    assert "bool_on_nonscalar" in _kinds(src)


def test_not_multi_element_flagged():
    src = _main(["x = torch.randn(2, 3)", "if not x:", "    pass"])
    assert "bool_on_nonscalar" in _kinds(src)


def test_empty_tensor_flagged():
    src = _main(["x = torch.zeros(0)", "if x:", "    pass"])
    assert "bool_on_nonscalar" in _kinds(src)


def test_scalar_clean():
    src = _main(["x = torch.randn(())", "if x:", "    pass"])
    assert "bool_on_nonscalar" not in _kinds(src)


def test_singleton_clean():
    src = _main(["x = torch.zeros(1)", "if x:", "    pass"])
    assert "bool_on_nonscalar" not in _kinds(src)


def test_abstains_on_unknown():
    src = "import torch\ndef f(x):\n    if x:\n        return 1\n    return 0\n"
    assert "bool_on_nonscalar" not in _kinds(src)


def test_message_and_fix():
    src = _main(["x = torch.randn(2, 3)", "if x:", "    pass"])
    bugs = [b for b in analyze_source(src).bugs if b.kind is SymBugKind.BOOL_ON_NONSCALAR]
    assert bugs
    assert "6" in bugs[0].message
    assert bugs[0].fix_suggestion


def test_kind_category():
    from src.symexec.bugs import _API_CATEGORY

    assert _API_CATEGORY[SymBugKind.BOOL_ON_NONSCALAR] == "TYPE_ERROR"


def test_corpus_fingerprint_unchanged():
    fp = analyze_file("tests/symexec_corpus/wild/matmul_dim_mismatch.py").fingerprint()
    assert fp == (
        "de466b6f54018384cb5b3c27b5b3f7be178001535d59bb785c46fdba83ead9e0"
    )
