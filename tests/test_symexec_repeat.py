"""Sound detector — ``tensor.repeat(*sizes)`` with too few repeat dims.

``Tensor.repeat`` requires the number of supplied repeat dims to be **at least**
the tensor's rank; fewer raises ``RuntimeError`` ("Number of dimensions of repeat
dims can not be smaller than number of dimensions of tensor").  The detector is
sound — it fires only when the rank is statically known and the number of
provided dims (separate positionals, or a single tuple/list of statically known
length) is strictly less than it; it abstains on unknown rank or an
unknown-length dim sequence.  Verified against torch: ``randn(2, 3).repeat(2)``
and ``.repeat((2,))`` raise; ``.repeat(2, 2)`` and ``.repeat(4, 2, 3)`` succeed.
"""

from src.symexec.bugs import SymBugKind
from src.symexec.config import SymConfig
from src.symexec.engine import analyze_source, analyze_file


def _kinds(src, cfg=None):
    r = analyze_source(src, config=cfg) if cfg is not None else analyze_source(src)
    return [b.kind.value for b in r.bugs]


def _fn(body):
    return "import torch\ndef f():\n" + "".join(
        "    " + line + "\n" for line in body
    )


def test_too_few_positional_dims_flagged():
    src = _fn(["x = torch.randn(2, 3)", "return x.repeat(2)"])
    assert "repeat_dims_too_few" in _kinds(src)


def test_too_few_dims_tuple_form_flagged():
    src = _fn(["x = torch.randn(2, 3)", "return x.repeat((2,))"])
    assert "repeat_dims_too_few" in _kinds(src)


def test_too_few_dims_list_form_flagged():
    src = _fn(["x = torch.randn(2, 3)", "return x.repeat([2])"])
    assert "repeat_dims_too_few" in _kinds(src)


def test_exact_rank_clean():
    src = _fn(["x = torch.randn(2, 3)", "return x.repeat(2, 2)"])
    assert "repeat_dims_too_few" not in _kinds(src)


def test_more_dims_clean():
    src = _fn(["x = torch.randn(2, 3)", "return x.repeat(4, 2, 3)"])
    assert "repeat_dims_too_few" not in _kinds(src)


def test_rank1_one_dim_clean():
    src = _fn(["x = torch.randn(5)", "return x.repeat(3)"])
    assert "repeat_dims_too_few" not in _kinds(src)


def test_abstains_on_unknown_rank():
    src = "import torch\ndef f(x):\n    return x.repeat(2)\n"
    assert "repeat_dims_too_few" not in _kinds(src)


def test_no_sizes_abstains():
    src = _fn(["x = torch.randn(2, 3)", "return x.repeat()"])
    assert "repeat_dims_too_few" not in _kinds(src)


def test_fires_in_sound_mode():
    src = _fn(["x = torch.randn(2, 3)", "return x.repeat(2)"])
    assert "repeat_dims_too_few" in _kinds(src, SymConfig.sound())


def test_message_and_fix():
    src = _fn(["x = torch.randn(2, 3, 4)", "return x.repeat(2)"])
    bugs = [
        b for b in analyze_source(src).bugs
        if b.kind is SymBugKind.REPEAT_DIMS_TOO_FEW
    ]
    assert bugs
    assert "rank 3" in bugs[0].message
    assert bugs[0].fix_suggestion


def test_einops_repeat_not_misfired():
    # einops.repeat(tensor, "h w -> h w c", c=3) is a different API: a literal
    # pattern with '->'.  It must not trip the tensor-method repeat check.
    src = (
        "import torch\nfrom einops import repeat\n"
        "def f():\n"
        "    x = torch.randn(2, 3)\n"
        "    return repeat(x, 'h w -> h w c', c=4)\n"
    )
    assert "repeat_dims_too_few" not in _kinds(src)


def test_kind_category():
    from src.symexec.bugs import _API_CATEGORY

    assert _API_CATEGORY[SymBugKind.REPEAT_DIMS_TOO_FEW] == "TYPE_ERROR"


def test_corpus_fingerprint_unchanged():
    fp = analyze_file("tests/symexec_corpus/wild/matmul_dim_mismatch.py").fingerprint()
    assert fp == (
        "de466b6f54018384cb5b3c27b5b3f7be178001535d59bb785c46fdba83ead9e0"
    )
