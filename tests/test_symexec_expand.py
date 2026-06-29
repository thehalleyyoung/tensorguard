"""Sound detector — ``tensor.expand(*sizes)`` shape violations.

``Tensor.expand`` can only broadcast singleton dims and keep/extend others; it
raises ``RuntimeError`` when (a) given fewer sizes than the tensor's rank, (b) an
aligned existing non-singleton dim is asked to become a different concrete size
(not ``-1``), or (c) a *new* leading dimension is given the ``-1`` placeholder.
The detector is sound — every case fires only on statically-known constants and
abstains otherwise.  Verified against torch: for ``x = randn(3, 1)``,
``x.expand(4, 1)`` / ``x.expand(6, -1)`` / ``x.expand(1)`` / ``x.expand(2, 4, 1)``
/ ``x.expand(-1, 3, 5)`` raise, while ``x.expand(3, 5)`` / ``x.expand(-1, 5)`` /
``x.expand(2, 3, 1)`` succeed.
"""

from src.symexec.bugs import SymBugKind
from src.symexec.config import SymConfig
from src.symexec.engine import analyze_source, analyze_file


def _kinds(src, cfg=None):
    r = analyze_source(src, config=cfg) if cfg is not None else analyze_source(src)
    return [b.kind.value for b in r.bugs]


def _fn(call):
    return f"import torch\ndef f():\n    x = torch.randn(3, 1)\n    return x.{call}\n"


def test_nonsingleton_mismatch_flagged():
    assert "expand_shape_mismatch" in _kinds(_fn("expand(4, 1)"))


def test_mismatch_with_keep_placeholder_flagged():
    # 3 -> 6 is a real mismatch even though the other dim uses -1.
    assert "expand_shape_mismatch" in _kinds(_fn("expand(6, -1)"))


def test_too_few_sizes_flagged():
    assert "expand_shape_mismatch" in _kinds(_fn("expand(1)"))


def test_leading_new_dim_mismatch_flagged():
    assert "expand_shape_mismatch" in _kinds(_fn("expand(2, 4, 1)"))


def test_leading_minus_one_flagged():
    assert "expand_shape_mismatch" in _kinds(_fn("expand(-1, 3, 5)"))


def test_tuple_form_flagged():
    assert "expand_shape_mismatch" in _kinds(_fn("expand((4, 1))"))


def test_exact_match_clean():
    assert "expand_shape_mismatch" not in _kinds(_fn("expand(3, 5)"))


def test_keep_dim_clean():
    assert "expand_shape_mismatch" not in _kinds(_fn("expand(-1, 5)"))


def test_add_leading_dim_clean():
    assert "expand_shape_mismatch" not in _kinds(_fn("expand(2, 3, 1)"))


def test_singleton_expands_clean():
    # The size-1 dim may expand to anything; the size-3 dim is kept.
    assert "expand_shape_mismatch" not in _kinds(_fn("expand(3, 7)"))


def test_abstains_on_unknown_rank():
    src = "import torch\ndef f(x):\n    return x.expand(4, 1)\n"
    assert "expand_shape_mismatch" not in _kinds(src)


def test_fires_in_sound_mode():
    assert "expand_shape_mismatch" in _kinds(_fn("expand(4, 1)"), SymConfig.sound())


def test_message_and_fix():
    bugs = [
        b for b in analyze_source(_fn("expand(4, 1)")).bugs
        if b.kind is SymBugKind.EXPAND_SHAPE_MISMATCH
    ]
    assert bugs
    assert bugs[0].fix_suggestion


def test_kind_category():
    from src.symexec.bugs import _API_CATEGORY

    assert _API_CATEGORY[SymBugKind.EXPAND_SHAPE_MISMATCH] == "TYPE_ERROR"


def test_corpus_fingerprint_unchanged():
    fp = analyze_file("tests/symexec_corpus/wild/matmul_dim_mismatch.py").fingerprint()
    assert fp == (
        "de466b6f54018384cb5b3c27b5b3f7be178001535d59bb785c46fdba83ead9e0"
    )
