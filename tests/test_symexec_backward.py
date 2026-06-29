"""Sound detector — ``.backward()`` on a non-scalar tensor.

Calling ``tensor.backward()`` with no ``gradient`` argument requires a scalar
output: a non-scalar raises ``RuntimeError`` ("grad can be implicitly created
only for scalar outputs").  The detector is sound — it fires only when
``requires_grad`` is positively known to be ``True`` (otherwise a different
"does not require grad" error masks this one), the element count is statically
known and != 1, and no ``gradient`` argument was supplied.  It abstains
otherwise.  Verified against torch: ``randn(3, requires_grad=True).backward()``
raises; ``.sum().backward()`` and ``.backward(gradient=ones(3))`` succeed.
"""

from src.symexec.bugs import SymBugKind
from src.symexec.config import SymConfig
from src.symexec.engine import analyze_source, analyze_file


def _kinds(src, cfg=None):
    r = analyze_source(src, config=cfg) if cfg is not None else analyze_source(src)
    return [b.kind.value for b in r.bugs]


def _main(body):
    return 'import torch\nif __name__ == "__main__":\n' + "".join(
        "    " + line + "\n" for line in body
    )


def test_nonscalar_backward_flagged():
    src = _main(["x = torch.randn(3, requires_grad=True)", "x.backward()"])
    assert "backward_on_nonscalar" in _kinds(src)


def test_scalar_backward_clean():
    src = _main(["x = torch.randn((), requires_grad=True)", "x.backward()"])
    assert "backward_on_nonscalar" not in _kinds(src)


def test_explicit_gradient_kwarg_clean():
    src = _main([
        "x = torch.randn(3, requires_grad=True)",
        "x.backward(gradient=torch.ones(3))",
    ])
    assert "backward_on_nonscalar" not in _kinds(src)


def test_explicit_gradient_positional_clean():
    src = _main([
        "x = torch.randn(3, requires_grad=True)",
        "x.backward(torch.ones(3))",
    ])
    assert "backward_on_nonscalar" not in _kinds(src)


def test_abstains_without_requires_grad():
    src = _main(["x = torch.randn(3)", "x.backward()"])
    assert "backward_on_nonscalar" not in _kinds(src)


def test_abstains_on_unknown_shape():
    src = "import torch\ndef f(x):\n    x.backward()\n"
    assert "backward_on_nonscalar" not in _kinds(src)


def test_fires_in_sound_mode():
    src = _main(["x = torch.randn(3, requires_grad=True)", "x.backward()"])
    assert "backward_on_nonscalar" in _kinds(src, SymConfig.sound())


def test_message_and_fix():
    src = _main(["x = torch.randn(2, 3, requires_grad=True)", "x.backward()"])
    bugs = [
        b for b in analyze_source(src).bugs
        if b.kind is SymBugKind.BACKWARD_ON_NONSCALAR
    ]
    assert bugs and "6" in bugs[0].message
    assert bugs[0].fix_suggestion


def test_kind_category():
    from src.symexec.bugs import _API_CATEGORY

    assert _API_CATEGORY[SymBugKind.BACKWARD_ON_NONSCALAR] == "TYPE_ERROR"


def test_corpus_fingerprint_unchanged():
    fp = analyze_file("tests/symexec_corpus/wild/matmul_dim_mismatch.py").fingerprint()
    assert fp == (
        "de466b6f54018384cb5b3c27b5b3f7be178001535d59bb785c46fdba83ead9e0"
    )
