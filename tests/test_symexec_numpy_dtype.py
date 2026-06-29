"""Sound autograd/dtype detectors.

`numpy_on_grad` — ``tensor.numpy()`` on a tensor that requires grad raises
``RuntimeError`` ("Can't call numpy() on Tensor that requires grad").  Sound:
fires only when ``requires_grad`` is positively known ``True``; ``.detach()``
clears the flag so ``x.detach().numpy()`` is never flagged.

`requires_grad_non_float` — a tensor constructor with ``requires_grad=True`` on
an integer/bool dtype raises ``RuntimeError`` ("Only Tensors of floating point
and complex dtype can require gradients").  Sound: fires only when the dtype is
a known integer/bool type and ``requires_grad`` is known ``True``.

Both verified against torch 2.x.
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


# ---- numpy_on_grad -----------------------------------------------------------

def test_numpy_on_grad_flagged():
    src = _main(["x = torch.randn(3, requires_grad=True)", "y = x.numpy()"])
    assert "numpy_on_grad" in _kinds(src)


def test_numpy_after_detach_clean():
    src = _main([
        "x = torch.randn(3, requires_grad=True)",
        "y = x.detach().numpy()",
    ])
    assert "numpy_on_grad" not in _kinds(src)


def test_numpy_without_grad_clean():
    src = _main(["x = torch.randn(3)", "y = x.numpy()"])
    assert "numpy_on_grad" not in _kinds(src)


def test_numpy_fires_in_sound_mode():
    src = _main(["x = torch.randn(3, requires_grad=True)", "y = x.numpy()"])
    assert "numpy_on_grad" in _kinds(src, SymConfig.sound())


def test_numpy_abstains_on_unknown():
    src = "import torch\ndef f(x):\n    return x.numpy()\n"
    assert "numpy_on_grad" not in _kinds(src)


def test_numpy_fix_suggestion():
    src = _main(["x = torch.randn(3, requires_grad=True)", "y = x.numpy()"])
    bugs = [b for b in analyze_source(src).bugs if b.kind is SymBugKind.NUMPY_ON_GRAD]
    assert bugs and bugs[0].fix_suggestion


# ---- requires_grad_non_float -------------------------------------------------

def test_int_dtype_requires_grad_flagged():
    src = _main(["x = torch.zeros(3, dtype=torch.long, requires_grad=True)"])
    assert "requires_grad_non_float" in _kinds(src)


def test_bool_dtype_requires_grad_flagged():
    src = _main(["x = torch.zeros(3, dtype=torch.bool, requires_grad=True)"])
    assert "requires_grad_non_float" in _kinds(src)


def test_float_dtype_requires_grad_clean():
    src = _main(["x = torch.zeros(3, dtype=torch.float32, requires_grad=True)"])
    assert "requires_grad_non_float" not in _kinds(src)


def test_int_dtype_without_requires_grad_clean():
    src = _main(["x = torch.zeros(3, dtype=torch.long)"])
    assert "requires_grad_non_float" not in _kinds(src)


def test_unknown_dtype_abstains():
    src = "import torch\ndef f(dt):\n    x = torch.zeros(3, dtype=dt, requires_grad=True)\n"
    assert "requires_grad_non_float" not in _kinds(src)


def test_dtype_fires_in_sound_mode():
    src = _main(["x = torch.zeros(3, dtype=torch.int64, requires_grad=True)"])
    assert "requires_grad_non_float" in _kinds(src, SymConfig.sound())


def test_dtype_kinds_category():
    from src.symexec.bugs import _API_CATEGORY

    assert _API_CATEGORY[SymBugKind.NUMPY_ON_GRAD] == "TYPE_ERROR"
    assert _API_CATEGORY[SymBugKind.REQUIRES_GRAD_NON_FLOAT] == "TYPE_ERROR"


def test_corpus_fingerprint_unchanged():
    fp = analyze_file("tests/symexec_corpus/wild/matmul_dim_mismatch.py").fingerprint()
    assert fp == (
        "de466b6f54018384cb5b3c27b5b3f7be178001535d59bb785c46fdba83ead9e0"
    )
