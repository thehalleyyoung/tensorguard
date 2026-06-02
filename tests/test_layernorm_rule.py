"""Step 155 — **nn.LayerNorm rule**, machine-checked in Lean and cross-checked
against real torch and ``_propagate_layernorm``.

``lean/TensorGuard/LayerNormRule.lean`` proves shape & numel preservation, the
suffix-length split, the trailing suffix match against ``normalized_shape`` and
the trailing-mismatch refutation.  This test replays those laws on **real**
``nn.LayerNorm`` modules and confirms a wrong trailing dim makes torch raise
exactly when the Lean guard flags it.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "LayerNormRule.lean")

_THEOREMS = [
    "TensorGuard.LayerNormRule.ln_preserves",
    "TensorGuard.LayerNormRule.ln_numel",
    "TensorGuard.LayerNormRule.ln_length",
    "TensorGuard.LayerNormRule.ln_suffix_match",
    "TensorGuard.LayerNormRule.ln_mismatch_flagged",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.LayerNormRule" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def _ld(normalized_shape):
    from src.model_checker import LayerDef, LayerKind
    return LayerDef(attr_name="ln", kind=LayerKind.LAYERNORM,
                    params={"normalized_shape": normalized_shape})


def _ts(*dims):
    from src.tensor_shapes import ShapeDim, TensorShape
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


def test_layernorm_preserves_shape_real_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_layernorm

    checked = 0
    for prefix in [(2,), (3, 5), (2, 4)]:
        for ns in [(8,), (4, 8)]:
            shape = prefix + ns
            mod = nn.LayerNorm(list(ns))
            real = list(mod(torch.zeros(*shape)).shape)
            pred, err = _propagate_layernorm(_ts(*shape), _ld(list(ns)))
            assert err is None
            got = [d.value for d in pred.dims]
            assert got == real == list(shape)     # ln_preserves
            checked += 1
    assert checked > 0


def test_layernorm_trailing_mismatch_refuted_like_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_layernorm

    mod = nn.LayerNorm(8)
    with pytest.raises(RuntimeError):
        mod(torch.zeros(2, 7))            # last dim 7 != normalized 8
    pred, err = _propagate_layernorm(_ts(2, 7), _ld(8))
    assert err is not None and pred is None   # ln_mismatch_flagged direction
