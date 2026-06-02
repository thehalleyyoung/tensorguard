"""Step 151 — **nn.Linear shape rule**, machine-checked in Lean and cross-checked
against the real torch engine and the verifier's own propagator.

``lean/TensorGuard/LinearRule.lean`` models ``nn.Linear`` as ``prefix ++ [in] ↦
prefix ++ [out]`` and proves rank preservation, the trailing dim = out_features,
prefix preservation, numel scaling, and the in_features guard.  This test replays
each law on a **real** ``nn.Linear`` and against ``_propagate_linear``, and
confirms a wrong last dim makes torch raise exactly when the Lean guard flags it.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "LinearRule.lean")

_THEOREMS = [
    "TensorGuard.LinearRule.lin_rank",
    "TensorGuard.LinearRule.lin_last",
    "TensorGuard.LinearRule.lin_prefix",
    "TensorGuard.LinearRule.lin_numel",
    "TensorGuard.LinearRule.linValid_iff",
    "TensorGuard.LinearRule.mismatch_flagged",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.LinearRule" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def _ld(in_f, out_f):
    from src.model_checker import LayerDef, LayerKind
    return LayerDef(attr_name="fc", kind=LayerKind.LINEAR, in_features=in_f,
                    out_features=out_f,
                    params={"in_features": in_f, "out_features": out_f})


def _ts(*dims):
    from src.tensor_shapes import ShapeDim, TensorShape
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


def test_linear_matches_real_torch_and_verifier():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_linear

    checked = 0
    for prefix in [(4,), (2, 3), (1, 5, 7)]:
        for in_f, out_f in [(8, 5), (16, 16), (10, 1), (3, 12)]:
            mod = nn.Linear(in_f, out_f)
            x = torch.zeros(*prefix, in_f)
            real = list(mod(x).shape)
            pred, err = _propagate_linear(_ts(*prefix, in_f), _ld(in_f, out_f))
            assert err is None
            got = [d.value for d in pred.dims]
            assert got == real
            # Lean laws: rank preserved, last dim = out_features, prefix kept,
            # numel scales by out/in.
            assert len(got) == len(prefix) + 1            # lin_rank
            assert got[-1] == out_f                       # lin_last
            assert got[:-1] == list(prefix)               # lin_prefix
            num = 1
            for d in prefix:
                num *= d
            assert _prod(got) == num * out_f              # lin_numel
            checked += 1
    assert checked > 0


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def test_linear_feature_mismatch_refuted_like_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_linear

    mod = nn.Linear(8, 5)
    # Wrong last dim: torch raises (mismatch_flagged / linValid_iff direction).
    with pytest.raises(RuntimeError):
        mod(torch.zeros(2, 7))
    pred, err = _propagate_linear(_ts(2, 7), _ld(8, 5))
    assert err is not None and pred is None
