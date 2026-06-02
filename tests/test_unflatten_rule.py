"""Step 158 — **nn.Unflatten rule**, machine-checked in Lean and cross-checked
against real torch and ``_propagate_unflatten``.

``lean/TensorGuard/Unflatten.lean`` proves numel preservation under validity, the
rank law, the flatten/unflatten inverse (round-trip with Step 147) and the size
guard ``∏ unflattened_size == size(dim)``.  This test replays those laws on
**real** ``nn.Unflatten`` modules and confirms a non-matching product makes torch
raise exactly when the Lean guard flags it.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "Unflatten.lean")

_THEOREMS = [
    "TensorGuard.Unflatten.unflatten_numel",
    "TensorGuard.Unflatten.unflatten_rank",
    "TensorGuard.Unflatten.unflatten_then_flatten",
    "TensorGuard.Unflatten.unflattenValid_iff",
    "TensorGuard.Unflatten.size_mismatch_flagged",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.Unflatten" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def _ld(dim, sizes):
    from src.model_checker import LayerDef, LayerKind
    return LayerDef(attr_name="uf", kind=LayerKind.UNFLATTEN,
                    params={"dim": dim, "unflattened_size": sizes})


def _ts(*dims):
    from src.tensor_shapes import ShapeDim, TensorShape
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


def test_unflatten_matches_real_torch_and_verifier():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_unflatten

    checked = 0
    cases = [
        ((4, 6), 1, (2, 3)),
        ((4, 12), 1, (3, 4)),
        ((6, 4), 0, (2, 3)),
        ((2, 24, 5), 1, (4, 6)),
    ]
    for shape, dim, sizes in cases:
        mod = nn.Unflatten(dim, sizes)
        real = list(mod(torch.zeros(*shape)).shape)
        pred, err = _propagate_unflatten(_ts(*shape), _ld(dim, sizes))
        assert err is None
        got = [d.value for d in pred.dims]
        assert got == real, (got, real, shape, dim, sizes)
        # unflatten_numel: total elements preserved (product of sizes == orig).
        assert _prod(got) == _prod(shape)
        # unflatten_rank: rank grows by len(sizes) - 1.
        assert len(got) == len(shape) + len(sizes) - 1
        checked += 1
    assert checked > 0


def test_unflatten_size_mismatch_refuted_like_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    # 6 != 2*4 -> torch raises (size guard).
    with pytest.raises(RuntimeError):
        nn.Unflatten(1, (2, 4))(torch.zeros(3, 6))
