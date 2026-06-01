"""Step 23 -- precise einsum shape inference (parse the equation, not a heuristic).

The engine previously handled `torch.einsum` with an inline heuristic that only
understood explicit-output equations, ignored ellipsis, and could not detect
rank mismatches. Step 23 routes einsum through the canonical, torch-equivalent
parser in `src/smt/einsum_theory.py` and wires it into the verification engine
(`src/model_checker.py`) plus the fx extractor (which now captures the equation
string).

This module proves the new behaviour:
  * `infer_einsum_shape` is **differential-tested** against real `torch.einsum`
    over many equations (explicit/implicit output, diagonals, ellipsis
    broadcasting, reductions);
  * `check_einsum_compatible` accepts every valid equation (no false positives,
    including on symbolic dims) and rejects malformed/incompatible ones;
  * end-to-end, `verify_module` reports a buggy einsum (mismatched contraction
    dim) as unsafe while leaving valid einsum models safe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.tensor_shapes import ShapeDim, TensorShape
from src.smt.einsum_theory import (
    check_einsum_compatible,
    infer_einsum_shape,
    parse_einsum,
)
from src.fx_extractor import verify_module


def TS(*ds):
    return TensorShape(tuple(ShapeDim(d) for d in ds))


def _shp(t):
    return tuple(t.dims[i].value for i in range(t.ndim)) if t is not None else None


# ---- differential vs torch -----------------------------------------------
EQ_CASES = [
    ("bij,bjk->bik", [(2, 3, 4), (2, 4, 5)]),
    ("ij,jk->ik", [(3, 4), (4, 5)]),
    ("ij,jk", [(3, 4), (4, 5)]),                      # implicit output
    ("...ij,...jk->...ik", [(2, 3, 4), (2, 4, 5)]),
    ("...ij,...jk->...ik", [(6, 2, 3, 4), (6, 2, 4, 5)]),
    ("bhqd,bhkd->bhqk", [(2, 8, 7, 16), (2, 8, 5, 16)]),
    ("bi,bi->b", [(2, 3), (2, 3)]),
    ("ii->i", [(4, 4)]),                              # diagonal
    ("bij->bji", [(2, 3, 4)]),
    ("...i->...", [(2, 3, 4)]),                       # reduce last, keep ell.
    ("...i->i", [(2, 3, 4)]),                         # reduce ellipsis away
    ("...i,...i->...", [(5, 3), (5, 3)]),
    ("...,...->...", [(2, 1, 4), (2, 3, 4)]),         # broadcast size-1 ell.
    ("i...,i...->i...", [(2, 1, 4), (2, 3, 1)]),
    ("abc->cba", [(2, 3, 4)]),
    ("ij->", [(3, 3)]),                               # full reduction
]


@pytest.mark.parametrize("eq,shapes", EQ_CASES)
def test_infer_matches_torch(eq, shapes):
    ins = [TS(*s) for s in shapes]
    got = _shp(infer_einsum_shape(eq, ins))
    ts = [torch.zeros(*s) if s else torch.zeros(()) for s in shapes]
    exp = tuple(torch.einsum(eq, *ts).shape)
    assert got == exp, f"{eq}: got {got}, torch {exp}"
    # Valid equations must never be flagged.
    assert check_einsum_compatible(eq, ins) is None


def test_random_differential_against_torch():
    import random
    rng = random.Random(7)
    labels = "abcde"
    checked = 0
    for _ in range(800):
        nin = rng.randint(1, 2)
        sizes = {c: rng.randint(1, 4) for c in labels}
        specs, shapes = [], []
        for _ in range(nin):
            sub = "".join(rng.choice(labels) for _ in range(rng.randint(0, 3)))
            specs.append(sub)
            shapes.append(tuple(sizes[c] for c in sub))
        present = sorted(set("".join(specs)))
        rng.shuffle(present)
        out = "".join(present[:rng.randint(0, len(present))])
        eq = ",".join(specs) + "->" + out
        ins = [TS(*s) if s else TensorShape(()) for s in shapes]
        ts = [torch.zeros(*s) if s else torch.zeros(()) for s in shapes]
        try:
            exp = tuple(torch.einsum(eq, *ts).shape)
        except Exception:
            continue
        assert _shp(infer_einsum_shape(eq, ins)) == exp, eq
        assert check_einsum_compatible(eq, ins) is None, eq
        checked += 1
    assert checked > 200


# ---- validation / error detection ----------------------------------------
def test_contraction_mismatch_detected():
    assert "mismatched" in check_einsum_compatible("ij,jk->ik", [TS(3, 4), TS(5, 6)])
    assert infer_einsum_shape("ij,jk->ik", [TS(3, 4), TS(5, 6)]) is None


def test_rank_mismatch_detected():
    err = check_einsum_compatible("bij,bjk->bik", [TS(2, 3), TS(2, 4, 5)])
    assert "expected" in err


@pytest.mark.parametrize("eq", [
    "i->ii",        # repeated output label
    "i->j",         # output label not in inputs
    "ij->i->j",     # multiple arrows
    "i1->i",        # non-letter label
])
def test_malformed_or_invalid_rejected(eq):
    # Use a generic single/double input; the malformed ones should error on
    # parse regardless of shapes.
    shapes = [TS(3, 3)] if eq.split("->")[0].count(",") == 0 else [TS(3), TS(3)]
    try:
        n_inputs = len(parse_einsum(eq).operands)
    except ValueError:
        # Parse-time rejection is acceptable.
        assert check_einsum_compatible(eq, shapes) is not None
        return
    shapes = [TS(3) for _ in range(n_inputs)]
    assert check_einsum_compatible(eq, shapes) is not None


def test_ellipsis_not_broadcastable_detected():
    err = check_einsum_compatible("...i,...i->...i", [TS(2, 3), TS(4, 3)])
    assert err is not None and "broadcast" in err


def test_symbolic_dims_no_false_positive():
    sym = TensorShape((ShapeDim("n"), ShapeDim(4)))
    assert check_einsum_compatible("ij,jk->ik", [sym, TS(4, 5)]) is None
    out = infer_einsum_shape("ij,jk->ik", [sym, TS(4, 5)])
    assert [d.value for d in out.dims] == ["n", 5]


# ---- end-to-end through the engine ---------------------------------------
class _GoodAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16)
        self.k = nn.Linear(16, 16)

    def forward(self, x):
        return torch.einsum("bsd,btd->bst", self.q(x), self.k(x))


class _BadAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 8)
        self.k = nn.Linear(16, 4)  # contraction dim 8 vs 4 -> bug

    def forward(self, x):
        return torch.einsum("bsd,btd->bst", self.q(x), self.k(x))


class _EllipsisAttn(nn.Module):
    def forward(self, x):
        return torch.einsum("...qd,...kd->...qk", x, x)


def test_engine_flags_bad_einsum():
    r = verify_module(_BadAttn(), input_shapes={"x": (2, 5, 16)}, backend="fx")
    assert r.safe is False
    msgs = [v.message for v in r.counterexample.violations]
    assert any("einsum" in m and "mismatched" in m for m in msgs), msgs


def test_engine_accepts_good_einsum():
    r = verify_module(_GoodAttn(), input_shapes={"x": (2, 5, 16)}, backend="fx")
    assert r.safe is True


def test_engine_accepts_ellipsis_einsum():
    r = verify_module(_EllipsisAttn(), input_shapes={"x": (2, 4, 7, 16)},
                      backend="fx")
    assert r.safe is True
