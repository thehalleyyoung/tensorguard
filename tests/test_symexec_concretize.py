"""Step 8 — concretization oracle (γ, α, force_counterexample).

The oracle supplies concrete representatives of abstract values and certifies
reports with concrete forced-failing inputs.  The defining soundness property is
the Galois law ``α(γ(v)) ⊑ v`` — every representative the oracle hands out is a
genuine member of the value's concretization.
"""

import pytest

from src.symexec import (
    analyze_source,
    SymBugKind,
    gamma,
    gamma_samples,
    alpha,
    force_counterexample,
    ConcreteTensor,
)
from src.symexec.concretize import is_sound_sample, ANY, NO_WITNESS
from src.symexec.values import (
    BoolVal,
    DictVal,
    IntVal,
    ListVal,
    NONE,
    TOP,
    BOTTOM,
    StrVal,
    TensorVal,
    TupleVal,
    int_const,
    int_range,
)
from src.symexec.symdim import SymDim


def _t(*sizes):
    return TensorVal(rank=len(sizes), shape=tuple(SymDim.const_dim(s) for s in sizes))


# ── γ produces genuine members: α(γ(v)) ⊑ v (soundness law) ─────────────────
@pytest.mark.parametrize(
    "value",
    [
        int_const(5),
        int_range(2, 4),
        int_range(-3, 0),
        NONE,
        BoolVal(),
        BoolVal(const=True),
        StrVal(const="hi"),
        _t(2, 3),
        TensorVal(rank=3),  # rank known, shape symbolic
        TupleVal(elems=(int_const(5), NONE), exact_len=True),
        ListVal(elem=int_const(7), length=2, exact_elems=(int_const(7), int_const(7))),
        DictVal(value=NONE, known=(("a", NONE),), exact_keys=True),
    ],
)
def test_gamma_samples_are_sound_members(value):
    samples = gamma_samples(value)
    assert samples, "expected at least one representative"
    for s in samples:
        assert is_sound_sample(value, s), f"{s!r} is not a member of γ({value!r})"


def test_int_range_enumerates_all_values():
    assert set(gamma_samples(int_range(2, 5))) == {2, 3, 4, 5}


def test_tensor_known_shape_concretizes_exactly():
    (c,) = gamma_samples(_t(2, 3))
    assert isinstance(c, ConcreteTensor) and c.shape == (2, 3)


def test_tensor_rank_only_uses_free_positive_dims():
    (c,) = gamma_samples(TensorVal(rank=2))
    assert c.rank == 2 and all(d >= 1 for d in c.shape)


# ── ⊤ / ⊥ corner cases ──────────────────────────────────────────────────────
def test_gamma_top_is_any():
    assert gamma(TOP) is ANY
    assert gamma_samples(TOP) == []


def test_gamma_bottom_raises():
    with pytest.raises(ValueError):
        gamma(BOTTOM)
    assert gamma_samples(BOTTOM) == []


# ── α round-trips the oracle's own representatives ──────────────────────────
def test_alpha_of_concrete_tensor_is_leq_original():
    v = _t(4, 5)
    assert is_sound_sample(v, gamma(v))
    assert alpha(gamma(v)).rank == 2


# ── force_counterexample certifies (or declines) a failing predicate ────────
def test_force_counterexample_finds_zero_in_range():
    # a division-by-zero is certified when 0 ∈ γ(divisor)
    w = force_counterexample(int_range(-2, 3), lambda x: x == 0)
    assert w == 0


def test_force_counterexample_declines_when_no_member_satisfies():
    w = force_counterexample(int_range(1, 5), lambda x: x == 0)
    assert w is NO_WITNESS


def test_force_counterexample_matmul_inner_mismatch():
    a, b = _t(2, 4), _t(7, 3)
    wa = force_counterexample(a, lambda c: True)
    wb = force_counterexample(b, lambda c: True)
    assert wa.shape[-1] != wb.shape[-2]  # 4 ≠ 7 — a real mismatch


# ── the oracle certifies a real report end-to-end ───────────────────────────
def test_matmul_report_carries_certified_counterexample():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 4)\n"
        "    a = torch.zeros(7, 3)\n"
        "    return x @ a\n"
    )
    bugs = [b for b in analyze_source(src, "m").bugs if b.kind == SymBugKind.MATMUL_DIM_MISMATCH]
    assert bugs, "expected a matmul mismatch"
    b = bugs[0]
    assert b.evidence is not None and "certified counterexample" in b.evidence
    assert "(2, 4)" in b.evidence and "(7, 3)" in b.evidence
    assert b.confidence >= 0.95  # certification raises confidence
