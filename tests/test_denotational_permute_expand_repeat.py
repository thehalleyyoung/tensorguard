"""Step 22 -- unit + property tests for the permute/expand/repeat transfer fns.

`permute` is the single highest-frequency previously-uncovered shape operator
in the real torchvision corpus (see `evaluation/operator_frequency.py`), and
`expand`/`repeat` are the next two shape-affecting long-tail ops. Step 22 added
their denotational transfer functions to `src/denotational_semantics.py`.

This module proves the new transfer functions are correct by:
  * **differential testing** the concrete transfer fns against real
    `torch.Tensor.{permute,expand,repeat}` on random shapes;
  * checking the **per-node soundness theorem** α(⟦op⟧(σ)) ⊑ ⟦op⟧♯(α(σ));
  * checking soundness over **symbolic** abstract dims; and
  * checking the documented **validation errors** (bad permutation, expand
    mismatch, too-few sizes/repeats, negative repeats, -1 leading expand dim).
"""

from __future__ import annotations

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pytest

settings.register_profile(
    "per_step22", suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("per_step22")

from src.denotational_semantics import (  # noqa: E402
    AbstractShape,
    abstract_expand,
    abstract_permute,
    abstract_repeat,
    concrete_expand,
    concrete_permute,
    concrete_repeat,
)

DIM = st.integers(min_value=1, max_value=5)


def shapes(min_rank=1, max_rank=4):
    return st.lists(DIM, min_size=min_rank, max_size=max_rank).map(tuple)


def _abs(t):
    return AbstractShape(dims=tuple(t))


def _assert_sound(concrete_result, abstract_result):
    assert _abs(concrete_result).leq(abstract_result), (
        "soundness violated: alpha(%r) not <= %r"
        % (concrete_result, abstract_result.dims))


# ---- permute --------------------------------------------------------------
@settings(max_examples=300)
@given(shapes(min_rank=1), st.randoms())
def test_permute_matches_torch_and_is_sound(a, rng):
    rank = len(a)
    perm = list(range(rank))
    rng.shuffle(perm)
    expected = tuple(torch.zeros(*a).permute(*perm).shape)
    got = concrete_permute([a], dims=perm)
    assert got == expected
    _assert_sound(got, abstract_permute([_abs(a)], dims=perm))


@settings(max_examples=200)
@given(st.integers(1, 4), st.randoms())
def test_permute_symbolic_soundness(rank, rng):
    sym = tuple("d%d" % i for i in range(rank))
    perm = list(range(rank))
    rng.shuffle(perm)
    out = abstract_permute([AbstractShape(dims=sym)], dims=perm)
    assert out.dims == tuple(sym[p] for p in perm)


def test_permute_rejects_non_permutation():
    with pytest.raises(ValueError):
        concrete_permute([(2, 3, 4)], dims=[0, 0, 1])
    with pytest.raises(ValueError):
        concrete_permute([(2, 3, 4)], dims=[0, 1])  # wrong length


# ---- expand ---------------------------------------------------------------
@settings(max_examples=300)
@given(shapes(min_rank=1), st.randoms())
def test_expand_matches_torch_and_is_sound(a, rng):
    rank = len(a)
    extra = rng.randint(0, 2)
    sizes = [rng.randint(1, 4) for _ in range(extra)]
    for d in a:
        if d == 1 and rng.random() < 0.5:
            sizes.append(rng.randint(1, 5))
        elif rng.random() < 0.3:
            sizes.append(-1)
        else:
            sizes.append(d)
    try:
        expected = tuple(torch.zeros(*a).expand(*sizes).shape)
    except RuntimeError:
        expected = None
    try:
        got = concrete_expand([a], sizes=sizes)
    except ValueError:
        got = None
    assert got == expected
    if got is not None:
        _assert_sound(got, abstract_expand([_abs(a)], sizes=sizes))


def test_expand_symbolic_keep_and_set():
    # -1 keeps a symbolic dim; a concrete request replaces a size-1 dim.
    out = abstract_expand([AbstractShape(dims=("n", 1))], sizes=[-1, 8])
    assert out.dims == ("n", 8)


def test_expand_validation_errors():
    with pytest.raises(ValueError):
        concrete_expand([(2, 3)], sizes=[3])              # too few
    with pytest.raises(ValueError):
        concrete_expand([(2, 3)], sizes=[4, 3])           # 2 -> 4 mismatch
    with pytest.raises(ValueError):
        concrete_expand([(2,)], sizes=[-1, 2])            # -1 leading dim


# ---- repeat ---------------------------------------------------------------
@settings(max_examples=300)
@given(shapes(min_rank=1), st.randoms())
def test_repeat_matches_torch_and_is_sound(a, rng):
    rank = len(a)
    extra = rng.randint(0, 2)
    reps = tuple(rng.randint(1, 3) for _ in range(rank + extra))
    expected = tuple(torch.zeros(*a).repeat(*reps).shape)
    got = concrete_repeat([a], repeats=reps)
    assert got == expected
    _assert_sound(got, abstract_repeat([_abs(a)], repeats=reps))


def test_repeat_symbolic():
    out = abstract_repeat([AbstractShape(dims=("n", 4))], repeats=[2, 3])
    assert out.dims == ("n_times_2", 12)
    # repeat of 1 keeps a symbolic dim unchanged
    out1 = abstract_repeat([AbstractShape(dims=("n",))], repeats=[1])
    assert out1.dims == ("n",)


def test_repeat_validation_errors():
    with pytest.raises(ValueError):
        concrete_repeat([(2, 3)], repeats=[2])       # too few
    with pytest.raises(ValueError):
        concrete_repeat([(2,)], repeats=[-1])        # negative
