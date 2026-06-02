"""Differential tests for ``src.distributions_verify`` vs real torch distributions.

The oracle constructs real ``torch.distributions`` with ``validate_args=False``
so value-support checks (for example Uniform rejecting a zero outside its
support) do not masquerade as shape failures.  Structural shape errors still
surface through the real constructors, ``sample()``, or ``log_prob()``.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
import torch.distributions as D  # noqa: E402

from src.distributions_verify import (  # noqa: E402
    DistributionSpec,
    verify_distribution,
    verify_log_prob,
)


def _zeros(shape, *, dtype=torch.float32):
    return torch.zeros(tuple(shape), dtype=dtype)


def _ones(shape):
    return torch.ones(tuple(shape), dtype=torch.float32)


def _covariance(shape):
    shape = tuple(shape)
    if len(shape) >= 2 and shape[-1] == shape[-2] and shape[-1] > 0:
        return torch.eye(shape[-1]).expand(shape).clone()
    return torch.zeros(shape)


def _make_distribution(name, params):
    if name == "Normal":
        return D.Normal(_zeros(params["loc"]), _ones(params["scale"]), validate_args=False)
    if name == "Uniform":
        return D.Uniform(_zeros(params["low"]), _ones(params["high"]), validate_args=False)
    if name == "Exponential":
        return D.Exponential(_ones(params["rate"]), validate_args=False)
    if name == "Bernoulli":
        kwargs = {}
        if "probs" in params:
            kwargs["probs"] = torch.full(tuple(params["probs"]), 0.5)
        if "logits" in params:
            kwargs["logits"] = _zeros(params["logits"])
        return D.Bernoulli(validate_args=False, **kwargs)
    if name == "Categorical":
        if "logits" in params:
            return D.Categorical(logits=_zeros(params["logits"]), validate_args=False)
        return D.Categorical(probs=_ones(params["probs"]), validate_args=False)
    if name == "MultivariateNormal":
        matrix_name = next(
            key for key in ("covariance_matrix", "precision_matrix", "scale_tril")
            if key in params
        )
        matrix = _covariance(params[matrix_name])
        kwargs = {matrix_name: matrix}
        return D.MultivariateNormal(_zeros(params["loc"]), validate_args=False, **kwargs)
    raise AssertionError(f"unknown test distribution {name}")


def _real_usable(name, params):
    try:
        dist = _make_distribution(name, params)
        dist.sample()
        return "ok", tuple(dist.batch_shape), tuple(dist.event_shape)
    except Exception:
        return "err", None, None


def _value(shape, dtype=torch.float32):
    return torch.zeros(tuple(shape), dtype=dtype)


def _real_log_prob(name, params, value_shape):
    try:
        dist = _make_distribution(name, params)
        dtype = torch.long if name == "Categorical" else torch.float32
        out = dist.log_prob(_value(value_shape, dtype))
        return "ok", tuple(out.shape)
    except Exception:
        return "err", None


def _check_constructor(name, params):
    real_status, real_batch, real_event = _real_usable(name, params)
    verdict = verify_distribution(name, **params)
    static_status = "ok" if verdict.ok else "err"
    assert static_status == real_status, (
        f"{name} params={params}: real={real_status} static={static_status} "
        f"({verdict.error})"
    )
    if real_status == "ok":
        assert verdict.spec is not None
        assert verdict.spec.batch_shape == real_batch
        assert verdict.spec.event_shape == real_event


def _check_log_prob(name, params, value_shape):
    real_status, real_shape = _real_log_prob(name, params, value_shape)
    verdict = verify_distribution(name, **params)
    lp = verify_log_prob(verdict, value_shape)
    static_status = "ok" if lp.ok else "err"
    assert static_status == real_status, (
        f"{name} params={params} value={value_shape}: real={real_status} "
        f"static={static_status} ({lp.error})"
    )
    if real_status == "ok":
        assert lp.output_shape == real_shape


VALID_CASES = [
    ("Normal", {"loc": (2, 1, 3), "scale": (1, 4, 3)}, (2, 4, 3), ()),
    ("Uniform", {"low": (), "high": (2, 3)}, (2, 3), ()),
    ("Exponential", {"rate": (3, 1)}, (3, 1), ()),
    ("Bernoulli", {"logits": (5,)}, (5,), ()),
    ("Categorical", {"logits": (2, 3, 4)}, (2, 3), ()),
    (
        "MultivariateNormal",
        {"loc": (4, 3, 5), "covariance_matrix": (1, 3, 5, 5)},
        (4, 3),
        (5,),
    ),
    (
        "MultivariateNormal",
        {"loc": (2, 5), "precision_matrix": (2, 5, 5)},
        (2,),
        (5,),
    ),
    (
        "MultivariateNormal",
        {"loc": (5,), "scale_tril": (5, 5)},
        (),
        (5,),
    ),
]


@pytest.mark.parametrize("name,params,batch,event", VALID_CASES)
def test_valid_constructor_shapes_match_torch(name, params, batch, event):
    _check_constructor(name, params)
    verdict = verify_distribution(name, **params)
    assert verdict.ok
    assert verdict.spec == DistributionSpec(name, batch, event)


INVALID_CASES = [
    ("Normal", {"loc": (2,), "scale": (3,)}, "param_broadcast"),
    ("Uniform", {"low": (2,), "high": (3,)}, "param_broadcast"),
    ("Bernoulli", {}, "parameter_choice"),
    ("Bernoulli", {"probs": (2,), "logits": (2,)}, "parameter_choice"),
    ("Categorical", {"logits": ()}, "categories"),
    ("Categorical", {"logits": (2, 0)}, "categories"),
    ("MultivariateNormal", {"loc": (), "covariance_matrix": (1, 1)}, "event_dim"),
    ("MultivariateNormal", {"loc": (3,), "covariance_matrix": (3, 4)}, "matrix_square"),
    # Torch constructs this object, but every real use fails; the static contract
    # is intentionally over usable sample/log_prob behavior, not bare __init__.
    ("MultivariateNormal", {"loc": (3,), "covariance_matrix": (4, 4)}, "matrix_event"),
    (
        "MultivariateNormal",
        {"loc": (2, 3), "covariance_matrix": (4, 3, 3)},
        "param_broadcast",
    ),
]


@pytest.mark.parametrize("name,params,kind", INVALID_CASES)
def test_invalid_constructor_shapes_match_torch_use_failures(name, params, kind):
    _check_constructor(name, params)
    verdict = verify_distribution(name, **params)
    assert not verdict.ok
    assert verdict.error_kind == kind


def test_independent_moves_batch_dims_to_event_dims():
    base = verify_distribution("Normal", loc=(2, 3, 4), scale=(1, 3, 4)).spec
    verdict = verify_distribution(
        "Independent", base=base, reinterpreted_batch_ndims=2
    )
    real = D.Independent(
        D.Normal(_zeros((2, 3, 4)), _ones((1, 3, 4)), validate_args=False),
        2,
        validate_args=False,
    )
    real.sample()
    assert verdict.ok
    assert verdict.spec is not None
    assert verdict.spec.batch_shape == tuple(real.batch_shape) == (2,)
    assert verdict.spec.event_shape == tuple(real.event_shape) == (3, 4)


def test_independent_rejects_too_many_reinterpreted_dims():
    base = verify_distribution("Normal", loc=(2, 3), scale=(2, 3)).spec
    verdict = verify_distribution(
        "Independent", base=base, reinterpreted_batch_ndims=3
    )
    assert not verdict.ok
    assert verdict.error_kind == "reinterpret_ndims"
    with pytest.raises(ValueError):
        D.Independent(
            D.Normal(_zeros((2, 3)), _ones((2, 3)), validate_args=False),
            3,
            validate_args=False,
        )


LOG_PROB_CASES = [
    ("Normal", {"loc": (2, 3), "scale": (2, 3)}, (3,), "ok", (2, 3)),
    ("Normal", {"loc": (2, 3), "scale": (2, 3)}, (5, 2, 3), "ok", (5, 2, 3)),
    ("Normal", {"loc": (2, 3), "scale": (2, 3)}, (4,), "err", None),
    ("Categorical", {"logits": (2, 3, 4)}, (3,), "ok", (2, 3)),
    ("Categorical", {"logits": (2, 3, 4)}, (5, 2, 3), "ok", (5, 2, 3)),
    ("Categorical", {"logits": (2, 3, 4)}, (5, 2), "err", None),
    (
        "MultivariateNormal",
        {"loc": (4, 3, 5), "covariance_matrix": (4, 3, 5, 5)},
        (5,),
        "ok",
        (4, 3),
    ),
    (
        "MultivariateNormal",
        {"loc": (4, 3, 5), "covariance_matrix": (4, 3, 5, 5)},
        (7, 4, 3, 1),
        "ok",
        (7, 4, 3),
    ),
    (
        "MultivariateNormal",
        {"loc": (4, 3, 5), "covariance_matrix": (4, 3, 5, 5)},
        (4, 3, 6),
        "err",
        None,
    ),
]


@pytest.mark.parametrize("name,params,value_shape,status,out_shape", LOG_PROB_CASES)
def test_log_prob_shapes_match_torch(name, params, value_shape, status, out_shape):
    _check_log_prob(name, params, value_shape)
    lp = verify_log_prob(verify_distribution(name, **params), value_shape)
    assert ("ok" if lp.ok else "err") == status
    if status == "ok":
        assert lp.output_shape == out_shape


def test_independent_log_prob_shapes_match_torch():
    base = verify_distribution("Normal", loc=(2, 3, 4), scale=(2, 3, 4)).spec
    spec = verify_distribution("Independent", base=base, reinterpreted_batch_ndims=2)
    real = D.Independent(
        D.Normal(_zeros((2, 3, 4)), _ones((2, 3, 4)), validate_args=False),
        2,
        validate_args=False,
    )
    for value_shape, expected in [
        ((7, 2, 1, 4), (7, 2)),
        ((4,), (2,)),
    ]:
        out = real.log_prob(_zeros(value_shape))
        assert tuple(out.shape) == expected
        assert verify_log_prob(spec, value_shape).output_shape == expected
    assert not verify_log_prob(spec, (5,)).ok


def test_symbolic_dims_are_not_refuted_when_undecidable():
    normal = verify_distribution("Normal", loc=("B", 4), scale=(3, 1))
    assert normal.ok
    assert verify_log_prob(normal, ("T", 1, 4)).ok

    mvn = verify_distribution(
        "MultivariateNormal",
        loc=("B", "D"),
        covariance_matrix=(1, "D", "D"),
    )
    assert mvn.ok
    assert mvn.spec is not None
    assert mvn.spec.event_shape == ("D",)
    assert verify_log_prob(mvn, ("N", 1, "D")).ok


def _rand_shape(rng, *, max_rank=3, dims=(1, 2, 3, 4)):
    return tuple(rng.choice(dims) for _ in range(rng.randint(0, max_rank)))


def test_scalar_family_fuzz_matches_real_torch():
    rng = random.Random(183)
    cases = 0
    for _ in range(240):
        family = rng.choice(["Normal", "Uniform", "Exponential", "Bernoulli"])
        if family == "Normal":
            params = {"loc": _rand_shape(rng), "scale": _rand_shape(rng)}
        elif family == "Uniform":
            params = {"low": _rand_shape(rng), "high": _rand_shape(rng)}
        elif family == "Exponential":
            params = {"rate": _rand_shape(rng)}
        else:
            key = rng.choice(["probs", "logits"])
            params = {key: _rand_shape(rng)}

        _check_constructor(family, params)
        if verify_distribution(family, **params).ok:
            _check_log_prob(family, params, _rand_shape(rng, max_rank=4))
        cases += 1
    assert cases == 240


def test_categorical_fuzz_matches_real_torch():
    rng = random.Random(184)
    cases = 0
    for _ in range(120):
        rank = rng.randint(0, 4)
        if rank == 0:
            shape = ()
        else:
            shape = tuple(rng.choice([1, 2, 3]) for _ in range(rank - 1))
            shape += (rng.choice([0, 1, 2, 5]),)
        params = {"logits": shape}
        _check_constructor("Categorical", params)
        if verify_distribution("Categorical", **params).ok:
            _check_log_prob("Categorical", params, _rand_shape(rng, max_rank=4))
        cases += 1
    assert cases == 120


def test_multivariate_normal_fuzz_matches_real_torch():
    rng = random.Random(185)
    cases = 0
    for _ in range(160):
        loc_batch = _rand_shape(rng, max_rank=2, dims=(1, 2, 3))
        event = rng.randint(1, 5)
        loc = loc_batch + (event,)

        matrix_batch = _rand_shape(rng, max_rank=2, dims=(1, 2, 3))
        left = rng.choice([event, max(1, event + 1)])
        right = rng.choice([left, event, max(1, event + 2)])
        matrix = matrix_batch + (left, right)
        params = {"loc": loc, "covariance_matrix": matrix}

        _check_constructor("MultivariateNormal", params)
        if verify_distribution("MultivariateNormal", **params).ok:
            _check_log_prob(
                "MultivariateNormal",
                params,
                _rand_shape(rng, max_rank=3, dims=(1, 2, event, max(1, event + 1))),
            )
        cases += 1
    assert cases == 160


def test_public_package_exports_distribution_checker():
    import tensorguard

    assert tensorguard.verify_distribution is verify_distribution
    assert tensorguard.verify_log_prob is verify_log_prob
