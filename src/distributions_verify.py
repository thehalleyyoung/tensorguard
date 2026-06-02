"""Static batch/event-shape verifier for ``torch.distributions``.

Probabilistic layers often fail far away from the distribution constructor:
``MultivariateNormal(loc, covariance_matrix)`` can construct with mismatched
event dimensions and only crash when sampled or scored, and ``log_prob`` shape
errors are easy to miss in smoke tests.  This module checks the batch/event
shape contract without constructing tensors or sampling.

The verifier intentionally models the *usable* shape contract: construction plus
the shape behavior needed by ``sample`` and ``log_prob``.  It is differentially
tested against real ``torch.distributions`` with value-support validation
disabled so the oracle isolates shape failures rather than support failures.
Symbolic dimensions are never refuted when compatibility cannot be decided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

Dim = Union[int, str]
Shape = Tuple[Dim, ...]

__all__ = [
    "DistributionSpec",
    "DistributionVerdict",
    "verify_distribution",
    "verify_log_prob",
]


@dataclass(frozen=True)
class DistributionSpec:
    """Resolved batch/event shape for a supported distribution family."""

    name: str
    batch_shape: Shape
    event_shape: Shape


@dataclass
class DistributionVerdict:
    """Result of checking a distribution constructor or ``log_prob`` call."""

    ok: bool
    spec: Optional[DistributionSpec] = None
    output_shape: Optional[Shape] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _shape(value: Optional[Sequence[Dim]]) -> Optional[Shape]:
    if value is None:
        return None
    return tuple(value)


def _fail(kind: str, message: str) -> DistributionVerdict:
    return DistributionVerdict(False, error=message, error_kind=kind)


def _ok(name: str, batch: Sequence[Dim], event: Sequence[Dim]) -> DistributionVerdict:
    return DistributionVerdict(
        True,
        spec=DistributionSpec(name=name, batch_shape=tuple(batch), event_shape=tuple(event)),
    )


def _known_mismatch(a: Dim, b: Dim) -> bool:
    return isinstance(a, int) and isinstance(b, int) and a != b


def _broadcast_dim(a: Dim, b: Dim) -> Optional[Dim]:
    if isinstance(a, int) and isinstance(b, int):
        if a == b:
            return a
        if a == 1:
            return b
        if b == 1:
            return a
        return None
    if a == 1:
        return b
    if b == 1:
        return a
    if a == b:
        return a
    # A symbolic dimension may equal either side at runtime.  Keep a symbolic
    # representative rather than inventing a concrete fact.
    return a if isinstance(a, str) else b


def _broadcast_shapes(shapes: Sequence[Sequence[Dim]]) -> Optional[Shape]:
    if not shapes:
        return ()
    out = []
    rank = max(len(s) for s in shapes)
    for i in range(1, rank + 1):
        cur: Dim = 1
        for shape in shapes:
            if i <= len(shape):
                nxt = _broadcast_dim(cur, shape[-i])
                if nxt is None:
                    return None
                cur = nxt
        out.append(cur)
    out.reverse()
    return tuple(out)


def _canonical_name(name: str) -> str:
    return name.rsplit(".", 1)[-1].replace("_", "").lower()


def _exactly_one(params: dict, *names: str) -> Optional[str]:
    present = [n for n in names if _shape(params.get(n)) is not None]
    if len(present) == 1:
        return present[0]
    return None


def _scalar_family(name: str, params: dict, *required: str) -> DistributionVerdict:
    shapes = []
    for key in required:
        value = _shape(params.get(key))
        if value is None:
            return _fail("missing_parameter", f"{name} requires parameter {key!r}")
        shapes.append(value)
    batch = _broadcast_shapes(shapes)
    if batch is None:
        rendered = ", ".join(f"{k}={tuple(s)}" for k, s in zip(required, shapes))
        return _fail("param_broadcast", f"{name} parameters do not broadcast: {rendered}")
    return _ok(name, batch, ())


def _single_parameter_family(name: str, params: dict, *choices: str) -> DistributionVerdict:
    chosen = _exactly_one(params, *choices)
    if chosen is None:
        return _fail(
            "parameter_choice",
            f"{name} requires exactly one of {', '.join(repr(c) for c in choices)}",
        )
    shape = _shape(params[chosen])
    assert shape is not None
    return _ok(name, shape, ())


def _categorical(params: dict) -> DistributionVerdict:
    chosen = _exactly_one(params, "probs", "logits")
    if chosen is None:
        return _fail("parameter_choice", "Categorical requires exactly one of 'probs' or 'logits'")
    shape = _shape(params[chosen])
    assert shape is not None
    if len(shape) < 1:
        return _fail("categories", "Categorical probabilities/logits must have rank >= 1")
    categories = shape[-1]
    if isinstance(categories, int) and categories < 1:
        return _fail("categories", f"Categorical must have at least one category, got {categories}")
    return _ok("Categorical", shape[:-1], ())


def _multivariate_normal(params: dict) -> DistributionVerdict:
    loc = _shape(params.get("loc"))
    if loc is None:
        return _fail("missing_parameter", "MultivariateNormal requires 'loc'")
    if len(loc) < 1:
        return _fail("event_dim", "MultivariateNormal loc must have at least one event dim")

    matrix_name = _exactly_one(params, "covariance_matrix", "precision_matrix", "scale_tril")
    if matrix_name is None:
        return _fail(
            "parameter_choice",
            "MultivariateNormal requires exactly one covariance/precision/scale_tril matrix",
        )
    matrix = _shape(params[matrix_name])
    assert matrix is not None
    if len(matrix) < 2:
        return _fail("matrix_rank", f"{matrix_name} must have rank >= 2, got {len(matrix)}")

    left, right = matrix[-2], matrix[-1]
    if _known_mismatch(left, right):
        return _fail("matrix_square", f"{matrix_name} must be square, got trailing dims {left}x{right}")

    event = loc[-1]
    if _known_mismatch(event, left) or _known_mismatch(event, right):
        return _fail(
            "matrix_event",
            f"loc event dim {event} disagrees with {matrix_name} trailing dims {left}x{right}",
        )

    batch = _broadcast_shapes([loc[:-1], matrix[:-2]])
    if batch is None:
        return _fail(
            "param_broadcast",
            f"loc batch {tuple(loc[:-1])} does not broadcast with {matrix_name} batch {tuple(matrix[:-2])}",
        )
    return _ok("MultivariateNormal", batch, (event,))


def _independent(params: dict) -> DistributionVerdict:
    base = params.get("base")
    if isinstance(base, DistributionVerdict):
        if not base.ok or base.spec is None:
            return _fail("base_distribution", base.error or "base distribution is invalid")
        spec = base.spec
    elif isinstance(base, DistributionSpec):
        spec = base
    else:
        return _fail("missing_parameter", "Independent requires a DistributionSpec base")

    k = params.get("reinterpreted_batch_ndims")
    if not isinstance(k, int):
        return _fail("reinterpret_ndims", "reinterpreted_batch_ndims must be an integer")
    if k < 0 or k > len(spec.batch_shape):
        return _fail(
            "reinterpret_ndims",
            f"cannot reinterpret {k} dims from batch shape {spec.batch_shape}",
        )
    if k == 0:
        batch = spec.batch_shape
        event = spec.event_shape
    else:
        batch = spec.batch_shape[:-k]
        event = spec.batch_shape[-k:] + spec.event_shape
    return _ok(f"Independent[{spec.name}]", batch, event)


def verify_distribution(name: str, **params: object) -> DistributionVerdict:
    """Verify constructor batch/event shapes for a supported distribution.

    Shape parameters are passed as tuples/lists of integers or symbolic strings,
    e.g. ``verify_distribution("Normal", loc=(2, 1), scale=(3,))``.
    """

    family = _canonical_name(name)
    if family == "normal":
        return _scalar_family("Normal", params, "loc", "scale")
    if family == "uniform":
        return _scalar_family("Uniform", params, "low", "high")
    if family == "exponential":
        return _scalar_family("Exponential", params, "rate")
    if family == "bernoulli":
        return _single_parameter_family("Bernoulli", params, "probs", "logits")
    if family == "categorical":
        return _categorical(params)
    if family == "multivariatenormal":
        return _multivariate_normal(params)
    if family == "independent":
        return _independent(params)
    return _fail(
        "unsupported_distribution",
        f"unsupported distribution {name!r}; supported: Normal, Bernoulli, Uniform, "
        "Exponential, Categorical, MultivariateNormal, Independent",
    )


def _spec_from(value: Union[DistributionSpec, DistributionVerdict]) -> Optional[DistributionSpec]:
    if isinstance(value, DistributionSpec):
        return value
    if isinstance(value, DistributionVerdict) and value.ok:
        return value.spec
    return None


def verify_log_prob(
    distribution: Union[DistributionSpec, DistributionVerdict],
    value_shape: Sequence[Dim],
) -> DistributionVerdict:
    """Verify the output shape of ``distribution.log_prob(value)``.

    The real torch implementations first broadcast ``value`` against
    ``batch_shape + event_shape`` and then reduce the rightmost event dimensions.
    This function mirrors that behavior and returns the resulting output shape.
    """

    spec = _spec_from(distribution)
    if spec is None:
        if isinstance(distribution, DistributionVerdict):
            return _fail(distribution.error_kind or "distribution", distribution.error or "invalid distribution")
        return _fail("distribution", "invalid distribution spec")

    full_shape = spec.batch_shape + spec.event_shape
    broadcast = _broadcast_shapes([tuple(value_shape), full_shape])
    if broadcast is None:
        return _fail(
            "value_broadcast",
            f"value shape {tuple(value_shape)} does not broadcast against "
            f"batch+event shape {full_shape}",
        )
    event_ndim = len(spec.event_shape)
    if spec.name == "MultivariateNormal" and event_ndim:
        event_out = broadcast[-event_ndim:]
        for expected, actual in zip(spec.event_shape, event_out):
            if _known_mismatch(expected, actual):
                return _fail(
                    "value_event",
                    f"value shape {tuple(value_shape)} broadcasts the event dims "
                    f"to {event_out}, but MultivariateNormal requires {spec.event_shape}",
                )
    output = broadcast if event_ndim == 0 else broadcast[:-event_ndim]
    return DistributionVerdict(True, spec=spec, output_shape=tuple(output))
