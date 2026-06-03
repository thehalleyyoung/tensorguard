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

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Callable, Optional, Sequence, Tuple, Union

Dim = Union[int, str]
Shape = Tuple[Dim, ...]

__all__ = [
    "DistributionSpec",
    "DistributionVerdict",
    "TransformSpec",
    "verify_distribution",
    "verify_log_prob",
]


@dataclass(frozen=True)
class DistributionSpec:
    """Resolved batch/event shape for a supported distribution family."""

    name: str
    batch_shape: Shape
    event_shape: Shape


@dataclass(frozen=True)
class TransformSpec:
    """Static shape contract for a supported distribution transform.

    The descriptor mirrors the shape-facing part of
    ``torch.distributions.Transform``: the domain/codomain event ranks and the
    forward/inverse shape maps used by ``TransformedDistribution``.  Users can
    also pass real torch Transform objects; this class exists for pure-static
    call sites that do not want to import torch.
    """

    name: str
    domain_event_dim: int = 0
    codomain_event_dim: int = 0
    input_event_shape: Optional[Shape] = None
    output_event_shape: Optional[Shape] = None
    _forward_shape: Optional[Callable[[Shape], Shape]] = field(
        default=None, repr=False, compare=False
    )
    _inverse_shape: Optional[Callable[[Shape], Shape]] = field(
        default=None, repr=False, compare=False
    )

    @staticmethod
    def identity(name: str = "IdentityTransform") -> "TransformSpec":
        return TransformSpec(name=name, domain_event_dim=0, codomain_event_dim=0)

    @staticmethod
    def reshape(
        input_event_shape: Sequence[Dim],
        output_event_shape: Sequence[Dim],
    ) -> "TransformSpec":
        in_shape = tuple(input_event_shape)
        out_shape = tuple(output_event_shape)
        return TransformSpec(
            name="ReshapeTransform",
            domain_event_dim=len(in_shape),
            codomain_event_dim=len(out_shape),
            input_event_shape=in_shape,
            output_event_shape=out_shape,
        )

    def forward_shape(self, shape: Sequence[Dim]) -> Optional[Shape]:
        base = tuple(shape)
        if self._forward_shape is not None:
            return _call_shape_fn(self._forward_shape, base)
        if self.input_event_shape is None or self.output_event_shape is None:
            return base
        return _reshape_shape(base, self.input_event_shape, self.output_event_shape)

    def inverse_shape(self, shape: Sequence[Dim]) -> Optional[Shape]:
        base = tuple(shape)
        if self._inverse_shape is not None:
            return _call_shape_fn(self._inverse_shape, base)
        if self.input_event_shape is None or self.output_event_shape is None:
            return base
        return _reshape_shape(base, self.output_event_shape, self.input_event_shape)


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


def _known_product(shape: Shape) -> Optional[int]:
    if not all(isinstance(dim, int) for dim in shape):
        return None
    return reduce(mul, (int(dim) for dim in shape), 1)


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


def _can_expand_batch(current: Shape, target: Shape) -> bool:
    """Return whether a torch Distribution batch can expand to ``target``."""

    if len(target) < len(current):
        return False
    for old, new in zip(reversed(current), reversed(target)):
        if old == new or old == 1:
            continue
        if isinstance(old, str) or isinstance(new, str):
            continue
        return False
    return True


def _suffix_compatible(shape: Shape, suffix: Shape) -> bool:
    if len(suffix) > len(shape):
        return False
    for actual, expected in zip(shape[-len(suffix) :] if suffix else (), suffix):
        if _known_mismatch(actual, expected):
            return False
        if actual != expected and not (
            isinstance(actual, str) or isinstance(expected, str)
        ):
            return False
    return True


def _reshape_shape(shape: Shape, input_event: Shape, output_event: Shape) -> Optional[Shape]:
    in_numel = _known_product(input_event)
    out_numel = _known_product(output_event)
    if in_numel is not None and out_numel is not None and in_numel != out_numel:
        return None
    if not _suffix_compatible(shape, input_event):
        return None
    return shape[: len(shape) - len(input_event)] + output_event


def _shape_result(value: object) -> Optional[Shape]:
    try:
        raw = tuple(value)  # torch.Size and ordinary tuples both work here.
    except TypeError:
        return None
    out = []
    for dim in raw:
        if isinstance(dim, (int, str)):
            out.append(dim)
        else:
            try:
                out.append(int(dim))
            except Exception:
                return None
    return tuple(out)


def _call_shape_fn(fn: Callable[[Shape], object], shape: Shape) -> Optional[Shape]:
    try:
        return _shape_result(fn(tuple(shape)))
    except Exception:
        return None


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


_IDENTITY_TRANSFORMS = {
    "identity",
    "identitytransform",
    "exp",
    "exptransform",
    "sigmoid",
    "sigmoidtransform",
    "tanh",
    "tanhtransform",
    "softplus",
    "softplustransform",
}


def _coerce_transform(value: object) -> Union[TransformSpec, DistributionVerdict]:
    if isinstance(value, TransformSpec):
        return value
    if isinstance(value, str):
        key = _canonical_name(value)
        if key in _IDENTITY_TRANSFORMS:
            return TransformSpec.identity(value)
        return _fail("unsupported_transform", f"unsupported transform descriptor {value!r}")
    if isinstance(value, dict):
        kind = _canonical_name(
            str(value.get("kind") or value.get("type") or value.get("name") or "")
        )
        if kind in _IDENTITY_TRANSFORMS:
            return TransformSpec.identity(
                str(value.get("name") or value.get("kind") or "IdentityTransform")
            )
        if kind in {"reshape", "reshapetransform"}:
            in_shape = value.get(
                "input_event_shape", value.get("input_shape", value.get("in_shape"))
            )
            out_shape = value.get(
                "output_event_shape", value.get("output_shape", value.get("out_shape"))
            )
            if in_shape is None or out_shape is None:
                return _fail(
                    "transform_descriptor",
                    "ReshapeTransform requires input and output event shapes",
                )
            spec = TransformSpec.reshape(tuple(in_shape), tuple(out_shape))
            in_numel = _known_product(spec.input_event_shape or ())
            out_numel = _known_product(spec.output_event_shape or ())
            if in_numel is not None and out_numel is not None and in_numel != out_numel:
                return _fail(
                    "reshape_numel",
                    "ReshapeTransform input/output event shapes must preserve numel",
                )
            return spec
        return _fail("unsupported_transform", f"unsupported transform descriptor {value!r}")

    domain = getattr(value, "domain", None)
    codomain = getattr(value, "codomain", None)
    forward_shape = getattr(value, "forward_shape", None)
    inverse_shape = getattr(value, "inverse_shape", None)
    domain_event_dim = getattr(domain, "event_dim", None)
    codomain_event_dim = getattr(codomain, "event_dim", None)
    if (
        isinstance(domain_event_dim, int)
        and isinstance(codomain_event_dim, int)
        and callable(forward_shape)
        and callable(inverse_shape)
    ):
        return TransformSpec(
            name=type(value).__name__,
            domain_event_dim=domain_event_dim,
            codomain_event_dim=codomain_event_dim,
            _forward_shape=forward_shape,
            _inverse_shape=inverse_shape,
        )
    return _fail("unsupported_transform", f"unsupported transform object {value!r}")


def _normalize_transforms(
    value: object,
) -> Union[Tuple[TransformSpec, ...], DistributionVerdict]:
    if isinstance(value, (TransformSpec, str, dict)) or (
        hasattr(value, "domain") and hasattr(value, "codomain")
    ):
        raw = (value,)
    elif isinstance(value, (list, tuple)):
        raw = tuple(value)
    else:
        return _fail(
            "missing_parameter",
            "TransformedDistribution requires a transform or non-empty transform list",
        )
    if not raw:
        return _fail("transform_list", "TransformedDistribution requires at least one transform")

    specs = []
    for item in raw:
        spec = _coerce_transform(item)
        if isinstance(spec, DistributionVerdict):
            return spec
        if spec.domain_event_dim < 0 or spec.codomain_event_dim < 0:
            return _fail(
                "transform_event_dim",
                "transform event dimensions must be non-negative",
            )
        specs.append(spec)
    return tuple(specs)


def _adjust_event_dim(current: int, plus: int, minus: int) -> int:
    return current + plus - minus


def _compose_domain_event_dim(transforms: Sequence[TransformSpec]) -> int:
    event_dim = transforms[-1].codomain_event_dim
    for transform in reversed(transforms):
        event_dim = _adjust_event_dim(
            event_dim,
            transform.domain_event_dim,
            transform.codomain_event_dim,
        )
        event_dim = max(event_dim, transform.domain_event_dim)
    return event_dim


def _compose_codomain_event_dim(transforms: Sequence[TransformSpec]) -> int:
    event_dim = transforms[0].domain_event_dim
    for transform in transforms:
        event_dim = _adjust_event_dim(
            event_dim,
            transform.codomain_event_dim,
            transform.domain_event_dim,
        )
        event_dim = max(event_dim, transform.codomain_event_dim)
    return event_dim


def _run_forward_shape(transforms: Sequence[TransformSpec], shape: Shape) -> Optional[Shape]:
    out = shape
    for transform in transforms:
        nxt = transform.forward_shape(out)
        if nxt is None:
            return None
        out = nxt
    return out


def _run_inverse_shape(transforms: Sequence[TransformSpec], shape: Shape) -> Optional[Shape]:
    out = shape
    for transform in reversed(transforms):
        nxt = transform.inverse_shape(out)
        if nxt is None:
            return None
        out = nxt
    return out


def _transformed_distribution(params: dict) -> DistributionVerdict:
    base = params.get("base")
    if isinstance(base, DistributionVerdict):
        if not base.ok or base.spec is None:
            return _fail("base_distribution", base.error or "base distribution is invalid")
        spec = base.spec
    elif isinstance(base, DistributionSpec):
        spec = base
    else:
        return _fail(
            "missing_parameter",
            "TransformedDistribution requires a DistributionSpec base",
        )

    transforms = _normalize_transforms(params.get("transforms"))
    if isinstance(transforms, DistributionVerdict):
        return transforms

    base_shape = spec.batch_shape + spec.event_shape
    base_event_dim = len(spec.event_shape)
    domain_event_dim = _compose_domain_event_dim(transforms)
    codomain_event_dim = _compose_codomain_event_dim(transforms)
    if len(base_shape) < domain_event_dim:
        return _fail(
            "transform_domain",
            f"base shape {base_shape} has rank {len(base_shape)}, "
            f"but transform domain requires {domain_event_dim} event dims",
        )

    forward_shape = _run_forward_shape(transforms, base_shape)
    if forward_shape is None:
        return _fail(
            "transform_forward_shape",
            "transform forward_shape rejected the base shape",
        )
    expanded_base_shape = _run_inverse_shape(transforms, forward_shape)
    if expanded_base_shape is None:
        return _fail(
            "transform_inverse_shape",
            "transform inverse_shape rejected its forward shape",
        )

    if expanded_base_shape != base_shape:
        if len(expanded_base_shape) < base_event_dim:
            return _fail(
                "transform_inverse_shape",
                "transform inverse shape is shorter than the base event rank",
            )
        expanded_event = expanded_base_shape[-base_event_dim:] if base_event_dim else ()
        if any(_known_mismatch(a, b) for a, b in zip(expanded_event, spec.event_shape)):
            return _fail(
                "transform_inverse_event",
                f"transform inverse event shape {expanded_event} disagrees with base event {spec.event_shape}",
            )
        expanded_batch = expanded_base_shape[: len(expanded_base_shape) - base_event_dim]
        if not _can_expand_batch(spec.batch_shape, expanded_batch):
            return _fail(
                "transform_expand",
                f"base batch {spec.batch_shape} cannot expand to transform-required {expanded_batch}",
            )

    transformed_event_base = base_event_dim + codomain_event_dim - domain_event_dim
    event_dim = max(codomain_event_dim, transformed_event_base)
    if event_dim < 0 or event_dim > len(forward_shape):
        return _fail(
            "transform_event_dim",
            f"transform event rank {event_dim} is invalid for forward shape {forward_shape}",
        )
    cut = len(forward_shape) - event_dim
    return _ok(
        f"TransformedDistribution[{spec.name}]",
        forward_shape[:cut],
        forward_shape[cut:],
    )


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
    if family == "transformeddistribution":
        return _transformed_distribution(params)
    return _fail(
        "unsupported_distribution",
        f"unsupported distribution {name!r}; supported: Normal, Bernoulli, Uniform, "
        "Exponential, Categorical, MultivariateNormal, Independent, TransformedDistribution",
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
