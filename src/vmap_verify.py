"""Static shape transfer for ``torch.vmap`` / ``torch.func.vmap``.

``vmap`` is a higher-order shape transform: it removes one mapped dimension from
each mapped input before the function body runs, then inserts the shared batch
dimension into each batched output.  Batch-size mismatches and invalid
``in_dims``/``out_dims`` fail before the user's kernel has a chance to explain
the problem.

This module checks that transfer without executing the mapped function.  It is
sound-by-abstention: symbolic batch sizes are never refuted, and the small
version-sensitive corner where an unbatched constant output is requested with a
non-leading integer ``out_dim`` returns an unknown output shape rather than a
false positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

Dim = Union[int, str]
Shape = Tuple[Dim, ...]

__all__ = ["VmapVerdict", "verify_vmap"]


@dataclass(frozen=True)
class VmapVerdict:
    """Result of checking one ``vmap`` shape transfer.

    ``ok=True`` means TensorGuard found no statically refutable vmap contract
    violation.  ``output_shapes=None`` with ``unknown_reason`` set means the
    operation is not rejected, but this shape-only checker deliberately abstains
    from claiming an exact output shape.
    """

    ok: bool
    batch_dim: Optional[Dim] = None
    body_input_shapes: Optional[Tuple[Shape, ...]] = None
    output_shapes: Any = None
    error: Optional[str] = None
    error_kind: Optional[str] = None
    unknown_reason: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _fail(kind: str, message: str) -> VmapVerdict:
    return VmapVerdict(False, error=message, error_kind=kind)


def _unknown(
    batch_dim: Dim,
    body_inputs: Sequence[Shape],
    reason: str,
) -> VmapVerdict:
    return VmapVerdict(
        True,
        batch_dim=batch_dim,
        body_input_shapes=tuple(body_inputs),
        output_shapes=None,
        unknown_reason=reason,
    )


def _is_dim(value: object) -> bool:
    return (type(value) is int) or isinstance(value, str)


def _is_shape(value: object) -> bool:
    return isinstance(value, (tuple, list)) and all(_is_dim(v) for v in value)


def _shape(value: Sequence[Dim]) -> Shape:
    return tuple(value)


def _normalise_dim(rank: int, dim: object, *, kind: str) -> Tuple[Optional[int], Optional[VmapVerdict]]:
    if type(dim) is not int:
        return None, _fail(kind, f"{kind} must be an integer or None, got {dim!r}")
    if dim < -rank or dim >= rank:
        return None, _fail(
            f"{kind}_range",
            f"{kind}={dim} is out of range for rank-{rank} tensor",
        )
    return dim % rank, None


def _normalise_out_dim(
    rank_after_insert: int,
    dim: object,
) -> Tuple[Optional[int], Optional[VmapVerdict]]:
    if dim is None:
        return None, None
    if type(dim) is not int:
        return None, _fail("out_dim", f"out_dim must be an integer or None, got {dim!r}")
    if dim < -rank_after_insert or dim >= rank_after_insert:
        return None, _fail(
            "out_dim_range",
            f"out_dim={dim} is out of range for batched output rank {rank_after_insert}",
        )
    return dim % rank_after_insert, None


def _normalise_in_dims(in_dims: object, n_inputs: int) -> Tuple[Optional[List[Optional[int]]], Optional[VmapVerdict]]:
    if in_dims is None:
        return None, _fail(
            "in_dim_structure",
            "top-level in_dims=None is invalid; use a tuple/list containing None "
            "for individual unmapped tensor arguments",
        )
    if type(in_dims) is int:
        return [in_dims for _ in range(n_inputs)], None
    if isinstance(in_dims, (tuple, list)):
        if len(in_dims) != n_inputs:
            return None, _fail(
                "in_dim_structure",
                f"in_dims has {len(in_dims)} entries for {n_inputs} tensor arguments",
            )
        dims: List[Optional[int]] = []
        for dim in in_dims:
            if dim is None or type(dim) is int:
                dims.append(dim)
            else:
                return None, _fail("in_dim", f"in_dim must be an integer or None, got {dim!r}")
        return dims, None
    return None, _fail("in_dim_structure", f"unsupported in_dims structure {in_dims!r}")


def _normalise_bool_tree(template: Any, value: object) -> Tuple[Optional[Any], Optional[VmapVerdict]]:
    if isinstance(value, bool):
        return _map_scalar_to_tree(template, value), None
    if _is_shape(template):
        return None, _fail(
            "batched_structure",
            "body_output_batched must be a bool for a tensor output shape",
        )
    if isinstance(template, tuple):
        if not isinstance(value, tuple) or len(value) != len(template):
            return None, _fail("batched_structure", "body_output_batched tuple does not match outputs")
        out = []
        for t, v in zip(template, value):
            child, err = _normalise_bool_tree(t, v)
            if err is not None:
                return None, err
            out.append(child)
        return tuple(out), None
    if isinstance(template, list):
        if not isinstance(value, list) or len(value) != len(template):
            return None, _fail("batched_structure", "body_output_batched list does not match outputs")
        out = []
        for t, v in zip(template, value):
            child, err = _normalise_bool_tree(t, v)
            if err is not None:
                return None, err
            out.append(child)
        return out, None
    if isinstance(template, dict):
        if not isinstance(value, dict) or set(value) != set(template):
            return None, _fail("batched_structure", "body_output_batched dict does not match outputs")
        out = {}
        for key in sorted(template):
            child, err = _normalise_bool_tree(template[key], value[key])
            if err is not None:
                return None, err
            out[key] = child
        return out, None
    return None, _fail("output_shape", f"unsupported output shape tree {template!r}")


def _map_scalar_to_tree(template: Any, value: object) -> Any:
    if _is_shape(template):
        return value
    if isinstance(template, tuple):
        return tuple(_map_scalar_to_tree(t, value) for t in template)
    if isinstance(template, list):
        return [_map_scalar_to_tree(t, value) for t in template]
    if isinstance(template, dict):
        return {k: _map_scalar_to_tree(template[k], value) for k in sorted(template)}
    return value


def _normalise_out_dims(template: Any, value: object) -> Tuple[Optional[Any], Optional[VmapVerdict]]:
    if value is None or type(value) is int:
        return _map_scalar_to_tree(template, value), None
    if _is_shape(template):
        return None, _fail("out_dim_structure", "out_dims structure does not match tensor output")
    if isinstance(template, tuple):
        if not isinstance(value, tuple) or len(value) != len(template):
            return None, _fail("out_dim_structure", "out_dims tuple does not match outputs")
        out = []
        for t, v in zip(template, value):
            child, err = _normalise_out_dims(t, v)
            if err is not None:
                return None, err
            out.append(child)
        return tuple(out), None
    if isinstance(template, list):
        if not isinstance(value, list) or len(value) != len(template):
            return None, _fail("out_dim_structure", "out_dims list does not match outputs")
        out = []
        for t, v in zip(template, value):
            child, err = _normalise_out_dims(t, v)
            if err is not None:
                return None, err
            out.append(child)
        return out, None
    if isinstance(template, dict):
        if not isinstance(value, dict) or set(value) != set(template):
            return None, _fail("out_dim_structure", "out_dims dict does not match outputs")
        out = {}
        for key in sorted(template):
            child, err = _normalise_out_dims(template[key], value[key])
            if err is not None:
                return None, err
            out[key] = child
        return out, None
    return None, _fail("output_shape", f"unsupported output shape tree {template!r}")


def _insert_dim(shape: Shape, index: int, dim: Dim) -> Shape:
    return shape[:index] + (dim,) + shape[index:]


def _map_output_shape(
    template: Any,
    out_dims: Any,
    batched: Any,
    batch_dim: Dim,
) -> Tuple[Optional[Any], Optional[VmapVerdict], Optional[str]]:
    if _is_shape(template):
        shape = _shape(template)
        out_dim = out_dims
        is_batched = batched
        if not isinstance(is_batched, bool):
            return None, _fail("batched_structure", "body_output_batched leaf must be bool"), None

        rank_after_insert = len(shape) + 1
        resolved, err = _normalise_out_dim(rank_after_insert, out_dim)
        if err is not None:
            return None, err, None

        if is_batched:
            if out_dim is None:
                return None, _fail(
                    "out_dim_unbatched",
                    "out_dim=None is invalid for an output that carries the vmap batch",
                ), None
            assert resolved is not None
            return _insert_dim(shape, resolved, batch_dim), None, None

        if out_dim is None:
            return shape, None, None
        if resolved == 0:
            return _insert_dim(shape, 0, batch_dim), None, None
        return None, None, (
            "unbatched constant outputs with a non-leading integer out_dim are "
            "version-sensitive in PyTorch; TensorGuard does not refute them but "
            "does not claim an exact shape"
        )

    if isinstance(template, tuple):
        out_values = []
        unknowns = []
        for t, od, b in zip(template, out_dims, batched):
            child, err, unk = _map_output_shape(t, od, b, batch_dim)
            if err is not None:
                return None, err, None
            out_values.append(child)
            if unk is not None:
                unknowns.append(unk)
        if unknowns:
            return None, None, "; ".join(dict.fromkeys(unknowns))
        return tuple(out_values), None, None

    if isinstance(template, list):
        out_values = []
        unknowns = []
        for t, od, b in zip(template, out_dims, batched):
            child, err, unk = _map_output_shape(t, od, b, batch_dim)
            if err is not None:
                return None, err, None
            out_values.append(child)
            if unk is not None:
                unknowns.append(unk)
        if unknowns:
            return None, None, "; ".join(dict.fromkeys(unknowns))
        return out_values, None, None

    if isinstance(template, dict):
        out_values = {}
        unknowns = []
        for key in sorted(template):
            child, err, unk = _map_output_shape(template[key], out_dims[key], batched[key], batch_dim)
            if err is not None:
                return None, err, None
            out_values[key] = child
            if unk is not None:
                unknowns.append(unk)
        if unknowns:
            return None, None, "; ".join(dict.fromkeys(unknowns))
        return out_values, None, None

    return None, _fail("output_shape", f"unsupported output shape tree {template!r}"), None


def verify_vmap(
    input_shapes: Sequence[Sequence[Dim]],
    body_output_shapes: Any,
    *,
    in_dims: object = 0,
    out_dims: object = 0,
    body_output_batched: object = True,
) -> VmapVerdict:
    """Verify a ``torch.vmap`` shape transfer.

    Args:
        input_shapes: Flat tensor-argument shapes at the vmap call boundary.
        body_output_shapes: Shape tree returned by the mapped function body after
            mapped input dimensions have been removed.  A single tensor output is
            written as a shape, e.g. ``(4, 8)``; multiple or nested outputs may be
            tuples/lists/dicts containing shapes.
        in_dims: Integer mapped dim for all inputs, or a tuple/list with one
            integer/``None`` per input.  A top-level ``None`` is rejected to match
            current PyTorch.
        out_dims: Integer/``None`` applied to every output, or a matching output
            tree of integer/``None`` leaves.
        body_output_batched: Whether each output leaf carries the vmap batch.
            This is caller-supplied shape provenance, not derived from shape
            alone.  Use ``False`` for constants or outputs depending only on
            unmapped inputs.
    """

    shapes: List[Shape] = []
    for shape in input_shapes:
        if not _is_shape(shape):
            return _fail("input_shape", f"unsupported input shape {shape!r}")
        shapes.append(_shape(shape))

    dims, err = _normalise_in_dims(in_dims, len(shapes))
    if err is not None:
        return err
    assert dims is not None

    body_inputs: List[Shape] = []
    batch_dims: List[Dim] = []
    mapped_any = False
    for shape, dim in zip(shapes, dims):
        if dim is None:
            body_inputs.append(shape)
            continue
        mapped_any = True
        resolved, dim_err = _normalise_dim(len(shape), dim, kind="in_dim")
        if dim_err is not None:
            return dim_err
        assert resolved is not None
        batch_dims.append(shape[resolved])
        body_inputs.append(shape[:resolved] + shape[resolved + 1:])

    if not mapped_any:
        return _fail("no_mapped_inputs", "vmap requires at least one mapped tensor input")

    concrete = [d for d in batch_dims if isinstance(d, int)]
    if concrete and any(d != concrete[0] for d in concrete):
        return _fail(
            "batch_size",
            f"mapped input batch sizes disagree: {tuple(batch_dims)}",
        )
    batch_dim = batch_dims[0]

    out_tree, err = _normalise_out_dims(body_output_shapes, out_dims)
    if err is not None:
        return err
    batched_tree, err = _normalise_bool_tree(body_output_shapes, body_output_batched)
    if err is not None:
        return err

    output_shapes, err, unknown = _map_output_shape(
        body_output_shapes,
        out_tree,
        batched_tree,
        batch_dim,
    )
    if err is not None:
        return err
    if unknown is not None:
        return _unknown(batch_dim, body_inputs, unknown)

    return VmapVerdict(
        True,
        batch_dim=batch_dim,
        body_input_shapes=tuple(body_inputs),
        output_shapes=output_shapes,
    )
