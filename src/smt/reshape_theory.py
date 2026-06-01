"""
Z3-backed reshape / view element-count compatibility checking.

A ``reshape`` / ``view`` is valid iff the **total number of elements is
preserved**.  Concretely, for an input of shape ``(i0, i1, ...)`` and a target
spec ``(t0, t1, ...)`` (possibly containing one ``-1`` to be inferred), the
reshape succeeds iff there exists an integer assignment of the symbolic
dimensions (every tensor dimension is ``>= 1``) and of the inferred ``-1``
dimension (``>= 1``) such that::

    prod(i0, i1, ...) == prod(specified t_k) * inferred

We model this directly in Z3 with the **same** integer variable shared between
the input and the output whenever a dimension has the same symbolic name (so a
copied/aliased dim such as the batch ``B`` is coupled on both sides).  We flag
the reshape as incompatible **only when this equation is UNSAT** over all
``dim >= 1`` assignments — i.e. provably impossible for every concretization.
This guarantees soundness (no false positives) per ``SOUNDNESS_CONTRACT.md``:
when Z3 returns ``sat`` or ``unknown`` (timeout) we abstain.

Sentinel handling mirrors ``tensor_shapes.compute_reshape_shape``:

  * ``0``       — copy this dimension from the corresponding input dim (index).
  * ``<= -2``   — copy from input dim ``-d - 2`` (``B, C, H, W = x.shape`` form).
  * ``-1``      — infer this dimension (standard PyTorch ``-1``).

A target spec containing **more than one** ``-1`` (after sentinel resolution) is
always a runtime error in PyTorch and is reported as incompatible.

``TensorShape`` / ``ShapeDim`` are the engine's own classes
(``src.tensor_shapes``).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple, Union

try:
    import z3

    HAS_Z3 = True
except ImportError:  # pragma: no cover - z3 always available in CI
    HAS_Z3 = False

from src.tensor_shapes import ShapeDim, TensorShape

# A resolved target entry is one of:
#   ("lit", int)   — a concrete, specified size
#   ("sym", name)  — a symbolic dim (shares a Z3 var with equal-named dims)
#   ("infer", None)— a ``-1`` dimension to be inferred
_ResolvedEntry = Tuple[str, Union[int, str, None]]


def _resolve_target(
    input_shape: TensorShape, new_dims: Tuple
) -> Tuple[Optional[List[_ResolvedEntry]], Optional[str]]:
    """Resolve sentinel values in ``new_dims`` against ``input_shape``.

    Returns ``(entries, error)``.  ``error`` is set (and ``entries`` is None)
    only for specs that are *always* invalid (more than one ``-1``).
    """
    resolved: List[_ResolvedEntry] = []
    infer_count = 0
    for d in new_dims:
        if isinstance(d, str):
            resolved.append(("sym", d))
            continue
        if not isinstance(d, int):
            # Unknown entry type — treat as a fresh symbolic (abstain-friendly).
            resolved.append(("sym", f"_unk_target_{len(resolved)}"))
            continue
        if d == -1:
            resolved.append(("infer", None))
            infer_count += 1
        elif d == 0:
            # Copy input dim at this position.
            idx = len(resolved)
            if idx < input_shape.ndim:
                resolved.append(_dim_entry(input_shape.dims[idx]))
            else:
                # Out of bounds — cannot copy; treat as infer (unknown).
                resolved.append(("infer", None))
                infer_count += 1
        elif d <= -2:
            src_k = -d - 2
            if 0 <= src_k < input_shape.ndim:
                resolved.append(_dim_entry(input_shape.dims[src_k]))
            else:
                resolved.append(("infer", None))
                infer_count += 1
        else:
            resolved.append(("lit", d))

    if infer_count > 1:
        return None, (
            "only one dimension can be inferred (-1) in a reshape, "
            f"but the target {tuple(new_dims)} specifies {infer_count}"
        )
    return resolved, None


def _dim_entry(dim: ShapeDim) -> _ResolvedEntry:
    if dim.is_symbolic:
        return ("sym", str(dim.value))
    return ("lit", int(dim.value))


def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


def check_reshape_compatible(
    input_shape: TensorShape, new_dims: Tuple
) -> Optional[str]:
    """Return an error message iff the reshape is *provably* incompatible.

    Returns ``None`` when the reshape is satisfiable (compatible for some
    concretization), when Z3 cannot decide (timeout), or when Z3 is
    unavailable.  This is sound: a non-None result means there is **no**
    assignment of dimensions ``>= 1`` that preserves the element count.
    """
    if not HAS_Z3:
        return None
    if input_shape is None or input_shape.ndim == 0:
        return None
    # A genuine zero-size input dimension makes element-count reasoning
    # degenerate; abstain rather than risk a confusing report.
    for d in input_shape.dims:
        if not d.is_symbolic and int(d.value) == 0:
            return None

    resolved, err = _resolve_target(input_shape, new_dims)
    if err is not None:
        return f"Reshape invalid: {err}"
    assert resolved is not None

    vars_by_name = {}
    constraints = []
    fresh_counter = [0]

    def fresh_var(tag: str):
        v = z3.Int(f"rs_free_{fresh_counter[0]}_{_sanitize(tag)}")
        fresh_counter[0] += 1
        constraints.append(v >= 1)
        return v

    def var_for(name: str):
        # Opaque / engine-internal symbolic names (``_dyn0``, ``_flat``,
        # ``_inferred``, ``_unk_...``) represent *untracked* unknown runtime
        # values.  Two such occurrences are NOT guaranteed to be the same
        # value — and placeholder names are only locally unique per extracted
        # op, so they collide across different reshapes (e.g. ShuffleNet's
        # channel-shuffle produces input ``(_dyn0, 2, _dyn1, ...)`` and a
        # later target ``(_dyn0, _dyn1, ...)`` whose ``_dynN`` are unrelated).
        # Coupling them would yield spurious UNSAT (false positives), so each
        # underscore-prefixed occurrence gets its own fresh, independent var.
        # Genuinely shared, semantically-meaningful dims (``B``, ``C``, ...)
        # are coupled by name so true incompatibilities like (B,5)->(B,3) are
        # still provable.
        if name.startswith("_"):
            return fresh_var(name)
        if name not in vars_by_name:
            v = z3.Int(f"rs_{len(vars_by_name)}_{_sanitize(name)}")
            vars_by_name[name] = v
            constraints.append(v >= 1)
        return vars_by_name[name]

    in_prod = z3.IntVal(1)
    for d in input_shape.dims:
        if d.is_symbolic:
            in_prod = in_prod * var_for(str(d.value))
        else:
            in_prod = in_prod * z3.IntVal(int(d.value))

    out_prod = z3.IntVal(1)
    infer_idx = 0
    for kind, val in resolved:
        if kind == "lit":
            out_prod = out_prod * z3.IntVal(int(val))
        elif kind == "sym":
            out_prod = out_prod * var_for(str(val))
        else:  # infer
            inf = z3.Int(f"rs_infer_{infer_idx}")
            infer_idx += 1
            constraints.append(inf >= 1)
            out_prod = out_prod * inf

    solver = z3.Solver()
    solver.set("timeout", 3000)
    for c in constraints:
        solver.add(c)
    solver.add(in_prod == out_prod)

    result = solver.check()
    if result == z3.unsat:
        return (
            f"Reshape incompatible: cannot reshape {input_shape} to "
            f"{_format_dims(new_dims)} (element count cannot be preserved "
            f"for any valid dimension sizes)"
        )
    return None


def _format_dims(new_dims: Tuple) -> str:
    return "(" + ", ".join(str(d) for d in new_dims) + ")"
