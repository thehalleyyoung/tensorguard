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

# Step 55: instrumentation for the solver-avoidance optimization. Counts how
# many reshape compatibility checks fell through to an actual Z3 `solver.check()`
# (as opposed to being decided analytically by constant folding / shared-factor
# cancellation). Tests assert that the concrete and single-infer hot paths cost
# zero solver calls.
_SOLVER_CALLS = [0]
_ANALYTIC_DECISIONS = [0]


def reshape_solver_call_count() -> int:
    return _SOLVER_CALLS[0]


def reshape_analytic_decision_count() -> int:
    return _ANALYTIC_DECISIONS[0]


def reset_reshape_counters() -> None:
    _SOLVER_CALLS[0] = 0
    _ANALYTIC_DECISIONS[0] = 0

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

    incompatible_msg = (
        f"Reshape incompatible: cannot reshape {input_shape} to "
        f"{_format_dims(new_dims)} (element count cannot be preserved "
        f"for any valid dimension sizes)"
    )

    # ---- Step 55: algebraic reduction before any nonlinear solving --------
    # The element-count equation is prod(inputs) == prod(targets) with every
    # dimension >= 1. Products of integer variables are *nonlinear* integer
    # arithmetic, which is what makes Z3 blow up on high-rank reshapes. But the
    # equation factors cleanly: fold concrete dims into a single integer, and
    # since every shared symbolic dim (same name) appears as the *same* variable
    # on both sides and is >= 1 (hence nonzero), it can be cancelled from both
    # products without changing satisfiability. After folding + cancellation
    # many real reshapes collapse to a pure integer (divisibility) test that
    # needs no solver at all; only a genuinely under-determined symbolic
    # remainder falls through to Z3 — on a much smaller constraint.
    #
    # Underscore-prefixed names (``_dyn0`` etc.) are engine-internal opaque
    # values that are NOT guaranteed equal across occurrences (see ``var_for``),
    # so they must NOT be cancelled by name; each is an independent free factor.
    from collections import Counter

    in_lit = 1
    in_named: "Counter[str]" = Counter()
    in_free = 0
    for d in input_shape.dims:
        if d.is_symbolic:
            name = str(d.value)
            if name.startswith("_"):
                in_free += 1
            else:
                in_named[name] += 1
        else:
            in_lit *= int(d.value)

    out_lit = 1
    out_named: "Counter[str]" = Counter()
    out_free = 0
    infer_count = 0
    for kind, val in resolved:
        if kind == "lit":
            out_lit *= int(val)
        elif kind == "sym":
            name = str(val)
            if name.startswith("_"):
                out_free += 1
            else:
                out_named[name] += 1
        else:  # infer
            infer_count += 1

    # Cancel shared named symbolic factors (all >= 1, so cancellation is exact).
    for name in list((in_named & out_named).keys()):
        shared = min(in_named[name], out_named[name])
        in_named[name] -= shared
        out_named[name] -= shared
        if in_named[name] == 0:
            del in_named[name]
        if out_named[name] == 0:
            del out_named[name]

    no_remaining_vars = (not in_named and not out_named
                         and in_free == 0 and out_free == 0)

    if no_remaining_vars:
        # Pure integer decision — no solver needed.
        _ANALYTIC_DECISIONS[0] += 1
        if infer_count == 0:
            return None if in_lit == out_lit else incompatible_msg
        # Exactly one inferred (-1) dimension: in_lit == out_lit * infer with
        # infer >= 1. Satisfiable iff out_lit divides in_lit and the quotient
        # is >= 1 (both literals are >= 1, so quotient >= 1 iff in_lit >= out_lit).
        if out_lit != 0 and in_lit % out_lit == 0 and in_lit >= out_lit:
            return None
        return incompatible_msg

    # ---- Reduced symbolic remainder: solve the *cancelled* equation --------
    vars_by_name = {}
    constraints = []
    fresh_counter = [0]

    def fresh_var(tag: str):
        v = z3.Int(f"rs_free_{fresh_counter[0]}_{_sanitize(tag)}")
        fresh_counter[0] += 1
        constraints.append(v >= 1)
        return v

    def var_for(name: str):
        if name not in vars_by_name:
            v = z3.Int(f"rs_{len(vars_by_name)}_{_sanitize(name)}")
            vars_by_name[name] = v
            constraints.append(v >= 1)
        return vars_by_name[name]

    in_prod = z3.IntVal(in_lit)
    for name, cnt in in_named.items():
        for _ in range(cnt):
            in_prod = in_prod * var_for(name)
    for _ in range(in_free):
        in_prod = in_prod * fresh_var("in_free")

    out_prod = z3.IntVal(out_lit)
    for name, cnt in out_named.items():
        for _ in range(cnt):
            out_prod = out_prod * var_for(name)
    for _ in range(out_free):
        out_prod = out_prod * fresh_var("out_free")
    for i in range(infer_count):
        inf = z3.Int(f"rs_infer_{i}")
        constraints.append(inf >= 1)
        out_prod = out_prod * inf

    solver = z3.Solver()
    solver.set("timeout", 3000)
    for c in constraints:
        solver.add(c)
    solver.add(in_prod == out_prod)

    _SOLVER_CALLS[0] += 1
    result = solver.check()
    if result == z3.unsat:
        return incompatible_msg
    return None


def _format_dims(new_dims: Tuple) -> str:
    return "(" + ", ".join(str(d) for d in new_dims) + ")"
