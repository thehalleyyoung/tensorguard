"""v5 / Track-C — SMT-checked divisibility for ``reshape(..., -1, ...)``.

PyTorch's ``view`` / ``reshape`` accept a single ``-1`` placeholder whose
value is inferred at runtime from the constraint ``numel(input) ==
prod(target)``.  When the input has symbolic dims this is no longer a
concrete check — but it *is* still decidable in the divisibility
fragment of linear integer arithmetic.

This module asks Z3:

    For every positive assignment of the symbolic dims in the input,
    is  prod(input_dims)  divisible by  prod(target_dims excluding -1)?

If unsat → safe (the reshape is always well-defined).
If sat   → counterexample shape that breaks the reshape.
If unknown → abstain (caller's choice whether to warn).

A second mode answers the *concrete-input* form ``check_concrete``: given
ints for both sides it returns a Boolean directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.tensor_shapes import ShapeDim, TensorShape
from src.v5.symbolic_config import SymInt, SymExpr

try:
    import z3
    HAS_Z3 = True
except Exception:  # pragma: no cover
    HAS_Z3 = False


Verdict = str  # one of {"safe", "unsafe", "unknown"}


@dataclass
class ReshapeCheck:
    verdict: Verdict
    counterexample: Optional[Dict[str, int]] = None
    inferred_neg1: Optional[Union[int, str]] = None
    detail: str = ""


# ────────────────────────────────────────────────────────────────────────────
# Internal: convert ShapeDim / SymInt / int / str → Z3 int expression
# ────────────────────────────────────────────────────────────────────────────

def _z3_for_dim(d: Any, env: Dict[str, Any]) -> Any:
    """Build a Z3 integer expression for ``d``.

    ``env`` caches ``str → z3.Int`` so that repeated occurrences of the
    same symbolic name unify.
    """
    if isinstance(d, ShapeDim):
        d = d.value
    if isinstance(d, int):
        return z3.IntVal(d)
    if isinstance(d, SymInt):
        return env.setdefault(d.name, z3.Int(d.name))
    if isinstance(d, SymExpr):
        a = _z3_for_dim(d.args[0], env)
        b = _z3_for_dim(d.args[1], env)
        if d.op == "*":
            return a * b
        if d.op == "+":
            return a + b
        if d.op == "//":
            return a / b   # z3 ints: integer division
        if d.op == "%":
            return a % b
        raise ValueError(f"unsupported SymExpr op {d.op}")
    if isinstance(d, str):
        # Attempt naive parsing for "(a*b)" style strings produced by
        # other v5 modules.  Anything more interesting → fresh Z3 var.
        return env.setdefault(d, z3.Int(_safe_name(d)))
    raise TypeError(f"cannot convert {d!r} ({type(d).__name__}) to z3 int")


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _collect_symbols(dims: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    def visit(d: Any) -> None:
        if isinstance(d, ShapeDim):
            visit(d.value)
        elif isinstance(d, SymInt):
            if d.name not in seen:
                seen.add(d.name); out.append(d.name)
        elif isinstance(d, SymExpr):
            for a in d.args: visit(a)
        elif isinstance(d, str):
            n = _safe_name(d)
            if n not in seen:
                seen.add(n); out.append(n)
    for d in dims: visit(d)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def check_reshape_divisibility(
    input_shape: Union[TensorShape, Sequence[Any]],
    target_dims: Sequence[Any],
    timeout_ms: int = 2000,
) -> ReshapeCheck:
    """Check that ``reshape(input_shape, target_dims)`` is always sound,
    given any positive assignment of the symbolic dims that appear.

    ``target_dims`` may contain at most one ``-1``.  Other entries may
    be ints, :class:`SymInt`, or :class:`SymExpr`.
    """
    if isinstance(input_shape, TensorShape):
        in_dims_raw = list(input_shape.dims)
    else:
        in_dims_raw = list(input_shape)

    tgt = list(target_dims)
    neg1_positions = [i for i, d in enumerate(tgt) if isinstance(d, int) and d == -1]
    if len(neg1_positions) > 1:
        return ReshapeCheck("unsafe", detail="multiple -1 in target")

    if not HAS_Z3:
        return ReshapeCheck("unknown", detail="z3 not available")

    env: Dict[str, Any] = {}

    # Build P (input numel) and Q (product of non-(-1) target dims).
    P = z3.IntVal(1)
    for d in in_dims_raw:
        P = P * _z3_for_dim(d, env)

    Q = z3.IntVal(1)
    for i, d in enumerate(tgt):
        if i in neg1_positions:
            continue
        Q = Q * _z3_for_dim(d, env)

    # Positivity for every named symbol.
    pos_constraints = [v > 0 for v in env.values()]

    s = z3.Solver()
    s.set("timeout", timeout_ms)
    for c in pos_constraints:
        s.add(c)

    # Q must be > 0 (otherwise dividing by zero).  Add the *negation* of
    # Q > 0 as an unsat-check: if sat, then there's a positive symbol
    # assignment with Q ≤ 0.
    qpos = z3.Solver(); qpos.set("timeout", timeout_ms)
    for c in pos_constraints: qpos.add(c)
    qpos.add(Q <= 0)
    if qpos.check() == z3.sat:
        m = qpos.model()
        cex = {}
        for d in m.decls():
            v = m[d]
            try:
                cex[str(d)] = v.as_long()
            except Exception:
                pass
        return ReshapeCheck("unsafe", counterexample=cex,
                            detail="target product ≤ 0")

    # Search for a counterexample: P % Q != 0.
    s.add(P % Q != 0)
    res = s.check()
    if res == z3.sat:
        m = s.model()
        cex = {}
        for d in m.decls():
            v = m[d]
            try:
                cex[str(d)] = v.as_long()
            except Exception:
                cex[str(d)] = str(v)
        return ReshapeCheck("unsafe", counterexample=cex,
                            detail="found positive assignment violating divisibility")
    if res == z3.unsat:
        # Compute symbolic value of -1 if any.
        inferred: Optional[Union[int, str]] = None
        if neg1_positions:
            # Try to simplify P / Q.
            try:
                expr = z3.simplify(P / Q)
                inferred = str(expr)
            except Exception:
                inferred = None
        return ReshapeCheck("safe", inferred_neg1=inferred,
                            detail="divisibility holds for all positive assignments")
    return ReshapeCheck("unknown", detail=f"z3 returned {res}")


def check_concrete(
    input_shape: Sequence[int],
    target_dims: Sequence[int],
) -> ReshapeCheck:
    """Fast path: when both sides are fully concrete ints (with at most
    one -1), check divisibility directly without invoking Z3.
    """
    if any(not isinstance(d, int) for d in input_shape):
        return ReshapeCheck("unknown", detail="non-int in input")
    if any(not isinstance(d, int) for d in target_dims):
        return ReshapeCheck("unknown", detail="non-int in target")
    P = 1
    for d in input_shape: P *= d
    neg1 = [i for i, d in enumerate(target_dims) if d == -1]
    if len(neg1) > 1:
        return ReshapeCheck("unsafe", detail="multiple -1")
    Q = 1
    for i, d in enumerate(target_dims):
        if i not in neg1:
            if d <= 0:
                return ReshapeCheck("unsafe", detail=f"non-positive dim {d}")
            Q *= d
    if not neg1:
        if P != Q:
            return ReshapeCheck("unsafe", detail=f"numel mismatch {P} vs {Q}")
        return ReshapeCheck("safe")
    if P % Q != 0:
        return ReshapeCheck("unsafe", detail=f"{P} not divisible by {Q}")
    return ReshapeCheck("safe", inferred_neg1=P // Q)


__all__ = ["ReshapeCheck", "check_reshape_divisibility", "check_concrete"]
