"""Source-level einops bug finder built on :mod:`src.einops_verify`.

Walks a module / function source, tracks the shapes of tensors that flow
through ``einops.rearrange`` / ``reduce`` / ``repeat`` calls with literal
patterns, and reports a :class:`~src.api.Bug` for every call that real einops
would reject (non-divisible decomposition, under-determined split, axis-set
mismatch, missing repeat length, …).

This is the integration that turns the differentially-verified static model in
:mod:`src.einops_verify` into a check over *real* model code:

    from src.einops_source import verify_einops_source
    res = verify_einops_source(open("model.py").read(),
                               input_shapes={"x": (12, 5)})
    for bug in res.bugs:
        print(bug.message)

Only calls whose tensor argument has a fully resolved shape and whose pattern
is a string literal are checked; everything else is conservatively skipped, so
the checker never raises a false positive on code it cannot understand.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple, Union

from src.api import Bug, BugCategory, SourceLocation
from src.einops_verify import Dim, verify_einops

__all__ = ["verify_einops_source", "find_einops_bugs"]

_EINOPS_OPS = {"rearrange", "reduce", "repeat"}

Shape = Tuple[Dim, ...]


def _literal(node: ast.AST) -> Optional[object]:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _call_op(node: ast.Call) -> Optional[str]:
    """Return 'rearrange'/'reduce'/'repeat' if this is such a call."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in _EINOPS_OPS:
        return func.id
    if isinstance(func, ast.Attribute) and func.attr in _EINOPS_OPS:
        # einops.rearrange(...) / ein.rearrange(...)
        return func.attr
    return None


def find_einops_bugs(
    source: str,
    input_shapes: Dict[str, Shape],
    filename: str = "<source>",
) -> List[Bug]:
    """Return every einops bug found in ``source`` given seed input shapes."""
    tree = ast.parse(source)
    bugs: List[Bug] = []

    # var name -> resolved shape (ints / symbolic strings)
    env: Dict[str, Shape] = {k: tuple(v) for k, v in input_shapes.items()}

    def resolve_axes(node: ast.Call) -> Dict[str, int]:
        axes: Dict[str, int] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            val = _literal(kw.value)
            if isinstance(val, int):
                axes[kw.arg] = val
        return axes

    def shape_of_arg(node: ast.AST) -> Optional[Shape]:
        if isinstance(node, ast.Name):
            return env.get(node.id)
        return None

    def check_call(node: ast.Call) -> Optional[Shape]:
        op = _call_op(node)
        if op is None or len(node.args) < 2:
            return None
        tensor_shape = shape_of_arg(node.args[0])
        pattern = _literal(node.args[1])
        if tensor_shape is None or not isinstance(pattern, str):
            return None
        axes = resolve_axes(node)
        verdict = verify_einops(op, pattern, tensor_shape, **axes)
        if not verdict.ok:
            bugs.append(
                Bug(
                    category=BugCategory.TYPE_ERROR,
                    message=(
                        f"einops.{op}({pattern!r}) is invalid for input shape "
                        f"{tuple(tensor_shape)}: {verdict.error}"
                    ),
                    location=SourceLocation(
                        file=filename,
                        line=getattr(node, "lineno", 0),
                        column=getattr(node, "col_offset", 0),
                    ),
                    severity="error",
                    confidence=0.95,
                    fix_suggestion=_suggest(verdict.error_kind, op),
                ),
            )
            return None
        return verdict.output_shape

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            out: Optional[Shape] = None
            if isinstance(node.value, ast.Call):
                out = check_call(node.value)
            self.generic_visit(node)
            if out is not None and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                env[node.targets[0].id] = out

        def visit_Call(self, node: ast.Call) -> None:
            # standalone calls (not captured by an assignment) still get checked
            parent_is_assign_value = getattr(node, "_tg_seen", False)
            if not parent_is_assign_value:
                check_call(node)
            self.generic_visit(node)

    # Mark calls that are the RHS of a simple assignment so we don't double-check
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            n.value._tg_seen = True  # type: ignore[attr-defined]

    Visitor().visit(tree)
    # de-duplicate by (line, message)
    seen = set()
    unique: List[Bug] = []
    for b in bugs:
        key = (b.location.line, b.message)
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def _suggest(kind: Optional[str], op: str) -> Optional[str]:
    return {
        "non_divisible": "Ensure the decomposed axis length is divisible by the "
        "given sub-axis size.",
        "underdetermined": "Give all but one sub-axis of the group an explicit "
        "length kwarg.",
        "axis_set_mismatch": f"Every named axis must appear on both sides of a "
        f"{op} pattern (except axes introduced by repeat / removed by reduce).",
        "missing_length": "Pass the new axis size as a keyword argument, e.g. "
        f"{op}(x, pattern, n=...).",
        "rank_mismatch": "The number of pattern axes must match the tensor rank.",
        "duplicate": "An axis name may appear at most once per side.",
    }.get(kind or "")


def verify_einops_source(
    source: str,
    input_shapes: Dict[str, Shape],
    filename: str = "<source>",
):
    """Convenience wrapper returning an :class:`~src.api.AnalysisResult`."""
    from src.api import AnalysisResult

    bugs = find_einops_bugs(source, input_shapes, filename)
    return AnalysisResult(bugs=bugs, functions_analyzed=1,
                          lines_analyzed=source.count("\n") + 1)
