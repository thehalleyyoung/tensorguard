"""Static verifier for PyTorch named-tensor ``refine_names`` / ``align_to``.

Named tensors turn axis order into a contract: ``align_to("C", "N")``
permutes dimensions by name, ``align_to("N", "C", "H")`` inserts singleton
axes, and ``refine_names`` may only fill previously-unnamed dimensions.  A
wrong name alignment raises at runtime, often far from the source of the bug.

This module models those two contracts without constructing a tensor.  It is
differentially tested against real PyTorch named tensors in
``tests/test_named_tensor_verify.py``.  Symbolic sizes are carried through
unchanged: the checker branches only on rank and names, never on dimension
values.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from src.api import AnalysisResult, Bug, BugCategory, SourceLocation

Dim = Union[int, str]
Shape = Tuple[Dim, ...]
Name = Optional[str]
Names = Tuple[Name, ...]
NameToken = Union[str, None, type(Ellipsis)]

__all__ = [
    "NamedTensorSpec",
    "NamedTensorVerdict",
    "verify_refine_names",
    "verify_align_to",
    "find_named_tensor_bugs",
    "verify_named_tensor_source",
]


@dataclass(frozen=True)
class NamedTensorSpec:
    """Shape and names carried by a named tensor."""

    shape: Shape
    names: Names


@dataclass
class NamedTensorVerdict:
    """Result of checking one named-tensor operation."""

    ok: bool
    spec: Optional[NamedTensorSpec] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _ok(shape: Sequence[Dim], names: Sequence[Name]) -> NamedTensorVerdict:
    return NamedTensorVerdict(True, spec=NamedTensorSpec(tuple(shape), tuple(names)))


def _fail(kind: str, message: str) -> NamedTensorVerdict:
    return NamedTensorVerdict(False, error=message, error_kind=kind)


def _valid_name(name: str) -> bool:
    if name == "...":
        return True
    if not name or name[0].isdigit():
        return False
    return all(ch == "_" or ch.isalnum() for ch in name)


def _normalise_token(token: object, *, allow_ellipsis: bool) -> Tuple[Optional[object], Optional[str]]:
    if token is Ellipsis or token == "...":
        if allow_ellipsis:
            return Ellipsis, None
        return None, "ellipsis is not valid in an existing tensor name"
    if token is None:
        return None, None
    if isinstance(token, str):
        if _valid_name(token):
            return token, None
        return None, (
            "invalid name: names must start with a non-digit and contain only "
            "letters, digits, or '_'"
        )
    return None, f"invalid name token {token!r}"


def _normalise_tokens(
    tokens: Sequence[object],
    *,
    allow_ellipsis: bool,
) -> Tuple[Optional[List[object]], Optional[NamedTensorVerdict]]:
    out: List[object] = []
    for token in tokens:
        normalised, error = _normalise_token(token, allow_ellipsis=allow_ellipsis)
        if error is not None:
            return None, _fail("invalid_name", error)
        out.append(normalised)
    return out, None


def _duplicate_name(names: Iterable[object]) -> Optional[str]:
    seen = set()
    for name in names:
        if name is None or name is Ellipsis:
            continue
        if name in seen:
            return str(name)
        seen.add(name)
    return None


def _validate_spec(
    shape: Sequence[Dim],
    names: Sequence[object],
) -> Tuple[Optional[NamedTensorSpec], Optional[NamedTensorVerdict]]:
    normalised, error = _normalise_tokens(names, allow_ellipsis=False)
    if error is not None:
        return None, error
    assert normalised is not None
    if len(shape) != len(normalised):
        return None, _fail(
            "rank",
            f"shape rank {len(shape)} and names rank {len(normalised)} differ",
        )
    duplicate = _duplicate_name(normalised)
    if duplicate is not None:
        return None, _fail("duplicate", f"duplicate tensor name {duplicate!r}")
    return NamedTensorSpec(tuple(shape), tuple(normalised)), None


def _ellipsis_count(tokens: Sequence[object]) -> int:
    return sum(1 for token in tokens if token is Ellipsis)


def verify_refine_names(
    shape: Sequence[Dim],
    current_names: Sequence[NameToken],
    requested_names: Sequence[NameToken],
) -> NamedTensorVerdict:
    """Verify ``tensor.refine_names(*requested_names)``.

    Existing concrete names may only be kept.  Unnamed dimensions may be filled
    with a concrete name or left unnamed.  At most one Ellipsis is allowed; it
    expands to the corresponding span of the existing names.
    """

    spec, error = _validate_spec(shape, current_names)
    if error is not None:
        return error
    assert spec is not None

    requested, error = _normalise_tokens(requested_names, allow_ellipsis=True)
    if error is not None:
        return error
    assert requested is not None

    if _ellipsis_count(requested) > 1:
        return _fail("ellipsis", "refine_names supports at most one Ellipsis")

    rank = len(spec.shape)
    if Ellipsis in requested:
        explicit = [name for name in requested if name is not Ellipsis]
        if len(explicit) > rank:
            return _fail(
                "rank",
                f"requested {len(explicit)} explicit names for rank-{rank} tensor",
            )
        ellipsis_index = requested.index(Ellipsis)
        span = rank - len(explicit)
        expanded = (
            requested[:ellipsis_index]
            + list(spec.names[ellipsis_index:ellipsis_index + span])
            + requested[ellipsis_index + 1:]
        )
    else:
        expanded = list(requested)

    if len(expanded) != rank:
        return _fail(
            "rank",
            f"refine_names expects {rank} names, got {len(expanded)}",
        )

    duplicate = _duplicate_name(expanded)
    if duplicate is not None:
        return _fail("duplicate", f"duplicate output name {duplicate!r}")

    final: List[Name] = []
    for old, new in zip(spec.names, expanded):
        assert new is None or isinstance(new, str)
        if old is None:
            final.append(new)
        elif new == old:
            final.append(old)
        elif new is None:
            return _fail(
                "demotion",
                f"cannot refine existing name {old!r} to unnamed None",
            )
        else:
            return _fail(
                "rename",
                f"cannot refine existing name {old!r} to different name {new!r}",
            )
    return _ok(spec.shape, final)


def verify_align_to(
    shape: Sequence[Dim],
    current_names: Sequence[NameToken],
    target_names: Sequence[NameToken],
) -> NamedTensorVerdict:
    """Verify ``tensor.align_to(*target_names)``.

    Existing named dimensions are reordered by name.  Target names not present
    in the input insert singleton dimensions.  Without Ellipsis every input
    dimension must be named and every existing name must appear explicitly.  With
    Ellipsis, unnamed dimensions and any named dimensions omitted from the target
    are carried through in their original order at the Ellipsis position.
    """

    spec, error = _validate_spec(shape, current_names)
    if error is not None:
        return error
    assert spec is not None

    target, error = _normalise_tokens(target_names, allow_ellipsis=True)
    if error is not None:
        return error
    assert target is not None

    if _ellipsis_count(target) > 1:
        return _fail("ellipsis", "align_to supports at most one Ellipsis")

    duplicate = _duplicate_name(target)
    if duplicate is not None:
        return _fail("duplicate", f"duplicate target name {duplicate!r}")

    has_ellipsis = Ellipsis in target
    if has_ellipsis and any(name is None for name in target):
        return _fail(
            "none_with_ellipsis",
            "align_to targets that use Ellipsis cannot also contain explicit None names",
        )

    explicit_named = {
        name for name in target
        if name is not None and name is not Ellipsis
    }

    if not has_ellipsis:
        for i, name in enumerate(spec.names):
            if name is None:
                return _fail(
                    "unnamed_dim",
                    f"align_to requires Ellipsis to carry unnamed dim at index {i}",
                )
        for name in spec.names:
            if name is not None and name not in explicit_named:
                return _fail(
                    "missing_name",
                    f"align_to target does not mention existing dim {name!r}",
                )

    by_name: Dict[str, Tuple[Dim, str]] = {}
    for dim, name in zip(spec.shape, spec.names):
        if name is not None:
            by_name[name] = (dim, name)

    ellipsis_block: List[Tuple[Dim, Name]] = []
    if has_ellipsis:
        for dim, name in zip(spec.shape, spec.names):
            if name is None or name not in explicit_named:
                ellipsis_block.append((dim, name))

    out_shape: List[Dim] = []
    out_names: List[Name] = []
    for token in target:
        if token is Ellipsis:
            for dim, name in ellipsis_block:
                out_shape.append(dim)
                out_names.append(name)
        elif token is None:
            out_shape.append(1)
            out_names.append(None)
        else:
            assert isinstance(token, str)
            if token in by_name:
                dim, name = by_name[token]
                out_shape.append(dim)
                out_names.append(name)
            else:
                out_shape.append(1)
                out_names.append(token)

    duplicate = _duplicate_name(out_names)
    if duplicate is not None:
        return _fail("duplicate", f"duplicate output name {duplicate!r}")
    return _ok(out_shape, out_names)


def _name_args(args: Sequence[ast.AST]) -> Optional[List[NameToken]]:
    out: List[NameToken] = []
    for arg in args:
        if isinstance(arg, ast.Constant):
            if arg.value is Ellipsis:
                out.append(Ellipsis)
            elif arg.value is None or isinstance(arg.value, str):
                out.append(arg.value)
            else:
                return None
        else:
            return None
    return out


def _coerce_input_specs(
    input_specs: Mapping[str, Union[NamedTensorSpec, Tuple[Sequence[Dim], Sequence[NameToken]]]],
) -> Dict[str, NamedTensorSpec]:
    env: Dict[str, NamedTensorSpec] = {}
    for name, value in input_specs.items():
        if isinstance(value, NamedTensorSpec):
            spec, error = _validate_spec(value.shape, value.names)
        else:
            shape, names = value
            spec, error = _validate_spec(shape, names)
        if error is None and spec is not None:
            env[name] = spec
    return env


def _suggest(kind: Optional[str]) -> Optional[str]:
    return {
        "rank": "Pass exactly one name per tensor dimension, or use a single Ellipsis.",
        "duplicate": "Named tensor dimensions must have unique non-None names.",
        "demotion": "Use rename(None) to drop names; refine_names can only make names more specific.",
        "rename": "Use refine_names only to fill unnamed dimensions; existing names must match.",
        "unnamed_dim": "Include Ellipsis in align_to so unnamed dimensions are carried through.",
        "missing_name": "Mention every existing named dimension, or cover omitted names with Ellipsis.",
        "none_with_ellipsis": "Use Ellipsis to carry unnamed input dims; insert explicit None axes only without Ellipsis.",
        "invalid_name": "Use valid identifier-like names, e.g. 'batch' or 'channels'.",
    }.get(kind or "")


def find_named_tensor_bugs(
    source: str,
    input_specs: Mapping[str, Union[NamedTensorSpec, Tuple[Sequence[Dim], Sequence[NameToken]]]],
    filename: str = "<source>",
) -> List[Bug]:
    """Return named-tensor alignment bugs found in ``source``.

    The source checker is deliberately conservative: it tracks variables seeded
    in ``input_specs`` through literal ``.refine_names(...)`` and ``.align_to(...)``
    calls, and skips dynamic name expressions.
    """

    tree = ast.parse(source)
    env = _coerce_input_specs(input_specs)
    bugs: List[Bug] = []

    def evaluate(node: ast.AST) -> Optional[NamedTensorSpec]:
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            base = evaluate(node.func.value)
            if base is None:
                return None
            names = _name_args(node.args)
            if names is None:
                return None
            if node.func.attr == "refine_names":
                verdict = verify_refine_names(base.shape, base.names, names)
            elif node.func.attr == "align_to":
                verdict = verify_align_to(base.shape, base.names, names)
            else:
                return None
            if verdict.ok:
                return verdict.spec
            bugs.append(
                Bug(
                    category=BugCategory.TYPE_ERROR,
                    message=(
                        f"named tensor {node.func.attr} is invalid for "
                        f"shape {base.shape} with names {base.names}: {verdict.error}"
                    ),
                    location=SourceLocation(
                        file=filename,
                        line=getattr(node, "lineno", 0),
                        column=getattr(node, "col_offset", 0),
                    ),
                    severity="error",
                    confidence=0.95,
                    fix_suggestion=_suggest(verdict.error_kind),
                )
            )
            return None
        return None

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            spec = evaluate(node.value)
            self.generic_visit(node)
            if spec is not None and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                env[node.targets[0].id] = spec

        def visit_Return(self, node: ast.Return) -> None:
            if node.value is not None:
                evaluate(node.value)
            self.generic_visit(node)

        def visit_Expr(self, node: ast.Expr) -> None:
            evaluate(node.value)
            self.generic_visit(node)

    Visitor().visit(tree)

    seen = set()
    unique: List[Bug] = []
    for bug in bugs:
        key = (bug.location.line, bug.location.column, bug.message)
        if key not in seen:
            seen.add(key)
            unique.append(bug)
    return unique


def verify_named_tensor_source(
    source: str,
    input_specs: Mapping[str, Union[NamedTensorSpec, Tuple[Sequence[Dim], Sequence[NameToken]]]],
    filename: str = "<source>",
) -> AnalysisResult:
    """Convenience wrapper returning an :class:`~src.api.AnalysisResult`."""

    bugs = find_named_tensor_bugs(source, input_specs, filename)
    return AnalysisResult(
        bugs=bugs,
        functions_analyzed=1,
        lines_analyzed=source.count("\n") + 1,
    )
