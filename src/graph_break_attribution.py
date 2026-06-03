"""Source-mapped attribution for Dynamo/export graph-break failures.

This module is intentionally side-effect free: it only inspects already-loaded
module classes or source strings and never executes the target model.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from src.verifiable_fragment import (
    UNSUPPORTED_CATEGORY_INFO,
    UnsupportedCategory,
    UnsupportedConstruct,
    analyze_source,
)


_IMPORT_PRELUDE = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
_CLASS_HEADER_RE = re.compile(r"^(\s*class\s+\w+\s*)\([^)]*\)(\s*:)", re.MULTILINE)


@dataclass(frozen=True)
class GraphBreakAttribution:
    """One likely source-site explanation for a graph capture failure."""

    backend: str
    category: str
    reason: str
    minimal_change: str
    confidence: str = "medium"
    line: Optional[int] = None
    col: Optional[int] = None
    snippet: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "category": self.category,
            "reason": self.reason,
            "minimal_change": self.minimal_change,
            "confidence": self.confidence,
            "line": self.line,
            "col": self.col,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class GraphBreakAttributionReport:
    """Structured attribution report attached to verifier failure metadata."""

    backend: str
    error_message: str
    attributions: Tuple[GraphBreakAttribution, ...] = ()
    source_available: bool = False
    fallback_used: Optional[str] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_attribution(self) -> bool:
        return bool(self.attributions)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "error_message": self.error_message,
            "source_available": self.source_available,
            "fallback_used": self.fallback_used,
            "notes": list(self.notes),
            "attributions": [a.to_dict() for a in self.attributions],
        }


def classify_graph_break_failure(
    model_or_source: Any,
    error_message: str,
    *,
    backend: str,
    fallback_used: Optional[str] = None,
) -> GraphBreakAttributionReport:
    """Map a Dynamo/export capture failure to verifiable-fragment categories.

    The classifier combines the canonical ``verifiable_fragment.analyze_source``
    taxonomy with backend-specific error-message evidence, then attaches a small
    minimal-change suggestion for each category.
    """

    try:
        source = _recover_source(model_or_source)
        if source is None:
            return _message_only_report(
                error_message,
                backend=backend,
                fallback_used=fallback_used,
                source_available=False,
                notes=("source unavailable; attribution is based on the backend error only",),
            )
        dedented = textwrap.dedent(source)
        static_attributions = _source_attributions(
            dedented,
            backend=backend,
        )
        message_category = _classify_backend_error(error_message)
        attributions = list(static_attributions)
        if not attributions or message_category != UnsupportedCategory.OTHER:
            attributions = _prioritize_message_category(
                attributions,
                message_category,
                error_message,
                backend,
            )
        notes: Tuple[str, ...] = ()
        if not attributions:
            notes = ("no source-level verifiable-fragment category matched",)
        return GraphBreakAttributionReport(
            backend=backend,
            error_message=str(error_message),
            source_available=True,
            fallback_used=fallback_used,
            attributions=tuple(attributions),
            notes=notes,
        )
    except Exception as exc:
        return GraphBreakAttributionReport(
            backend=backend,
            error_message=str(error_message),
            fallback_used=fallback_used,
            notes=(f"attribution failed safely: {type(exc).__name__}",),
        )


def _recover_source(model_or_source: Any) -> Optional[str]:
    if isinstance(model_or_source, str):
        return model_or_source
    if model_or_source is None:
        return None
    try:
        source = inspect.getsource(type(model_or_source))
    except (OSError, TypeError):
        return None
    try:
        import torch.nn as nn

        if isinstance(model_or_source, nn.Module):
            source = _CLASS_HEADER_RE.sub(r"\1(nn.Module)\2", source, count=1)
            return _IMPORT_PRELUDE + source
    except Exception:
        pass
    return source


def _source_attributions(source: str, *, backend: str) -> List[GraphBreakAttribution]:
    issues = analyze_source(source)
    locations = _locate_issue_nodes(source)
    attributions: List[GraphBreakAttribution] = []
    seen: set[Tuple[str, Optional[int], str]] = set()
    for issue in issues:
        line = _line_from_location(issue.location)
        node_line, node_col, snippet = locations.get(
            (issue.category, line),
            (line, None, _snippet(source, line)),
        )
        key = (issue.category.name, node_line, issue.description)
        if key in seen:
            continue
        seen.add(key)
        attributions.append(
            GraphBreakAttribution(
                backend=backend,
                category=issue.category.name,
                reason=_reason_for_issue(issue),
                minimal_change=_minimal_change(issue.category),
                confidence="high",
                line=node_line,
                col=node_col,
                snippet=snippet,
            )
        )
    return attributions


def _line_from_location(location: Optional[str]) -> Optional[int]:
    if not location:
        return None
    match = re.search(r"line\s+(\d+)", location)
    return int(match.group(1)) if match else None


def _snippet(source: str, line: Optional[int]) -> Optional[str]:
    if line is None:
        return None
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1].strip()
    return None


def _reason_for_issue(issue: UnsupportedConstruct) -> str:
    info = UNSUPPORTED_CATEGORY_INFO.get(issue.category, {})
    description = info.get("description", issue.description)
    return f"{issue.description}; {description}"


class _AttributionLocator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.locations: dict[
            Tuple[UnsupportedCategory, Optional[int]], Tuple[int, int, str]
        ] = {}
        self._source_lines: Sequence[str] = ()
        self._in_forward = False

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        if not self._source_lines:
            self._source_lines = getattr(node, "_source_lines", ())
        return super().visit(node)

    def visit_Module(self, node: ast.Module) -> Any:
        self._source_lines = getattr(node, "_source_lines", ())
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        old = self._in_forward
        if node.name == "forward":
            self._in_forward = True
        self.generic_visit(node)
        self._in_forward = old

    def visit_If(self, node: ast.If) -> Any:
        if self._in_forward and _expr_likely_tensor_dependent(node.test):
            self._record(UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW, node)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        if self._in_forward and _expr_likely_tensor_dependent(node.test):
            self._record(UnsupportedCategory.DATA_DEPENDENT_ITERATION, node)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        if self._in_forward and _range_has_runtime_call(node.iter):
            self._record(UnsupportedCategory.DATA_DEPENDENT_ITERATION, node)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> Any:
        if self._in_forward:
            self._record(UnsupportedCategory.DYNAMIC_ASSERTION, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if self._in_forward and _call_name(node) in {"item", "tolist", "numpy"}:
            self._record(UnsupportedCategory.TENSOR_TO_SCALAR, node)
        self.generic_visit(node)

    def _record(self, category: UnsupportedCategory, node: ast.AST) -> None:
        line = getattr(node, "lineno", None)
        if line is None:
            return
        if (category, line) in self.locations:
            return
        snippet = ""
        if 1 <= line <= len(self._source_lines):
            snippet = self._source_lines[line - 1].strip()
        self.locations[(category, line)] = (
            line,
            getattr(node, "col_offset", 0),
            snippet,
        )


def _call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _range_has_runtime_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and _call_name(node) == "range"
        and any(isinstance(arg, ast.Call) for arg in node.args)
    )


def _expr_likely_tensor_dependent(node: ast.AST) -> bool:
    if _is_safe_self_training(node) or _is_none_check(node):
        return False
    if isinstance(node, ast.Call):
        return True
    if isinstance(node, ast.Compare):
        return any(
            isinstance(child, ast.Call)
            or (isinstance(child, ast.Attribute) and child.attr in {"sum", "max", "min", "mean", "item", "numel"})
            for child in ast.walk(node)
        )
    return False


def _is_safe_self_training(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "training"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _is_none_check(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    return any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops) and any(
        isinstance(comp, ast.Constant) and comp.value is None
        for comp in node.comparators
    )


def _classify_backend_error(error_message: str) -> UnsupportedCategory:
    err = str(error_message).lower()
    if any(
        token in err
        for token in (
            "data-dependent expression",
            "guardondatadependentsymnode",
            "could not guard",
            "control flow",
            "conditional",
            "dynamic control flow",
            "symbolically traced variables cannot be used as inputs to control flow",
        )
    ):
        return UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW
    if any(
        token in err
        for token in (
            ".item",
            " item",
            "to a scalar",
            "python scalar",
            "symbolic int",
            "symbolic float",
            "get a value out of a symbolic",
            "constrain_as_size",
        )
    ):
        return UnsupportedCategory.TENSOR_TO_SCALAR
    if "assert" in err:
        return UnsupportedCategory.DYNAMIC_ASSERTION
    if "range" in err or "iteration" in err or "loop" in err:
        return UnsupportedCategory.DATA_DEPENDENT_ITERATION
    if "inplace" in err or "in-place" in err:
        return UnsupportedCategory.INPLACE_MUTATION
    if "autograd" in err or "custom function" in err:
        return UnsupportedCategory.CUSTOM_AUTOGRAD
    if "jit" in err or "script" in err:
        return UnsupportedCategory.JIT_SCRIPT
    if "unsupported" in err or "not defined" in err or "external" in err:
        return UnsupportedCategory.OPAQUE_EXTERNAL_CALL
    return UnsupportedCategory.OTHER


def _prioritize_message_category(
    attributions: List[GraphBreakAttribution],
    category: UnsupportedCategory,
    error_message: str,
    backend: str,
) -> List[GraphBreakAttribution]:
    if category == UnsupportedCategory.OTHER:
        return attributions
    matching = [a for a in attributions if a.category == category.name]
    rest = [a for a in attributions if a.category != category.name]
    if matching:
        return matching + rest
    return [
        GraphBreakAttribution(
            backend=backend,
            category=category.name,
            reason=(
                f"{backend} reported a {category.name.lower()}-like capture failure: "
                f"{str(error_message).splitlines()[0][:220]}"
            ),
            minimal_change=_minimal_change(category),
            confidence="medium",
        )
    ] + rest


def _message_only_report(
    error_message: str,
    *,
    backend: str,
    fallback_used: Optional[str],
    source_available: bool,
    notes: Iterable[str] = (),
) -> GraphBreakAttributionReport:
    category = _classify_backend_error(error_message)
    attributions: Tuple[GraphBreakAttribution, ...] = ()
    if category != UnsupportedCategory.OTHER:
        attributions = (
            GraphBreakAttribution(
                backend=backend,
                category=category.name,
                reason=(
                    f"{backend} error message matches {category.name}: "
                    f"{str(error_message).splitlines()[0][:220]}"
                ),
                minimal_change=_minimal_change(category),
                confidence="low" if not source_available else "medium",
            ),
        )
    return GraphBreakAttributionReport(
        backend=backend,
        error_message=str(error_message),
        source_available=source_available,
        fallback_used=fallback_used,
        attributions=attributions,
        notes=tuple(notes),
    )


def _minimal_change(category: UnsupportedCategory) -> str:
    suggestions = {
        UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW: (
            "Replace tensor-value Python branching with torch.cond/torch.where, "
            "or move the branch outside forward behind an explicit shape/static flag."
        ),
        UnsupportedCategory.DATA_DEPENDENT_ITERATION: (
            "Use a static loop bound, nn.ModuleList unrolling, or tensorized masking "
            "instead of deriving the trip count from a tensor value."
        ),
        UnsupportedCategory.DYNAMIC_ASSERTION: (
            "Convert the assert into a TensorGuard input-shape contract or a runtime "
            "precondition outside forward."
        ),
        UnsupportedCategory.TENSOR_TO_SCALAR: (
            "Keep the value as a tensor/symbolic shape expression; avoid .item(), "
            ".tolist(), int(), bool(), or Python scalar extraction inside forward."
        ),
        UnsupportedCategory.CUSTOM_AUTOGRAD: (
            "Wrap the custom autograd op in a reviewed TensorGuard stub or expose an "
            "equivalent torch.fx/export-traceable tensor implementation."
        ),
        UnsupportedCategory.INPLACE_MUTATION: (
            "Rewrite the mutation as an out-of-place tensor expression or confine it "
            "to a local intermediate that the tracer can represent."
        ),
        UnsupportedCategory.JIT_SCRIPT: (
            "Verify the eager nn.Module source before scripting, or provide a "
            "TensorGuard stub for the scripted boundary."
        ),
        UnsupportedCategory.OPAQUE_EXTERNAL_CALL: (
            "Inline the tensor-shape-affecting helper, register a community stub, or "
            "replace it with supported torch/nn/functional operations."
        ),
        UnsupportedCategory.DYNAMIC_MODULE_CONSTRUCTION: (
            "Construct submodules in __init__ and call the existing attributes from forward."
        ),
        UnsupportedCategory.UNSUPPORTED_BUILTIN: (
            "Replace the Python builtin with an equivalent supported torch operation "
            "or move the computation outside forward."
        ),
        UnsupportedCategory.OTHER: (
            "Isolate the failing statement, prefer torch.fx/export-traceable tensor "
            "ops, or add a focused TensorGuard stub for the opaque boundary."
        ),
    }
    return suggestions.get(category, suggestions[UnsupportedCategory.OTHER])


def _attach_source_lines_for_locator(source: str) -> ast.Module:
    tree = ast.parse(source)
    setattr(tree, "_source_lines", source.splitlines())
    return tree


def _locate_issue_nodes(
    source: str,
) -> dict[Tuple[UnsupportedCategory, Optional[int]], Tuple[int, int, str]]:
    try:
        tree = _attach_source_lines_for_locator(source)
    except SyntaxError:
        return {}
    locator = _AttributionLocator()
    locator.visit(tree)
    return locator.locations
