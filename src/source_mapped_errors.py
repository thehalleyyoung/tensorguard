"""
Source-mapped error diagnostics for TensorGuard verification output.

Translates raw SafetyViolation objects into human-readable, source-located
diagnostic messages suitable for terminal display, IDE integration (JSON),
or plain-text reports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Defensive imports from model_checker
# ---------------------------------------------------------------------------
try:
    from src.model_checker import SafetyViolation, ComputationGraph, ComputationStep
except ImportError:
    SafetyViolation = None  # type: ignore[assignment,misc]
    ComputationGraph = None  # type: ignore[assignment,misc]
    ComputationStep = None  # type: ignore[assignment,misc]

try:
    from src.model_checker import LayerKind, OpKind
except ImportError:
    LayerKind = None  # type: ignore[assignment,misc]
    OpKind = None  # type: ignore[assignment,misc]

try:
    from src.model_checker import Device
except ImportError:
    Device = None  # type: ignore[assignment,misc]


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RelatedLocation:
    """A secondary source location relevant to a diagnostic."""
    message: str
    line: int
    col: int
    snippet: str = ""


@dataclass
class SourceMappedDiagnostic:
    """A user-friendly diagnostic with source-location information."""
    message: str
    severity: str  # "error" | "warning" | "info"
    source_line: int
    source_col: int
    source_snippet: str
    related_locations: List[RelatedLocation] = field(default_factory=list)
    fix_suggestion: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Source helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _source_lines(source: str) -> List[str]:
    """Split source into lines (1-indexed access via lines[lineno - 1])."""
    return source.splitlines()


def _get_snippet(source: str, line: int) -> str:
    """Return the source line at *line* (1-indexed), stripped, or ''."""
    lines = _source_lines(source)
    if 1 <= line <= len(lines):
        return lines[line - 1].rstrip()
    return ""


def _find_layer_def_line(source: str, layer_name: str) -> int:
    """Heuristically find where a layer attribute is defined in __init__."""
    pattern = re.compile(
        r"self\." + re.escape(layer_name) + r"\s*=",
    )
    for idx, line in enumerate(_source_lines(source), start=1):
        if pattern.search(line):
            return idx
    return 0


def _find_input_param_line(source: str, param_name: str) -> int:
    """Find the forward() def line or the line where *param_name* first appears."""
    for idx, line in enumerate(_source_lines(source), start=1):
        if re.search(r"def\s+forward\b", line) and param_name in line:
            return idx
    # Fallback: first usage of the name
    for idx, line in enumerate(_source_lines(source), start=1):
        if param_name in line:
            return idx
    return 0


def _step_line(step: Any) -> int:
    """Extract line from a ComputationStep (defensive)."""
    return getattr(step, "line", 0) or 0


def _step_col(step: Any) -> int:
    """Extract col from a ComputationStep (defensive)."""
    return getattr(step, "col", 0) or 0


def _layer_ref(step: Any) -> Optional[str]:
    return getattr(step, "layer_ref", None)


def _op_name(step: Any) -> str:
    op = getattr(step, "op", None)
    if op is None:
        return "unknown"
    return getattr(op, "name", str(op))


def _shape_str(shape: Any) -> str:
    """Pretty-print a TensorShape or tuple."""
    if shape is None:
        return "unknown"
    dims = getattr(shape, "dims", None)
    if dims is not None:
        parts = []
        for d in dims:
            sym = getattr(d, "symbol", None)
            val = getattr(d, "value", None)
            if sym:
                parts.append(str(sym))
            elif val is not None:
                parts.append(str(val))
            else:
                parts.append(str(d))
        return "(" + ", ".join(parts) + ")"
    if isinstance(shape, (list, tuple)):
        return "(" + ", ".join(str(d) for d in shape) + ")"
    return str(shape)


def _device_str(device: Any) -> str:
    if device is None:
        return "unknown"
    val = getattr(device, "value", None)
    if val is not None:
        return str(val)
    return str(device)


# ═══════════════════════════════════════════════════════════════════════════════
# Message generators (per violation kind)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_shape_message(v: Any, source: str, graph: Any) -> Tuple[str, List[RelatedLocation], Optional[str]]:
    """Build a plain-language message for a shape_incompatible violation."""
    step = v.step
    line = _step_line(step)
    layer = _layer_ref(step)
    shape_a = _shape_str(getattr(v, "shape_a", None))
    shape_b = _shape_str(getattr(v, "shape_b", None))
    tensor_a = getattr(v, "tensor_a", None) or "input"
    tensor_b = getattr(v, "tensor_b", None) or ""

    related: List[RelatedLocation] = []
    fix: Optional[str] = None

    if layer:
        # Try to get layer def info
        layer_line = _find_layer_def_line(source, layer)
        layer_def = None
        if graph is not None:
            layers = getattr(graph, "layers", {})
            layer_def = layers.get(layer)

        in_feat = getattr(layer_def, "in_features", None) if layer_def else None
        out_feat = getattr(layer_def, "out_features", None) if layer_def else None
        in_chan = getattr(layer_def, "in_channels", None) if layer_def else None

        expected = in_feat or in_chan
        if expected and shape_a != "unknown":
            msg = (
                f"Layer {layer} (line {line}) expects input dimension {expected}, "
                f"but receives {shape_a} from {tensor_a}"
            )
            if layer_line:
                related.append(RelatedLocation(
                    message=f"Layer {layer} defined here with in_features={expected}",
                    line=layer_line,
                    col=0,
                    snippet=_get_snippet(source, layer_line),
                ))
            fix = f"Ensure the preceding layer outputs dimension {expected}, or change {layer} to accept {shape_a}"
        else:
            msg = (
                f"Shape mismatch at layer {layer} (line {line}): "
                f"input has shape {shape_a}, expected compatible shape"
            )
            if layer_line:
                related.append(RelatedLocation(
                    message=f"Layer {layer} defined here",
                    line=layer_line,
                    col=0,
                    snippet=_get_snippet(source, layer_line),
                ))
    else:
        op = _op_name(step)
        msg = (
            f"Shape mismatch in {op} operation (line {line}): "
            f"tensor {tensor_a} has shape {shape_a}"
        )
        if tensor_b and shape_b != "unknown":
            msg += f", tensor {tensor_b} has shape {shape_b}"

    return msg, related, fix


def _make_broadcast_message(v: Any, source: str, graph: Any) -> Tuple[str, List[RelatedLocation], Optional[str]]:
    """Build a plain-language message for a broadcast_incompatible violation."""
    step = v.step
    line = _step_line(step)
    shape_a = _shape_str(getattr(v, "shape_a", None))
    shape_b = _shape_str(getattr(v, "shape_b", None))
    tensor_a = getattr(v, "tensor_a", None) or "a"
    tensor_b = getattr(v, "tensor_b", None) or "b"

    related: List[RelatedLocation] = []

    # Try to identify the mismatching dimensions
    dims_detail = ""
    sa = getattr(v, "shape_a", None)
    sb = getattr(v, "shape_b", None)
    if sa is not None and sb is not None:
        da = getattr(sa, "dims", None) or (sa if isinstance(sa, (list, tuple)) else None)
        db = getattr(sb, "dims", None) or (sb if isinstance(sb, (list, tuple)) else None)
        if da and db:
            for i, (a_d, b_d) in enumerate(zip(reversed(list(da)), reversed(list(db)))):
                a_v = getattr(a_d, "value", a_d)
                b_v = getattr(b_d, "value", b_d)
                if a_v is not None and b_v is not None and a_v != b_v and a_v != 1 and b_v != 1:
                    dims_detail = f" — dimensions {a_v} and {b_v} are neither equal nor 1"
                    break

    msg = (
        f"Broadcast incompatibility: tensor {tensor_a} has shape {shape_a} at line {line}, "
        f"tensor {tensor_b} has shape {shape_b}"
        f"{dims_detail}"
    )

    fix = "Reshape one of the tensors so that corresponding dimensions are either equal or 1"
    return msg, related, fix


def _make_device_message(v: Any, source: str, graph: Any) -> Tuple[str, List[RelatedLocation], Optional[str]]:
    """Build a plain-language message for a device_mismatch violation."""
    step = v.step
    line = _step_line(step)
    layer = _layer_ref(step)
    device_a = _device_str(getattr(v, "device_a", None))
    device_b = _device_str(getattr(v, "device_b", None))
    tensor_a = getattr(v, "tensor_a", None) or "tensor"
    tensor_b = getattr(v, "tensor_b", None) or ""

    related: List[RelatedLocation] = []

    if layer:
        layer_line = _find_layer_def_line(source, layer)
        if tensor_b:
            msg = (
                f"Device mismatch: layer {layer} (line {line}) is on {device_a}, "
                f"but input tensor {tensor_b} is on {device_b}"
            )
        else:
            msg = (
                f"Device mismatch: layer {layer} (line {line}) is on {device_a}, "
                f"but input tensor {tensor_a} is on {device_b}"
            )
        if layer_line:
            related.append(RelatedLocation(
                message=f"Layer {layer} defined here",
                line=layer_line,
                col=0,
                snippet=_get_snippet(source, layer_line),
            ))
        fix = f"Move tensors to the same device using .to('{device_a}') or .to('{device_b}')"
    else:
        msg = (
            f"Device mismatch at line {line}: "
            f"tensor {tensor_a} is on {device_a}, tensor {tensor_b} is on {device_b}"
        )
        fix = f"Move tensors to the same device using .to('{device_a}')"

    return msg, related, fix


def _make_gradient_message(v: Any, source: str, graph: Any) -> Tuple[str, List[RelatedLocation], Optional[str]]:
    """Build a plain-language message for a gradient_invalid violation."""
    step = v.step
    line = _step_line(step)
    tensor_a = getattr(v, "tensor_a", None) or "tensor"

    msg = f"Gradient error at line {line}: {v.message}"
    fix = "Check requires_grad settings and detach() calls"
    return msg, [], fix


def _make_generic_message(v: Any, source: str, graph: Any) -> Tuple[str, List[RelatedLocation], Optional[str]]:
    """Fallback for unrecognised violation kinds."""
    step = v.step
    line = _step_line(step)
    msg = f"Verification error at line {line}: {v.message}"
    return msg, [], None


_MESSAGE_BUILDERS = {
    "shape_incompatible": _make_shape_message,
    "broadcast_incompatible": _make_broadcast_message,
    "device_mismatch": _make_device_message,
    "gradient_invalid": _make_gradient_message,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Severity mapping
# ═══════════════════════════════════════════════════════════════════════════════

def _severity_for(v: Any) -> str:
    """Map a violation to a severity level."""
    confidence = getattr(v, "confidence", None)
    fp_cat = getattr(v, "fp_category", None)
    if fp_cat:
        return "warning"
    if confidence is not None:
        conf_val = getattr(confidence, "value", str(confidence))
        if conf_val == "low":
            return "info"
        if conf_val == "medium":
            return "warning"
    return "error"


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def map_violations_to_diagnostics(
    source: str,
    violations: list,
    graph: Any = None,
) -> List[SourceMappedDiagnostic]:
    """Convert a list of SafetyViolation objects into SourceMappedDiagnostic.

    Parameters
    ----------
    source : str
        The original Python source code of the nn.Module.
    violations : list
        List of SafetyViolation instances from the verifier.
    graph : ComputationGraph, optional
        The computation graph extracted from *source*.  Used to resolve
        layer definitions for richer messages.

    Returns
    -------
    List[SourceMappedDiagnostic]
    """
    diagnostics: List[SourceMappedDiagnostic] = []
    for v in violations:
        kind = getattr(v, "kind", "")
        builder = _MESSAGE_BUILDERS.get(kind, _make_generic_message)
        msg, related, fix = builder(v, source, graph)

        step = getattr(v, "step", None)
        line = _step_line(step) if step else 0
        col = _step_col(step) if step else 0
        snippet = _get_snippet(source, line)
        severity = _severity_for(v)

        diagnostics.append(SourceMappedDiagnostic(
            message=msg,
            severity=severity,
            source_line=line,
            source_col=col,
            source_snippet=snippet,
            related_locations=related,
            fix_suggestion=fix,
        ))
    return diagnostics


# ═══════════════════════════════════════════════════════════════════════════════
# Formatters
# ═══════════════════════════════════════════════════════════════════════════════

def _caret_line(snippet: str, col: int, width: int = 1) -> str:
    """Build a caret underline that points at *col* (0-indexed) in *snippet*.

    The snippet keeps its original leading indentation, so aligning ``col``
    spaces under it lands the caret on the offending token.  ``width`` carets are
    drawn (at least one).  Tabs in the indentation are preserved so the caret
    stays aligned in terminals that render tabs identically.
    """
    if col < 0:
        col = 0
    # Preserve tabs from the snippet's leading whitespace for alignment.
    prefix_chars = []
    for i in range(min(col, len(snippet))):
        prefix_chars.append("\t" if snippet[i] == "\t" else " ")
    prefix = "".join(prefix_chars) + " " * max(0, col - len(prefix_chars))
    return prefix + "^" * max(1, width)


def format_plain(diagnostics: List[SourceMappedDiagnostic]) -> str:
    """Format diagnostics as plain text."""
    parts: List[str] = []
    for d in diagnostics:
        header = f"{d.severity.upper()}: {d.message}"
        loc = f"  --> line {d.source_line}, col {d.source_col}"
        lines = [header, loc]
        if d.source_snippet:
            lines.append(f"  | {d.source_snippet}")
            if d.source_col and d.source_col > 0:
                lines.append(f"  | {_caret_line(d.source_snippet, d.source_col)}")
        for rel in d.related_locations:
            lines.append(f"  note: {rel.message} (line {rel.line})")
            if rel.snippet:
                lines.append(f"  | {rel.snippet}")
        if d.fix_suggestion:
            lines.append(f"  fix: {d.fix_suggestion}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ANSI colour codes
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "dim": "\033[2m",
    "blue": "\033[34m",
}

_SEVERITY_COLOUR = {
    "error": _ANSI["red"],
    "warning": _ANSI["yellow"],
    "info": _ANSI["cyan"],
}


def format_ansi(diagnostics: List[SourceMappedDiagnostic]) -> str:
    """Format diagnostics with ANSI colour codes for terminal display."""
    parts: List[str] = []
    for d in diagnostics:
        colour = _SEVERITY_COLOUR.get(d.severity, "")
        r = _ANSI["reset"]
        b = _ANSI["bold"]
        dim = _ANSI["dim"]
        green = _ANSI["green"]
        blue = _ANSI["blue"]

        header = f"{b}{colour}{d.severity.upper()}{r}: {b}{d.message}{r}"
        loc = f"  {blue}-->{r} line {d.source_line}, col {d.source_col}"
        lines = [header, loc]
        if d.source_snippet:
            lines.append(f"  {dim}|{r} {d.source_snippet}")
            if d.source_col and d.source_col > 0:
                caret = _caret_line(d.source_snippet, d.source_col)
                lines.append(f"  {dim}|{r} {colour}{b}{caret}{r}")
        for rel in d.related_locations:
            lines.append(f"  {green}note{r}: {rel.message} (line {rel.line})")
            if rel.snippet:
                lines.append(f"  {dim}|{r} {rel.snippet}")
        if d.fix_suggestion:
            lines.append(f"  {green}fix{r}: {d.fix_suggestion}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def format_json(diagnostics: List[SourceMappedDiagnostic]) -> str:
    """Format diagnostics as a JSON array (for IDE integration)."""
    items = []
    for d in diagnostics:
        items.append({
            "message": d.message,
            "severity": d.severity,
            "source_line": d.source_line,
            "source_col": d.source_col,
            "source_snippet": d.source_snippet,
            "related_locations": [
                {
                    "message": r.message,
                    "line": r.line,
                    "col": r.col,
                    "snippet": r.snippet,
                }
                for r in d.related_locations
            ],
            "fix_suggestion": d.fix_suggestion,
        })
    return json.dumps(items, indent=2)
