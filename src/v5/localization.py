"""
src/v5/localization.py
======================

Localization tracing for TensorGuard refutations.

Exposes:
  localize(source, bug_message, counterexample) -> int | None
      Returns the 1-based source line of the FIRST unsatisfied symbolic
      constraint:
        1. If counterexample.violations[*].line is populated (>0), use it.
        2. Otherwise fall back to AST-based attribution.

  enrich_result(result, source)
      Mutates result.bugs[*].location.line to the localized line when the
      existing line is 0 or suspiciously large (> len(source.splitlines())).

  _extract_offending_vars(message) -> dict
      Helper that pulls tensor variable names and dim names out of a TG
      bug message.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Bracket-enclosed shapes:  '[B, 64]'  or  '(2048, 384)'
_SHAPE_RE = re.compile(r"[\[\(]([^\[\]\(\)]+)[\]\)]")

# Quoted shape strings: "'[B, 64]'" or "'(2048, 384)'"
_QUOTED_SHAPE_RE = re.compile(r"'[\[\(]([^\[\]\(\)']+)[\]\)]'")

# Identifiers that look like tensor variable names (lower-case, short)
_VAR_RE = re.compile(r"\b([a-z_][a-zA-Z0-9_]{0,15})\b")

# Known operation keywords that appear in TG messages
_OP_KEYWORDS = {
    "view": [r"\.view\s*\(", r"\.view\("],
    "reshape": [r"\.reshape\s*\(", r"\.reshape\("],
    "conv2d": [r"Conv2d\s*\(", r"nn\.Conv2d\s*\(", r"F\.conv2d\s*\("],
    "linear": [r"Linear\s*\(", r"nn\.Linear\s*\(", r"F\.linear\s*\("],
    "batchnorm": [r"BatchNorm\w*\s*\(", r"nn\.BatchNorm\w*\s*\("],
    "broadcast": [r"\+|\-|\*|/"],
    "matmul": [r"@\s", r"torch\.matmul\s*\(", r"torch\.mm\s*\("],
    "cat": [r"torch\.cat\s*\("],
    "stack": [r"torch\.stack\s*\("],
    "scaled_dot_product_attention": [r"scaled_dot_product_attention\s*\("],
}

# TG message prefix tags → operation hint
_TAG_TO_OPS = {
    "SHAPE-INCOMPATIBLE": ["view", "reshape", "conv2d", "linear", "batchnorm",
                           "matmul", "cat", "broadcast"],
    "VIEW": ["view", "reshape"],
    "RESHAPE": ["view", "reshape"],
    "DEVICE-MISMATCH": [],
    "MODEL-CHECK": ["view", "reshape", "conv2d", "linear", "batchnorm",
                    "matmul", "cat"],
}

# Dim-name tokens we should NOT treat as variable names
_NOISE_TOKENS: Set[str] = {
    "is", "of", "to", "in", "at", "for", "and", "or", "not",
    "the", "size", "shape", "input", "output", "dims", "dim",
    "cannot", "must", "match", "invalid", "tensor", "tensors",
    "expected", "got", "but", "does", "with", "from", "into",
    "reshape", "view", "broadcast", "incompatible", "mismatch",
    "warning", "error", "true", "false", "none", "self", "cls",
    "kind", "step", "line", "col",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _extract_offending_vars(message: str) -> Dict[str, List[str]]:
    """Extract tensor variable names and dim names from a TG bug message.

    Returns:
        {
          "vars":  list of likely tensor variable names,
          "dims":  list of symbolic dim tokens (e.g. "B", "batch_size"),
          "shapes": list of raw shape strings (e.g. "2048, 384"),
        }
    """
    shapes: List[str] = []
    for m in _QUOTED_SHAPE_RE.finditer(message):
        shapes.append(m.group(1))
    for m in _SHAPE_RE.finditer(message):
        shapes.append(m.group(1))

    # Dim tokens: alphanumeric tokens from shapes that are symbolic (non-numeric)
    dims: List[str] = []
    seen_dims: Set[str] = set()
    for s in shapes:
        for tok in re.split(r"[,\s*]+", s):
            tok = tok.strip("*B ")
            if tok and not tok.isdigit() and tok not in seen_dims:
                seen_dims.add(tok)
                dims.append(tok)

    # Variable-name candidates: identifiers from message body NOT in noise set
    # Look specifically after "cannot reshape", "view", "for input" etc.
    var_candidates: List[str] = []
    seen_vars: Set[str] = set()

    # Heuristic: grab identifiers appearing before ".view" or ".reshape"
    for m in re.finditer(r"\b([a-z_][a-zA-Z0-9_]{0,20})\s*\.\s*(view|reshape)\b",
                         message):
        v = m.group(1)
        if v not in _NOISE_TOKENS and v not in seen_vars:
            seen_vars.add(v)
            var_candidates.append(v)

    # Also look for "tensor 'x'" or "variable 'x'" patterns
    for m in re.finditer(r"(?:tensor|variable|param)\s+'([a-zA-Z_][a-zA-Z0-9_]*)'",
                         message):
        v = m.group(1)
        if v not in _NOISE_TOKENS and v not in seen_vars:
            seen_vars.add(v)
            var_candidates.append(v)

    # Generic identifier sweep from message
    for m in _VAR_RE.finditer(message):
        v = m.group(1)
        if (v not in _NOISE_TOKENS and v not in seen_vars
                and not v[0].isupper()):
            seen_vars.add(v)
            var_candidates.append(v)

    return {"vars": var_candidates, "dims": dims, "shapes": shapes}


def _op_hints_from_message(message: str) -> List[str]:
    """Return a priority-ordered list of operation kinds suggested by the message."""
    # Tag-based
    tag_m = re.match(r"\[([A-Z_-]+)\]", message)
    if tag_m:
        tag = tag_m.group(1)
        if tag in _TAG_TO_OPS:
            hints = list(_TAG_TO_OPS[tag])
        else:
            hints = list(_TAG_TO_OPS.get("SHAPE-INCOMPATIBLE", []))
    else:
        hints = list(_TAG_TO_OPS.get("SHAPE-INCOMPATIBLE", []))

    # Refine based on keywords in the message body
    msg_lower = message.lower()
    if "view" in msg_lower or "reshape" in msg_lower:
        # Move view/reshape to front
        for op in ("reshape", "view"):
            if op in hints:
                hints.remove(op)
                hints.insert(0, op)
    if "conv" in msg_lower:
        if "conv2d" in hints:
            hints.remove("conv2d")
            hints.insert(0, "conv2d")
    if "linear" in msg_lower or "mat1" in msg_lower or "mat2" in msg_lower:
        if "linear" in hints:
            hints.remove("linear")
            hints.insert(0, "linear")
    if "broadcast" in msg_lower or "must match the size" in msg_lower:
        if "broadcast" in hints:
            hints.remove("broadcast")
            hints.insert(0, "broadcast")
    if "batchnorm" in msg_lower or "running_mean" in msg_lower:
        if "batchnorm" in hints:
            hints.remove("batchnorm")
            hints.insert(0, "batchnorm")
    return hints


# ---------------------------------------------------------------------------
# AST visitor for localization
# ---------------------------------------------------------------------------

class _LocVisitor(ast.NodeVisitor):
    """Collect candidate lines for shape-violating operations."""

    def __init__(self, op_hints: List[str], offending_vars: List[str]) -> None:
        self.op_hints = op_hints
        self.offending_vars: Set[str] = set(offending_vars)
        self.candidates: List[Tuple[int, str]] = []  # (line, op_kind)

    def _record(self, node: ast.AST, kind: str) -> None:
        line = getattr(node, "lineno", None)
        if line is not None:
            self.candidates.append((line, kind))

    # --- attribute calls: x.view(...), x.reshape(...) ---
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        fn = node.func

        # Method calls: <expr>.method(...)
        if isinstance(fn, ast.Attribute):
            method = fn.attr.lower()
            if method in ("view", "reshape"):
                self._record(node, method)
            # Check for Conv2d / Linear on self.xxx / local var
            elif method in ("conv2d",):
                self._record(node, "conv2d")

        # Direct calls: nn.Conv2d(...), nn.Linear(...), F.conv2d(...), etc.
        if isinstance(fn, ast.Attribute):
            name = fn.attr
            if re.match(r"Conv\d*d?", name):
                self._record(node, "conv2d")
            elif name == "Linear":
                self._record(node, "linear")
            elif re.match(r"BatchNorm", name):
                self._record(node, "batchnorm")
            elif name == "scaled_dot_product_attention":
                self._record(node, "scaled_dot_product_attention")
            elif name in ("cat", "stack"):
                self._record(node, name)
            elif name in ("matmul", "mm", "bmm", "einsum"):
                self._record(node, "matmul")
            elif name == "MultiheadAttention":
                self._record(node, "multihead_attention")
            elif name in ("CrossEntropyLoss", "cross_entropy"):
                self._record(node, "cross_entropy")
            elif name in ("LSTM", "GRU", "RNN"):
                self._record(node, "rnn")
            elif name in ("Embedding",):
                self._record(node, "embedding")
            elif name in ("LayerNorm", "GroupNorm", "InstanceNorm1d",
                          "InstanceNorm2d"):
                self._record(node, "norm")
            elif name in ("permute", "transpose"):
                self._record(node, "permute")
            elif name in ("isclose",):
                self._record(node, "broadcast")
        elif isinstance(fn, ast.Name):
            name = fn.id
            if re.match(r"Conv\d*d?", name):
                self._record(node, "conv2d")
            elif name == "Linear":
                self._record(node, "linear")
            elif re.match(r"BatchNorm", name):
                self._record(node, "batchnorm")
            elif name == "MultiheadAttention":
                self._record(node, "multihead_attention")
            elif name in ("LSTM", "GRU", "RNN"):
                self._record(node, "rnn")

        self.generic_visit(node)

    # --- binary ops: +, *, etc. (broadcast) ---
    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                ast.MatMult)):
            if isinstance(node.op, ast.MatMult):
                self._record(node, "matmul")
            else:
                self._record(node, "broadcast")
        self.generic_visit(node)

    # --- augmented assign: x *= scale ---
    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self._record(node, "broadcast")
        self.generic_visit(node)


def _ast_localize(source: str, message: str) -> Optional[int]:
    """Walk the source AST and return the best candidate line."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _fallback_grep(source, message)

    op_hints = _op_hints_from_message(message)
    info = _extract_offending_vars(message)
    visitor = _LocVisitor(op_hints, info["vars"])
    visitor.visit(tree)

    if not visitor.candidates:
        return _fallback_grep(source, message)

    # Score candidates: prefer op_hints earlier in list, prefer lower line
    def _score(c: Tuple[int, str]) -> Tuple[int, int]:
        line, kind = c
        try:
            rank = op_hints.index(kind)
        except ValueError:
            rank = len(op_hints)
        return (rank, line)

    visitor.candidates.sort(key=_score)
    return visitor.candidates[0][0]


def _fallback_grep(source: str, message: str) -> Optional[int]:
    """Last-resort: scan lines for operation keywords mentioned in message."""
    lines = source.splitlines()
    msg_lower = message.lower()

    # Determine what pattern to search for
    patterns: List[str] = []
    if "view" in msg_lower or "reshape" in msg_lower:
        patterns = [r"\.view\s*\(", r"\.reshape\s*\("]
    elif "conv" in msg_lower:
        patterns = [r"Conv\d*d?\s*\(", r"conv2d\s*\("]
    elif "linear" in msg_lower or "mat1" in msg_lower:
        patterns = [r"Linear\s*\(", r"linear\s*\("]
    elif "batchnorm" in msg_lower or "running_mean" in msg_lower:
        patterns = [r"BatchNorm\w*\s*\("]
    elif "multihead" in msg_lower or "embed_dim" in msg_lower or "num_heads" in msg_lower:
        patterns = [r"MultiheadAttention\s*\("]
    elif "einsum" in msg_lower or "subscript" in msg_lower:
        patterns = [r"einsum\s*\("]
    elif "broadcast" in msg_lower or "must match" in msg_lower:
        patterns = [r"\+", r"\*", r"\*="]
    elif "cross_entropy" in msg_lower or "weight tensor" in msg_lower:
        patterns = [r"CrossEntropyLoss\s*\(", r"cross_entropy\s*\("]
    elif "scaled_dot_product" in msg_lower or "gqa" in msg_lower:
        patterns = [r"scaled_dot_product_attention\s*\("]
    else:
        patterns = [r"\.view\s*\(", r"\.reshape\s*\(", r"Conv\d*d?\s*\(",
                    r"Linear\s*\(", r"\+", r"@", r"einsum\s*\(",
                    r"MultiheadAttention\s*\("]

    for lineno, line in enumerate(lines, 1):
        for pat in patterns:
            if re.search(pat, line):
                return lineno
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def localize(
    source: str,
    bug_message: str,
    counterexample: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Return the 1-based source line of the first unsatisfied constraint.

    Strategy:
      1. If counterexample has violations with a populated line field, use
         the minimum non-zero line across all violations.
      2. Fall back to AST-based attribution.

    Args:
        source:         Full Python source code passed to verify_architecture.
        bug_message:    The bug.message string from the result (e.g.
                        "[SHAPE-INCOMPATIBLE] Reshape incompatible: ...").
        counterexample: result.counterexample dict (may be None).

    Returns:
        1-based line number or None if unable to localize.
    """
    if not bug_message:
        return None

    # Strategy 1: use counterexample's populated line
    if counterexample:
        ce_lines: List[int] = []
        for v in counterexample.get("violations", []):
            # The line may appear directly (from our serialization) or
            # nested under "step"
            line = v.get("line", 0)
            if not line:
                step = v.get("step")
                if isinstance(step, dict):
                    line = step.get("line", 0)
            if line and line > 0:
                ce_lines.append(line)
        if ce_lines:
            return min(ce_lines)

    # Strategy 2: AST-based attribution
    return _ast_localize(source, bug_message)


def enrich_result(result: Any, source: str) -> None:
    """Mutate result.bugs[*].location.line to the localized line.

    Only updates a bug's line when:
      - The existing line is 0 (unknown), or
      - The existing line exceeds the number of lines in the source
        (stale / out-of-range attribution).

    Args:
        result: An AnalysisResult (or any object with a .bugs list of Bug
                objects, each having .message, .location.line, and a
                .counterexample dict on result).
        source: The Python source string that was passed to verify_architecture.
    """
    n_lines = len(source.splitlines())
    ce = getattr(result, "counterexample", None)

    for bug in getattr(result, "bugs", []):
        current_line = getattr(getattr(bug, "location", None), "line", 0) or 0
        if current_line > 0 and current_line <= n_lines:
            continue  # already reasonable
        new_line = localize(source, bug.message, ce)
        if new_line is not None:
            bug.location.line = new_line
