"""Mechanical autofix suggestions for a small, *sound* set of bug shapes.

Step 59.  Some shape bugs have a single, unambiguous mechanical repair: a
``nn.Linear`` whose ``in_features`` does not match the dimension actually
flowing into it, or a ``nn.Conv*d`` whose ``in_channels`` does not match the
channel dimension it receives.  In those cases the verifier already knows the
*concrete* dimension the layer is fed (it is recorded on the counterexample's
``SafetyViolation.shape_a``), so the fix is simply to set the constructor
argument to that value.

This module turns such violations into concrete, line-level edit suggestions
(:class:`AutoFix`).  Suggestions are *only* emitted when the rewrite is
unambiguous — the layer is defined on a single source line, the offending
dimension is concrete (not symbolic), and the old value appears exactly where
expected in the constructor call.  Everything is defensive: any uncertainty
means no suggestion is produced rather than a wrong one.

The CLI surfaces these via ``tensorguard verify --fix`` (print a diff) and can
optionally apply them in place with ``--fix --write``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional

from src.source_mapped_errors import _find_layer_def_line, _source_lines


# LayerKind.name -> (constructor token(s), keyword-arg name, which input dim
# carries the offending size).  "last" = final dim (Linear feature dim);
# "channels" = index 1 (Conv channel dim).
_FIXABLE_LAYERS = {
    "LINEAR": (("Linear",), "in_features", "last"),
    "CONV1D": (("Conv1d",), "in_channels", "channels"),
    "CONV2D": (("Conv2d",), "in_channels", "channels"),
    "CONV3D": (("Conv3d",), "in_channels", "channels"),
}


@dataclass
class AutoFix:
    """A concrete, mechanical, single-line edit suggestion."""

    layer: str
    kind: str  # e.g. "linear_in_features", "conv_in_channels"
    line: int  # 1-indexed source line of the layer definition (0 if unknown)
    original: str  # original source line text (rstripped)
    suggested: str  # rewritten source line text
    description: str
    old_value: int
    new_value: int


def _as_int(dim: Any) -> Optional[int]:
    """Return the concrete integer value of a ShapeDim/int, else None."""
    if dim is None:
        return None
    if isinstance(dim, int):
        return dim
    # ShapeDim: concrete iff not symbolic.
    is_sym = getattr(dim, "is_symbolic", None)
    val = getattr(dim, "value", None)
    if is_sym is False and isinstance(val, int):
        return val
    if is_sym is None and isinstance(val, int):
        return val
    return None


def _dims(shape: Any) -> List[Any]:
    dims = getattr(shape, "dims", None)
    if dims is None:
        return []
    return list(dims)


def _rewrite_arg(
    line_text: str, ctor_tokens: tuple, old: int, new: int, kw_name: str
) -> Optional[str]:
    """Rewrite a single constructor argument from *old* to *new*.

    Looks for one of *ctor_tokens* followed by ``(``, restricts the rewrite to
    that call's argument list (up to the first matching ``)`` on the same
    line), and prefers ``kw_name=old`` over the first bare integer literal equal
    to *old*.  Returns the rewritten line, or None if no unambiguous edit was
    possible (the caller then emits no suggestion).
    """
    for tok in ctor_tokens:
        m = re.search(r"\b" + re.escape(tok) + r"\s*\(", line_text)
        if not m:
            continue
        args_start = m.end()
        close = line_text.find(")", args_start)
        if close == -1:
            # Multi-line constructor; refuse (cannot scope the edit safely).
            return None
        head = line_text[:args_start]
        args = line_text[args_start:close]
        tail = line_text[close:]

        # Prefer the keyword form: in_features=OLD / in_channels=OLD.
        kw_pat = re.compile(
            r"(\b" + re.escape(kw_name) + r"\s*=\s*)" + str(old) + r"\b"
        )
        if kw_pat.search(args):
            new_args = kw_pat.sub(r"\g<1>" + str(new), args, count=1)
            return head + new_args + tail

        # Otherwise replace the first bare integer literal equal to OLD.
        int_pat = re.compile(r"(?<![\w.])" + str(old) + r"(?![\w.])")
        if int_pat.search(args):
            new_args = int_pat.sub(str(new), args, count=1)
            return head + new_args + tail
        return None
    return None


def build_autofixes(
    source: str, violations: List[Any], graph: Any
) -> List[AutoFix]:
    """Produce mechanical :class:`AutoFix` suggestions for *violations*.

    Only ``shape_incompatible`` violations at a fixable layer (Linear / Conv*d)
    whose offending input dimension is concrete and differs from the declared
    constructor argument yield a suggestion.  De-duplicated by layer.
    """
    fixes: List[AutoFix] = []
    seen = set()
    layers = getattr(graph, "layers", {}) or {}

    for v in violations or []:
        try:
            if str(getattr(v, "kind", "")).lower() != "shape_incompatible":
                continue
            step = getattr(v, "step", None)
            layer = getattr(step, "layer_ref", None) if step else None
            if not layer or layer in seen:
                continue
            layer_def = layers.get(layer)
            if layer_def is None:
                continue
            kind_obj = getattr(layer_def, "kind", None)
            kind_name = getattr(kind_obj, "name", "") or ""
            spec = _FIXABLE_LAYERS.get(kind_name)
            if spec is None:
                continue
            ctor_tokens, kw_name, which = spec

            # Declared (wrong) value.
            if kw_name == "in_features":
                declared = getattr(layer_def, "in_features", None)
                fix_kind = "linear_in_features"
            else:
                declared = getattr(layer_def, "in_channels", None)
                fix_kind = "conv_in_channels"
            if not isinstance(declared, int):
                continue

            # Actual incoming dimension the verifier observed.
            dims = _dims(getattr(v, "shape_a", None))
            if not dims:
                continue
            if which == "last":
                actual = _as_int(dims[-1])
            else:  # channels
                actual = _as_int(dims[1]) if len(dims) > 1 else None
            if actual is None or actual == declared:
                continue

            def_line = _find_layer_def_line(source, layer)
            if not def_line:
                continue
            lines = _source_lines(source)
            if not (1 <= def_line <= len(lines)):
                continue
            original = lines[def_line - 1].rstrip()
            suggested = _rewrite_arg(
                original, ctor_tokens, declared, actual, kw_name
            )
            if suggested is None or suggested == original:
                continue

            arg = "in_features" if which == "last" else "in_channels"
            fixes.append(
                AutoFix(
                    layer=layer,
                    kind=fix_kind,
                    line=def_line,
                    original=original,
                    suggested=suggested,
                    description=(
                        f"Layer {layer} declares {arg}={declared} but is fed a "
                        f"tensor whose matching dimension is {actual}; set "
                        f"{arg}={actual}."
                    ),
                    old_value=declared,
                    new_value=actual,
                )
            )
            seen.add(layer)
        except Exception:
            # Autofixes are advisory; never let one bad violation break the set.
            continue
    return fixes


def apply_autofixes(source: str, fixes: List[AutoFix]) -> str:
    """Return *source* with each fix's line replaced by its suggestion.

    Only lines that still match the recorded ``original`` text are replaced, so
    applying a stale fix set is a no-op rather than a corruption.
    """
    if not fixes:
        return source
    lines = source.splitlines(keepends=True)
    by_line = {}
    for f in fixes:
        by_line.setdefault(f.line, f)
    out = []
    for idx, raw in enumerate(lines, start=1):
        f = by_line.get(idx)
        if f is not None and raw.rstrip("\n").rstrip() == f.original:
            newline = "\n" if raw.endswith("\n") else ""
            # Preserve original leading indentation already captured in
            # f.suggested (it is the full line text), just re-attach newline.
            out.append(f.suggested + newline)
        else:
            out.append(raw)
    return "".join(out)


def format_autofixes_plain(fixes: List[AutoFix]) -> str:
    """Render fixes as a plain unified-diff-style suggestion block."""
    if not fixes:
        return ""
    lines = [f"Suggested fixes ({len(fixes)}):"]
    for f in fixes:
        lines.append(f"  {f.description}")
        lines.append(f"    line {f.line}:")
        lines.append(f"      - {f.original.strip()}")
        lines.append(f"      + {f.suggested.strip()}")
    return "\n".join(lines)


def format_autofixes_ansi(fixes: List[AutoFix]) -> str:
    """Render fixes with ANSI color (red removal, green addition)."""
    if not fixes:
        return ""
    red = "\033[31m"
    green = "\033[32m"
    bold = "\033[1m"
    reset = "\033[0m"
    lines = [f"{bold}Suggested fixes ({len(fixes)}):{reset}"]
    for f in fixes:
        lines.append(f"  {f.description}")
        lines.append(f"    line {f.line}:")
        lines.append(f"      {red}- {f.original.strip()}{reset}")
        lines.append(f"      {green}+ {f.suggested.strip()}{reset}")
    return "\n".join(lines)
