"""Editor-integration layer: turn a verification result into LSP payloads.

Step 61.  A VSCode (or any LSP) extension needs three things to give a
first-class editing experience: inline squiggles (``textDocument/publishDiagnostics``),
hover shapes (``textDocument/hover``), and quick-fixes
(``textDocument/codeAction``).  TensorGuard already computes the substance of
all three -- source-mapped diagnostics (Step 57), the shape-inference chain
(Step 58), and mechanical autofixes (Step 59).  This module is the thin, pure
adapter that renders those into the exact JSON shapes the LSP spec defines, so
an extension is a trivial transport shim over a real verifier rather than a
re-implementation.

Everything here is pure and defensive: it reads only public ``AnalysisResult``
fields, converts TensorGuard's 1-indexed (line, col) source positions to LSP's
0-indexed positions, and never raises on missing data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# LSP DiagnosticSeverity
_SEVERITY = {"error": 1, "warning": 2, "info": 3, "information": 3, "hint": 4}


def _pos(line_1indexed: int, col_0indexed: int) -> Dict[str, int]:
    """Convert a 1-indexed line / 0-indexed col to an LSP Position."""
    return {
        "line": max(0, int(line_1indexed) - 1),
        "character": max(0, int(col_0indexed)),
    }


def to_lsp_diagnostics(
    result: Any, uri: str = ""
) -> List[Dict[str, Any]]:
    """Render ``result.diagnostics`` as LSP ``Diagnostic`` objects."""
    out: List[Dict[str, Any]] = []
    for d in getattr(result, "diagnostics", []) or []:
        try:
            line = int(getattr(d, "source_line", 0) or 0)
            col = int(getattr(d, "source_col", 0) or 0)
            if line <= 0:
                continue
            snippet = getattr(d, "source_snippet", None) or ""
            end_col = len(snippet) if snippet else col + 1
            if end_col <= col:
                end_col = col + 1
            sev = _SEVERITY.get(
                str(getattr(d, "severity", "error") or "error").lower(), 1
            )
            message = getattr(d, "message", "") or ""
            fix = getattr(d, "fix_suggestion", None)
            if fix:
                message = f"{message}\n\nfix: {fix}"
            diag: Dict[str, Any] = {
                "range": {"start": _pos(line, col), "end": _pos(line, end_col)},
                "severity": sev,
                "source": "tensorguard",
                "message": message,
            }
            rels = []
            for rl in getattr(d, "related_locations", []) or []:
                rline = int(getattr(rl, "line", 0) or 0)
                if rline <= 0:
                    continue
                rcol = int(getattr(rl, "col", 0) or 0)
                rels.append({
                    "location": {
                        "uri": uri,
                        "range": {
                            "start": _pos(rline, rcol),
                            "end": _pos(rline, rcol + 1),
                        },
                    },
                    "message": getattr(rl, "message", "") or "",
                })
            if rels:
                diag["relatedInformation"] = rels
            out.append(diag)
        except Exception:
            continue
    return out


def collect_shape_hovers(result: Any) -> Dict[int, str]:
    """Map each source line to a human-readable produced-shape string.

    Built from the inference chain: every link that produced a concrete output
    shape contributes ``"<tensor> : <shape>"`` for its source line, so hovering
    that line in an editor shows the tensor's shape.
    """
    hovers: Dict[int, str] = {}
    chain = getattr(result, "inference_chain", None)
    if chain is None:
        return hovers
    for link in getattr(chain, "links", []) or []:
        line = int(getattr(link, "line", 0) or 0)
        shape = getattr(link, "output_shape", None)
        name = getattr(link, "output", None) or "result"
        if line > 0 and shape and shape != "?":
            # Keep the first (earliest) concrete shape seen for a line.
            hovers.setdefault(line, f"{name} : {shape}")
    return hovers


def hover_at(result: Any, line_1indexed: int) -> Optional[Dict[str, Any]]:
    """Return an LSP ``Hover`` for *line_1indexed*, or None if no shape known."""
    text = collect_shape_hovers(result).get(int(line_1indexed))
    if text is None:
        return None
    return {
        "contents": {
            "kind": "markdown",
            "value": f"**tensorguard** &nbsp; `{text}`",
        }
    }


def to_lsp_code_actions(
    result: Any, uri: str = ""
) -> List[Dict[str, Any]]:
    """Render ``result.autofixes`` as LSP ``CodeAction`` quick-fixes."""
    actions: List[Dict[str, Any]] = []
    for f in getattr(result, "autofixes", []) or []:
        try:
            line = int(getattr(f, "line", 0) or 0)
            if line <= 0:
                continue
            original = getattr(f, "original", "") or ""
            suggested = getattr(f, "suggested", "") or ""
            arg = (
                "in_features"
                if getattr(f, "kind", "") == "linear_in_features"
                else "in_channels"
            )
            edit = {
                "range": {
                    "start": {"line": line - 1, "character": 0},
                    "end": {"line": line - 1, "character": len(original)},
                },
                "newText": suggested,
            }
            actions.append({
                "title": (
                    f"tensorguard: set {arg}={getattr(f, 'new_value', '?')} "
                    f"on {getattr(f, 'layer', '')}"
                ),
                "kind": "quickfix",
                "isPreferred": True,
                "edit": {"changes": {uri: [edit]}},
            })
        except Exception:
            continue
    return actions


def build_lsp_report(result: Any, uri: str = "") -> Dict[str, Any]:
    """Aggregate diagnostics, code actions and hovers into one payload.

    This is what an extension fetches per document: publish ``diagnostics`` as
    squiggles, register ``codeActions`` as quick-fixes, and answer hover
    requests from ``hovers`` (a list of ``{line, contents}`` with 1-indexed
    lines, matching the editor's view of the source).
    """
    hovers = [
        {"line": line, "contents": text}
        for line, text in sorted(collect_shape_hovers(result).items())
    ]
    return {
        "uri": uri,
        "diagnostics": to_lsp_diagnostics(result, uri),
        "codeActions": to_lsp_code_actions(result, uri),
        "hovers": hovers,
    }
