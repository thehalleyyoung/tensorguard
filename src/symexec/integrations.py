"""Editor / hook / CI wiring for the symbolic-execution engine (Step 69).

The symexec engine is **torch-free**, so it can run in exactly the environments
where the FX/SMT path cannot: a contributor's editor, a ``pre-commit`` hook on a
laptop without a GPU/torch install, or a lightweight CI job.  This module surfaces
:class:`~src.symexec.engine.SymResult` findings in the three shapes those
surfaces consume:

* :func:`to_lsp_diagnostics` — Language-Server-Protocol ``Diagnostic[]`` for
  ``textDocument/publishDiagnostics`` (inline editor squiggles).
* :func:`to_github_annotations` / :func:`render_github_annotations` — GitHub
  Actions ``::error file=…::`` workflow commands (inline PR annotations).

The pre-commit surface needs no new code: the Step-66 ``tensorguard symexec``
command already takes the staged file paths and exits non-zero on a real bug, so
``.pre-commit-hooks.yml`` simply points a torch-free hook at it.

Like the rest of ``src/symexec``, this module is self-contained (it inlines the
tiny pure escaping/position helpers rather than importing ``src.github_action`` /
``src.lsp_provider``) so the engine keeps no hard dependency on the rest of
TensorGuard, and it is pure: it only reads already-computed result data.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = [
    "to_lsp_diagnostics",
    "to_publish_diagnostics",
    "to_github_annotations",
    "render_github_annotations",
    "LSP_SOURCE",
]

#: ``source`` field stamped on every emitted LSP diagnostic so an editor can
#: distinguish symexec findings from the FX path's (``"tensorguard"``).
LSP_SOURCE = "tensorguard-symexec"

# LSP DiagnosticSeverity numbering.
_LSP_SEVERITY = {"error": 1, "warning": 2, "warn": 2, "note": 3, "info": 3,
                 "information": 3, "hint": 4}
# GitHub workflow-command annotation levels (severity → level).
_GH_LEVEL = {"error": "error", "fatal": "error", "warning": "warning",
             "warn": "warning", "note": "notice", "info": "notice"}


# -- LSP -----------------------------------------------------------------


def _lsp_pos(line_1indexed: int, col_0indexed: int) -> Dict[str, int]:
    """A 1-indexed line / 0-indexed column as a 0-indexed LSP Position."""
    return {
        "line": max(0, int(line_1indexed) - 1),
        "character": max(0, int(col_0indexed)),
    }


def to_lsp_diagnostics(result: Any, uri: str = "") -> List[Dict[str, Any]]:
    """Render a :class:`SymResult`'s bugs as LSP ``Diagnostic`` objects.

    The calibrated confidence and the mechanical fix suggestion are folded into
    the message; ``code`` carries the bug kind and ``source`` is
    :data:`LSP_SOURCE`.  Lines with no usable position are skipped."""
    out: List[Dict[str, Any]] = []
    for b in getattr(result, "bugs", []) or []:
        line = int(getattr(b, "line", 0) or 0)
        if line <= 0:
            continue
        col = int(getattr(b, "col", 0) or 0)
        sev = _LSP_SEVERITY.get(str(getattr(b, "severity", "error") or "error").lower(), 1)
        message = getattr(b, "message", "") or ""
        conf = getattr(b, "confidence", None)
        if conf is not None:
            message = f"{message} (confidence {float(conf):.2f})"
        fix = getattr(b, "fix_suggestion", None)
        if fix:
            message = f"{message}\n\nfix: {fix}"
        diag: Dict[str, Any] = {
            "range": {
                "start": _lsp_pos(line, col),
                "end": _lsp_pos(line, col + 1),
            },
            "severity": sev,
            "source": LSP_SOURCE,
            "code": getattr(getattr(b, "kind", None), "value", ""),
            "message": message,
        }
        out.append(diag)
    return out


def to_publish_diagnostics(result: Any, uri: str) -> Dict[str, Any]:
    """A full LSP ``textDocument/publishDiagnostics`` JSON-RPC notification.

    VS Code (and any LSP client) consumes this verbatim to populate the Problems
    panel and the inline squiggles for ``uri``.  An empty ``diagnostics`` list
    clears any previously-published findings for the document, so re-publishing
    after a fix removes the markers.  The ``uri`` must be a document URI such as
    ``file:///abs/path/model.py``."""
    return {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": uri,
            "diagnostics": to_lsp_diagnostics(result, uri=uri),
        },
    }


# -- GitHub Actions ------------------------------------------------------


def _escape_data(message: str) -> str:
    return (
        (message or "")
        .replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def _escape_property(value: str) -> str:
    return _escape_data(value).replace(":", "%3A").replace(",", "%2C")


def _annotation(filename: str, bug) -> str:
    line = int(getattr(bug, "line", 0) or 0)
    col = int(getattr(bug, "col", 0) or 0)
    level = _GH_LEVEL.get(str(getattr(bug, "severity", "error") or "error").lower(), "error")
    kind = getattr(getattr(bug, "kind", None), "value", "")
    props = [f"file={_escape_property(filename)}"]
    if line > 0:
        props.append(f"line={line}")
    # GitHub annotation columns are 1-based; symexec columns are 0-based.
    if col >= 0:
        props.append(f"col={col + 1}")
    if kind:
        props.append(f"title={_escape_property('TensorGuard: ' + kind)}")
    message = getattr(bug, "message", "") or ""
    conf = getattr(bug, "confidence", None)
    if conf is not None:
        message = f"{message} (confidence {float(conf):.2f})"
    return f"::{level} {','.join(props)}::{_escape_data(message)}"


def to_github_annotations(result: Any, filename: str = "") -> List[str]:
    """Render a :class:`SymResult`'s bugs as GitHub Actions annotation commands."""
    return [_annotation(filename, b) for b in (getattr(result, "bugs", []) or [])]


def render_github_annotations(result: Any, filename: str = "") -> str:
    """Newline-joined GitHub annotation commands (ready to print in a CI job)."""
    return "\n".join(to_github_annotations(result, filename))
