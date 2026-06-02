"""Step 164 — a real Language Server (JSON-RPC over stdio) for TensorGuard.

``src/lsp_provider.py`` renders an ``AnalysisResult`` into LSP payloads; this
module is the *transport*: a minimal but spec-correct LSP server that an editor
(or the bundled VS Code extension in ``editors/vscode/``) talks to over stdio.

It analyses the **unsaved editor buffer** (the text in ``didOpen`` / ``didChange``,
not the file on disk), so squiggles update as you type, and clears diagnostics
when the buffer becomes clean or is closed. Run it directly::

    python -m src.lsp_server

Implemented messages: ``initialize`` / ``initialized``, ``textDocument/didOpen``
/ ``didChange`` / ``didClose``, ``textDocument/hover``, ``textDocument/codeAction``,
``shutdown`` / ``exit``. Diagnostics are pushed via ``textDocument/publishDiagnostics``
notifications. Framing uses byte-accurate ``Content-Length`` headers per the LSP
base protocol.
"""

from __future__ import annotations

import json
import sys
from typing import Any, BinaryIO, Dict, List, Optional
from urllib.parse import unquote, urlparse

from src.lsp_provider import (
    build_lsp_report,
    hover_at,
    to_lsp_code_actions,
    to_lsp_diagnostics,
)

_SERVER_NAME = "tensorguard"


def uri_to_path(uri: str) -> str:
    """Best-effort ``file://`` URI → filesystem path (defensive, never raises)."""
    try:
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            return unquote(parsed.path) or uri
    except Exception:
        pass
    return uri


def _analyze(text: str, path: str) -> Any:
    """Verify a buffer's text; return an ``AnalysisResult`` or ``None``."""
    try:
        from src.api import verify_architecture

        return verify_architecture(text, filename=path, infer_inputs=True)
    except Exception:
        return None


class LSPServer:
    """A stdio JSON-RPC LSP server over arbitrary binary streams.

    The streams are injectable so the server can be driven in-process by a test
    (feeding framed requests through ``io.BytesIO``) exactly as an editor would.
    """

    def __init__(self, rstream: BinaryIO, wstream: BinaryIO):
        self._r = rstream
        self._w = wstream
        self._documents: Dict[str, str] = {}
        self._shutdown = False
        self._running = True

    # ---- framing ---------------------------------------------------------
    def _read_message(self) -> Optional[Dict[str, Any]]:
        headers: Dict[str, str] = {}
        while True:
            line = self._r.readline()
            if not line:
                return None  # EOF
            line = line.strip()
            if line == b"":
                break
            if b":" in line:
                key, _, value = line.partition(b":")
                headers[key.strip().decode("ascii").lower()] = value.strip().decode("ascii")
        length = int(headers.get("content-length", 0))
        if length <= 0:
            return None
        body = self._r.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return None

    def _write(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._w.write(header)
        self._w.write(body)
        try:
            self._w.flush()
        except Exception:
            pass

    def _respond(self, req_id: Any, result: Any) -> None:
        self._write({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    # ---- diagnostics -----------------------------------------------------
    def _publish(self, uri: str) -> None:
        text = self._documents.get(uri, "")
        result = _analyze(text, uri_to_path(uri))
        diagnostics: List[Dict[str, Any]] = (
            to_lsp_diagnostics(result, uri) if result is not None else []
        )
        self._notify(
            "textDocument/publishDiagnostics",
            {"uri": uri, "diagnostics": diagnostics},
        )

    # ---- dispatch --------------------------------------------------------
    def _handle(self, msg: Dict[str, Any]) -> None:
        method = msg.get("method")
        params = msg.get("params") or {}
        req_id = msg.get("id")

        if method == "initialize":
            self._respond(req_id, {
                "capabilities": {
                    "textDocumentSync": {"openClose": True, "change": 1},
                    "hoverProvider": True,
                    "codeActionProvider": True,
                },
                "serverInfo": {"name": _SERVER_NAME},
            })
        elif method == "initialized":
            pass
        elif method == "textDocument/didOpen":
            doc = params.get("textDocument", {})
            uri = doc.get("uri", "")
            self._documents[uri] = doc.get("text", "")
            self._publish(uri)
        elif method == "textDocument/didChange":
            uri = params.get("textDocument", {}).get("uri", "")
            changes = params.get("contentChanges") or []
            if changes:
                # Full-sync (change=1): the last change carries the whole buffer.
                self._documents[uri] = changes[-1].get("text", "")
            self._publish(uri)
        elif method == "textDocument/didClose":
            uri = params.get("textDocument", {}).get("uri", "")
            self._documents.pop(uri, None)
            # Clear squiggles for a closed file.
            self._notify(
                "textDocument/publishDiagnostics",
                {"uri": uri, "diagnostics": []},
            )
        elif method == "textDocument/hover":
            uri = params.get("textDocument", {}).get("uri", "")
            line0 = (params.get("position", {}) or {}).get("line", 0)
            result = _analyze(self._documents.get(uri, ""), uri_to_path(uri))
            hover = hover_at(result, int(line0) + 1) if result is not None else None
            self._respond(req_id, hover)
        elif method == "textDocument/codeAction":
            uri = params.get("textDocument", {}).get("uri", "")
            result = _analyze(self._documents.get(uri, ""), uri_to_path(uri))
            actions = to_lsp_code_actions(result, uri) if result is not None else []
            self._respond(req_id, actions)
        elif method == "shutdown":
            self._shutdown = True
            self._respond(req_id, None)
        elif method == "exit":
            self._running = False
        elif req_id is not None:
            # Unknown request: respond with a MethodNotFound error rather than hang.
            self._write({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            })

    def run(self) -> int:
        while self._running:
            msg = self._read_message()
            if msg is None:
                break
            self._handle(msg)
        return 0 if self._shutdown else 1


def build_report(text: str, uri: str = "") -> Dict[str, Any]:
    """One-shot helper: full LSP report (diagnostics/hovers/codeActions) for text."""
    result = _analyze(text, uri_to_path(uri))
    if result is None:
        return {"uri": uri, "diagnostics": [], "codeActions": [], "hovers": []}
    return build_lsp_report(result, uri)


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - stdio loop
    server = LSPServer(sys.stdin.buffer, sys.stdout.buffer)
    return server.run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
