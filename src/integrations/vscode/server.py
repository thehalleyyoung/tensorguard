"""Minimal LSP server that runs TensorGuard on file save.

This is a single-threaded LSP server using JSON-RPC over stdio.
It responds to ``textDocument/didSave`` by running TensorGuard analysis
and publishing diagnostics.

Launch::

    python -m src.integrations.vscode.server

In VS Code ``settings.json``::

    {
        "tensorguard.server.path": "python -m src.integrations.vscode.server"
    }
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, Dict, List, Optional

from .diagnostics import analysis_result_to_diagnostics


class LSPServer:
    """Minimal Language Server Protocol server for TensorGuard.

    Handles initialize, textDocument/didOpen, textDocument/didSave,
    and shutdown/exit.
    """

    def __init__(self) -> None:
        self._running = True
        self._initialized = False
        self._documents: Dict[str, str] = {}  # uri -> content

    def run(self) -> None:
        """Main loop: read JSON-RPC messages from stdin, dispatch."""
        while self._running:
            try:
                message = self._read_message()
                if message is None:
                    break
                self._handle_message(message)
            except Exception as e:
                self._log(f"Error: {e}")

    def _read_message(self) -> Optional[Dict[str, Any]]:
        """Read a JSON-RPC message from stdin (Content-Length header)."""
        headers: Dict[str, str] = {}
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            line_str = line.decode("utf-8").strip()
            if not line_str:
                break
            if ":" in line_str:
                key, value = line_str.split(":", 1)
                headers[key.strip()] = value.strip()

        content_length = int(headers.get("Content-Length", "0"))
        if content_length == 0:
            return None

        body = sys.stdin.buffer.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _send_message(self, msg: Dict[str, Any]) -> None:
        """Write a JSON-RPC message to stdout."""
        body = json.dumps(msg)
        content = f"Content-Length: {len(body)}\r\n\r\n{body}"
        sys.stdout.buffer.write(content.encode("utf-8"))
        sys.stdout.buffer.flush()

    def _send_response(self, req_id: Any, result: Any) -> None:
        self._send_message({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _send_notification(self, method: str, params: Any) -> None:
        self._send_message({"jsonrpc": "2.0", "method": method, "params": params})

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        method = msg.get("method", "")
        req_id = msg.get("id")

        if method == "initialize":
            self._handle_initialize(req_id, msg.get("params", {}))
        elif method == "initialized":
            self._initialized = True
        elif method == "textDocument/didOpen":
            self._handle_did_open(msg.get("params", {}))
        elif method == "textDocument/didSave":
            self._handle_did_save(msg.get("params", {}))
        elif method == "textDocument/didClose":
            self._handle_did_close(msg.get("params", {}))
        elif method == "shutdown":
            self._send_response(req_id, None)
            self._running = False
        elif method == "exit":
            self._running = False

    def _handle_initialize(self, req_id: Any, params: Dict[str, Any]) -> None:
        capabilities = {
            "textDocumentSync": {
                "openClose": True,
                "save": {"includeText": True},
                "change": 0,  # None — we only analyze on save
            },
        }
        self._send_response(req_id, {
            "capabilities": capabilities,
            "serverInfo": {"name": "tensorguard-lsp", "version": "0.2.0"},
        })

    def _handle_did_open(self, params: Dict[str, Any]) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        text = td.get("text", "")
        self._documents[uri] = text
        self._analyze_and_publish(uri, text)

    def _handle_did_save(self, params: Dict[str, Any]) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        text = params.get("text") or self._documents.get(uri, "")
        if text:
            self._documents[uri] = text
        self._analyze_and_publish(uri, text)

    def _handle_did_close(self, params: Dict[str, Any]) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        self._documents.pop(uri, None)
        # Clear diagnostics
        self._send_notification("textDocument/publishDiagnostics", {
            "uri": uri,
            "diagnostics": [],
        })

    def _analyze_and_publish(self, uri: str, text: str) -> None:
        """Run TensorGuard analysis and publish diagnostics."""
        if not text or not uri.endswith(".py"):
            return

        try:
            from src.api import analyze
            result = analyze(text, filename=uri)
            diagnostics = analysis_result_to_diagnostics(result, uri=uri)
        except Exception as e:
            self._log(f"Analysis failed for {uri}: {e}")
            diagnostics = []

        self._send_notification("textDocument/publishDiagnostics", {
            "uri": uri,
            "diagnostics": diagnostics,
        })

    def _log(self, message: str) -> None:
        """Send a log message to the client."""
        self._send_notification("window/logMessage", {
            "type": 4,  # Log
            "message": f"[tensorguard] {message}",
        })


def main() -> None:
    """Entry point for the LSP server."""
    server = LSPServer()
    server.run()


if __name__ == "__main__":
    main()
