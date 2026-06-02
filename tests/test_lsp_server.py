"""Step 164 — TensorGuard language server (LSP over stdio) + VS Code extension.

Drives the real server in-process with byte-framed JSON-RPC exactly as an editor
would, proving:

* ``initialize`` returns capabilities (textDocumentSync / hover / codeAction);
* ``didOpen`` analyses the **unsaved buffer** (buggy text) even when the file on
  disk is clean, and publishes non-empty diagnostics for the real bug;
* ``didChange`` to corrected text clears the diagnostics (empty list);
* ``didClose`` clears diagnostics;
* ``Content-Length`` is byte-accurate (validated by re-parsing every framed
  reply, including any multibyte content);
* ``shutdown``/``exit`` terminate the loop cleanly (``run() == 0``).

A separate test launches ``python -m src.lsp_server`` as a real subprocess and
completes an ``initialize`` handshake, and a third validates the VS Code
extension manifest wires the server command.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from src.lsp_server import LSPServer, uri_to_path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Conv channel mismatch (expects 16 in-channels, gets 8) — a real bug detected
# without caller-provided input shapes.
_BUGGY = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.c1 = nn.Conv2d(3, 8, 3)\n"
    "        self.c2 = nn.Conv2d(16, 32, 3)\n"
    "    def forward(self, x):\n"
    "        return self.c2(self.c1(x))\n"
)
_CLEAN = _BUGGY.replace("nn.Conv2d(16, 32, 3)", "nn.Conv2d(8, 32, 3)")


def _frame(obj) -> bytes:
    body = json.dumps(obj).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def _parse_all(raw: bytes):
    """Parse a stream of LSP-framed messages, asserting byte-accurate framing."""
    msgs = []
    i = 0
    while i < len(raw):
        header_end = raw.index(b"\r\n\r\n", i)
        header = raw[i:header_end].decode("ascii")
        length = None
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1].strip())
        assert length is not None, header
        start = header_end + 4
        body = raw[start:start + length]
        # Byte-accuracy: the declared length must equal the actual body bytes.
        assert len(body) == length, (length, len(body))
        msgs.append(json.loads(body.decode("utf-8")))
        i = start + length
    return msgs


def _run_server(messages):
    rstream = io.BytesIO(b"".join(_frame(m) for m in messages))
    wstream = io.BytesIO()
    server = LSPServer(rstream, wstream)
    rc = server.run()
    return rc, _parse_all(wstream.getvalue())


def test_initialize_didopen_didchange_clear(tmp_path):
    # The file on disk is CLEAN; the editor buffer is BUGGY. The server must
    # analyse the buffer, not disk.
    disk = tmp_path / "m.py"
    disk.write_text(_CLEAN, encoding="utf-8")
    uri = disk.as_uri()

    rc, msgs = _run_server([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "initialized", "params": {}},
        {"jsonrpc": "2.0", "method": "textDocument/didOpen",
         "params": {"textDocument": {"uri": uri, "languageId": "python",
                                     "version": 1, "text": _BUGGY}}},
        {"jsonrpc": "2.0", "method": "textDocument/didChange",
         "params": {"textDocument": {"uri": uri, "version": 2},
                    "contentChanges": [{"text": _CLEAN}]}},
        {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
        {"jsonrpc": "2.0", "method": "exit", "params": {}},
    ])

    assert rc == 0  # clean shutdown before exit

    init = next(m for m in msgs if m.get("id") == 1)
    caps = init["result"]["capabilities"]
    assert "textDocumentSync" in caps
    assert caps["hoverProvider"] is True
    assert caps["codeActionProvider"] is True

    publishes = [m for m in msgs if m.get("method") == "textDocument/publishDiagnostics"]
    assert len(publishes) == 2, msgs
    # First publish (from the buggy buffer) is non-empty and references the bug.
    first = publishes[0]["params"]
    assert first["uri"] == uri
    assert first["diagnostics"], "buffer-based diagnostics missing"
    assert any("tensorguard" == d.get("source") for d in first["diagnostics"])
    # Second publish (after fixing the buffer) clears the squiggles.
    assert publishes[1]["params"]["diagnostics"] == []

    # shutdown answered with a null result.
    sd = next(m for m in msgs if m.get("id") == 2)
    assert sd["result"] is None


def test_didclose_clears_diagnostics(tmp_path):
    uri = (tmp_path / "n.py").as_uri()
    _rc, msgs = _run_server([
        {"jsonrpc": "2.0", "method": "textDocument/didOpen",
         "params": {"textDocument": {"uri": uri, "text": _BUGGY}}},
        {"jsonrpc": "2.0", "method": "textDocument/didClose",
         "params": {"textDocument": {"uri": uri}}},
        {"jsonrpc": "2.0", "method": "exit", "params": {}},
    ])
    publishes = [m["params"] for m in msgs
                 if m.get("method") == "textDocument/publishDiagnostics"]
    assert len(publishes) == 2
    assert publishes[0]["diagnostics"]          # open -> bug
    assert publishes[1]["diagnostics"] == []     # close -> cleared


def test_unknown_request_gets_method_not_found(tmp_path):
    _rc, msgs = _run_server([
        {"jsonrpc": "2.0", "id": 9, "method": "textDocument/nonsense", "params": {}},
        {"jsonrpc": "2.0", "method": "exit", "params": {}},
    ])
    err = next(m for m in msgs if m.get("id") == 9)
    assert err["error"]["code"] == -32601


def test_uri_to_path_roundtrip(tmp_path):
    p = tmp_path / "a b.py"  # space exercises percent-decoding
    assert uri_to_path(p.as_uri()) == str(p)


def test_server_runs_as_subprocess_module():
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.lsp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
    )
    payload = b"".join(_frame(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
        {"jsonrpc": "2.0", "method": "exit", "params": {}},
    ])
    out, _ = proc.communicate(payload, timeout=120)
    msgs = _parse_all(out)
    init = next(m for m in msgs if m.get("id") == 1)
    assert "capabilities" in init["result"]
    assert proc.returncode == 0


def test_vscode_extension_manifest():
    manifest = os.path.join(_REPO, "editors", "vscode", "package.json")
    data = json.loads(open(manifest, encoding="utf-8").read())
    assert "onLanguage:python" in data["activationEvents"]
    assert data["main"].endswith("extension.js")
    props = data["contributes"]["configuration"]["properties"]
    assert props["tensorguard.serverModule"]["default"] == "src.lsp_server"
    ext = open(os.path.join(_REPO, "editors", "vscode", "src", "extension.js"),
               encoding="utf-8").read()
    assert "-m" in ext and "serverModule" in ext
