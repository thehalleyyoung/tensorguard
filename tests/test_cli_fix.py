"""Tests for the `tensorguard fix` CLI command (machine-verified auto-repair).

The command runs the symexec verified-repair loop over real files: it prints
unified diffs by default and applies them in place with ``--write``. Every fix
it surfaces has already been re-checked by the engine (targeted bug gone, no new
bug kind), so these tests assert both the diff output and that ``--write``
produces a file that no longer triggers the bug.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from src.cli.main import ReftypeCliApp

BUGGY = (
    "import torch.nn as nn\n"
    "\n"
    "\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        self.fc = nn.Linear(3, 4)\n"
    "\n"
    "    def forward(self, x):\n"
    "        return self.fc.forward(x)\n"
)


def _run(argv):
    app = ReftypeCliApp()
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = app.run(argv)
    return code, buf.getvalue()


def test_fix_prints_diffs_without_writing(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(BUGGY, encoding="utf-8")

    code, out = _run(["fix", str(f)])
    assert code == 0
    # Both the intent bug (missing super) and the forward-call bug are surfaced.
    assert "missing_super_init" in out
    assert "direct_forward_call" in out
    assert "+        super().__init__()" in out
    assert "+        return self.fc(x)" in out
    assert "verified fix" in out
    # Diff-only mode must NOT modify the file.
    assert f.read_text(encoding="utf-8") == BUGGY


def test_fix_write_applies_and_is_idempotent(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(BUGGY, encoding="utf-8")

    code, out = _run(["fix", str(f), "--write"])
    assert code == 0
    assert "applied" in out

    patched = f.read_text(encoding="utf-8")
    assert "super().__init__()" in patched
    assert "self.fc(x)" in patched
    assert ".forward(" not in patched

    # Re-running finds nothing left to fix and leaves the file untouched.
    code2, out2 = _run(["fix", str(f)])
    assert code2 == 0
    assert "no repairable bugs found" in out2
    assert f.read_text(encoding="utf-8") == patched


def test_fix_json_format(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(BUGGY, encoding="utf-8")

    code, out = _run(["fix", str(f), "--format", "json"])
    assert code == 0
    data = json.loads(out)
    assert data["verified_fixes"] >= 2
    kinds = {fx["kind"] for rec in data["files"] for fx in rec["fixes"]}
    assert "missing_super_init" in kinds
    assert "direct_forward_call" in kinds


def test_fix_clean_file_reports_nothing(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text(
        "import torch\n"
        "def f():\n"
        "    x = torch.randn(2, 3)\n"
        "    return x.reshape(6)\n",
        encoding="utf-8",
    )
    code, out = _run(["fix", str(f)])
    assert code == 0
    assert "no repairable bugs found" in out
