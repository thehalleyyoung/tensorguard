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

def test_fix_sarif_format_attaches_suggested_fix(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(
        "import torch\n"
        "def g():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    return x.reshape(6, 5)\n",
        encoding="utf-8",
    )
    code, out = _run(["fix", str(f), "--format", "sarif"])
    assert code == 0
    log = json.loads(out)
    assert log["version"] == "2.1.0"
    results = log["runs"][0]["results"]
    res = next(r for r in results if r["ruleId"] == "reshape_size_mismatch")
    # The verified repair is surfaced as a SARIF "Apply suggested fix".
    fix = res["fixes"][0]
    rep = fix["artifactChanges"][0]["replacements"][0]
    assert rep["deletedRegion"]["startLine"] == 4
    assert "reshape(6, -1)" in rep["insertedContent"]["text"]
    # SARIF mode must never modify the source file.
    assert "reshape(6, 5)" in f.read_text(encoding="utf-8")


def test_fix_sarif_insertion_is_zero_width_region(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(BUGGY, encoding="utf-8")
    code, out = _run(["fix", str(f), "--format", "sarif"])
    assert code == 0
    log = json.loads(out)
    results = log["runs"][0]["results"]
    res = next(r for r in results if r["ruleId"] == "missing_super_init")
    rep = res["fixes"][0]["artifactChanges"][0]["replacements"][0]
    # Pure insertion: zero-width deleted region, inserted super() call.
    assert rep["deletedRegion"]["startColumn"] == rep["deletedRegion"]["endColumn"]
    assert "super().__init__()" in rep["insertedContent"]["text"]


def test_fix_patch_format_is_git_applyable(tmp_path):
    import subprocess

    # A real git repo so we can validate the patch with `git apply --check`.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    src = (
        "import torch\n"
        "def g():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(2, 5)\n"
        "    y = x.reshape(6, 5)\n"
        "    z = torch.cat([a, b], dim=0)\n"
        "    return y, z\n"
    )
    (tmp_path / "model.py").write_text(src, encoding="utf-8")
    subprocess.run(["git", "add", "model.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=tmp_path, check=True,
    )

    # Run with a repo-relative path so the patch carries `a/model.py`.
    app = ReftypeCliApp()
    buf = io.StringIO()
    import os
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        with redirect_stdout(buf):
            code = app.run(["fix", "model.py", "--format", "patch"])
    finally:
        os.chdir(cwd)
    out = buf.getvalue()
    assert code == 0

    # One coherent patch composing BOTH verified fixes.
    assert out.startswith("diff --git a/model.py b/model.py\n")
    assert "--- a/model.py" in out
    assert "+++ b/model.py" in out
    assert "@@" in out
    assert "reshape(6, -1)" in out
    assert "dim=1" in out

    # patch mode must not touch the file.
    assert "reshape(6, 5)" in (tmp_path / "model.py").read_text(encoding="utf-8")

    # And `git apply` accepts it.
    (tmp_path / "fix.patch").write_text(out, encoding="utf-8")
    chk = subprocess.run(
        ["git", "apply", "--check", "-p1", "fix.patch"], cwd=tmp_path,
        capture_output=True, text=True,
    )
    assert chk.returncode == 0, chk.stderr


def test_fix_patch_format_empty_when_clean(tmp_path):
    f = tmp_path / "model.py"
    f.write_text("import torch\ndef g():\n    return torch.zeros(2, 3)\n", encoding="utf-8")
    code, out = _run(["fix", str(f), "--format", "patch"])
    assert code == 0
    assert out == ""


def test_fix_explain_shows_provenance_and_proof(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(
        "import torch\n"
        "def g():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    return x.reshape(6, 5)\n",
        encoding="utf-8",
    )
    code, out = _run(["fix", str(f), "--explain"])
    assert code == 0
    # The originating finding (the constraint that justified the value)...
    assert "finding: reshape target (6, 5) is incompatible" in out
    # ...and the re-verification proof.
    assert "proof: re-verified: targeted bug gone, no new bug introduced" in out
    # explain must not modify the file.
    assert "reshape(6, 5)" in f.read_text(encoding="utf-8")


def test_fix_without_explain_omits_provenance(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(
        "import torch\n"
        "def g():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    return x.reshape(6, 5)\n",
        encoding="utf-8",
    )
    code, out = _run(["fix", str(f)])
    assert code == 0
    assert "finding:" not in out
    assert "proof:" not in out
