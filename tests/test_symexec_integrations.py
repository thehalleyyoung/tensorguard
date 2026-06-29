"""Step 69 — editor / hook / CI wiring for the symexec engine.

The torch-free symexec engine surfaces its findings to the three integration
surfaces an owner uses while developing: LSP diagnostics (editor squiggles) and
GitHub Actions annotations (inline PR comments) via
:mod:`src.symexec.integrations`, plus the ``tensorguard symexec`` pre-commit hook
(declared in ``.pre-commit-hooks.yml``).  These tests pin the shapes of each and
the ``--format github|lsp`` CLI surface.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from src.cli.main import main
from src.symexec import (
    LSP_SOURCE,
    analyze_source,
    render_github_annotations,
    to_github_annotations,
    to_lsp_diagnostics,
)

_BUGGY = (
    "import torch\n"
    "def f():\n"
    "    a = torch.zeros(3)\n"
    "    b = torch.zeros(2)\n"
    "    return a + b\n"
)
_CLEAN = (
    "import torch\n"
    "def f():\n"
    "    a = torch.zeros(2, 3)\n"
    "    b = torch.zeros(3, 4)\n"
    "    return a @ b\n"
)


def _analyze(src, name="m.py"):
    return analyze_source(src, filename=name)


# -- LSP -----------------------------------------------------------------


def test_lsp_diagnostic_shape():
    sr = _analyze(_BUGGY, "buggy.py")
    diags = to_lsp_diagnostics(sr, uri="buggy.py")
    assert len(diags) == 1
    d = diags[0]
    assert d["source"] == LSP_SOURCE
    assert d["code"] == "broadcast_mismatch"
    assert d["severity"] == 1  # error
    assert set(d["range"]) == {"start", "end"}
    assert set(d["range"]["start"]) == {"line", "character"}


def test_lsp_line_is_zero_based():
    sr = _analyze(_BUGGY, "buggy.py")
    bug = sr.bugs[0]
    d = to_lsp_diagnostics(sr, uri="buggy.py")[0]
    # symexec line is 1-based; LSP line is 0-based.
    assert d["range"]["start"]["line"] == bug.line - 1
    assert d["range"]["start"]["character"] == bug.col


def test_lsp_message_carries_confidence_and_fix():
    d = to_lsp_diagnostics(_analyze(_BUGGY, "buggy.py"), uri="buggy.py")[0]
    assert "confidence" in d["message"]


def test_lsp_clean_is_empty():
    assert to_lsp_diagnostics(_analyze(_CLEAN, "clean.py"), uri="clean.py") == []


def test_symresult_to_lsp_diagnostics_method():
    sr = _analyze(_BUGGY, "buggy.py")
    assert sr.to_lsp_diagnostics(uri="buggy.py") == to_lsp_diagnostics(sr, uri="buggy.py")


# -- GitHub Actions ------------------------------------------------------


def test_github_annotation_command():
    anns = to_github_annotations(_analyze(_BUGGY, "buggy.py"), filename="buggy.py")
    assert len(anns) == 1
    cmd = anns[0]
    assert cmd.startswith("::error ")
    assert "file=buggy.py" in cmd
    assert "broadcast_mismatch" in cmd


def test_github_column_is_one_based():
    sr = _analyze(_BUGGY, "buggy.py")
    bug = sr.bugs[0]
    cmd = to_github_annotations(sr, filename="buggy.py")[0]
    # GitHub annotation col is 1-based; symexec col is 0-based.
    assert f"col={bug.col + 1}" in cmd
    assert f"line={bug.line}" in cmd


def test_github_escapes_special_chars():
    # ',' and ':' in the title must be percent-escaped per the workflow grammar.
    cmd = to_github_annotations(_analyze(_BUGGY, "buggy.py"), filename="buggy.py")[0]
    assert "%3A" in cmd  # the ':' in "TensorGuard:"


def test_render_github_annotations_joins():
    sr = _analyze(_BUGGY, "buggy.py")
    rendered = render_github_annotations(sr, filename="buggy.py")
    assert rendered == "\n".join(to_github_annotations(sr, filename="buggy.py"))


def test_github_clean_is_empty():
    assert to_github_annotations(_analyze(_CLEAN, "clean.py"), filename="clean.py") == []


def test_symresult_to_github_annotations_method():
    sr = _analyze(_BUGGY, "buggy.py")
    assert sr.to_github_annotations(filename="buggy.py") == to_github_annotations(sr, filename="buggy.py")


# -- CLI --format github / lsp ------------------------------------------


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


def test_cli_github_format(tmp_path):
    p = tmp_path / "buggy.py"
    p.write_text(_BUGGY, encoding="utf-8")
    code, out = _run(["symexec", str(p), "--format", "github"])
    assert code == 1
    assert out.startswith("::error ")
    assert "broadcast_mismatch" in out


def test_cli_lsp_format(tmp_path):
    p = tmp_path / "buggy.py"
    p.write_text(_BUGGY, encoding="utf-8")
    code, out = _run(["symexec", str(p), "--format", "lsp"])
    assert code == 1
    payload = json.loads(out)
    assert payload[0]["diagnostics"][0]["source"] == LSP_SOURCE


def test_cli_lsp_clean_exits_zero(tmp_path):
    p = tmp_path / "clean.py"
    p.write_text(_CLEAN, encoding="utf-8")
    code, out = _run(["symexec", str(p), "--format", "lsp"])
    assert code == 0
    payload = json.loads(out)
    assert payload[0]["diagnostics"] == []


# -- pre-commit hook declaration ----------------------------------------


def test_precommit_hook_declared():
    import pathlib

    import yaml  # PyYAML ships with the repo's dev deps

    root = pathlib.Path(__file__).resolve().parents[1]
    hooks = yaml.safe_load((root / ".pre-commit-hooks.yml").read_text())
    ids = {h["id"] for h in hooks}
    assert "tensorguard-symexec" in ids
    hook = next(h for h in hooks if h["id"] == "tensorguard-symexec")
    assert hook["entry"] == "tensorguard symexec"
    assert hook["types"] == ["python"]
