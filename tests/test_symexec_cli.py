"""Step 66 — CLI surface for the symbolic-execution engine.

``tensorguard symexec <path> [--engine symexec|fx|both] [--explain]
[--fingerprint] [--format text|json]`` runs the symexec engine (and optionally
the FX path) over files/directories and renders the findings.  These tests drive
the command through ``ReftypeCliApp``/``main`` and assert on output + exit codes.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from src.cli.main import SymexecCommand, main


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


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return str(p)


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


# -- registration --------------------------------------------------------


def test_symexec_command_is_registered():
    from src.cli.main import ReftypeCliApp

    assert "symexec" in ReftypeCliApp.COMMANDS
    assert isinstance(ReftypeCliApp.COMMANDS["symexec"](), SymexecCommand)


# -- text output + exit codes -------------------------------------------


def test_text_reports_bug_and_exits_nonzero(tmp_path):
    f = _write(tmp_path, "buggy.py", _BUGGY)
    code, out = _run(["symexec", f])
    assert code == 1
    assert "broadcast_mismatch" in out
    assert "1 bug(s) found" in out


def test_clean_file_exits_zero(tmp_path):
    f = _write(tmp_path, "clean.py", _CLEAN)
    code, out = _run(["symexec", f])
    assert code == 0
    assert "0 bug(s) found" in out


def test_missing_path_exits_nonzero():
    code, out = _run(["symexec", "/no/such/path_xyz"])
    assert code == 1


# -- engine selection ----------------------------------------------------


def test_engine_both_labels_sections(tmp_path):
    f = _write(tmp_path, "buggy.py", _BUGGY)
    code, out = _run(["symexec", f, "--engine", "both"])
    assert code == 1
    assert "[symexec]" in out
    assert "[fx]" in out


def test_engine_symexec_only_has_no_fx_section(tmp_path):
    f = _write(tmp_path, "buggy.py", _BUGGY)
    code, out = _run(["symexec", f, "--engine", "symexec"])
    assert "[fx]" not in out


# -- format json ---------------------------------------------------------


def test_json_format_is_parseable(tmp_path):
    f = _write(tmp_path, "buggy.py", _BUGGY)
    code, out = _run(["symexec", f, "--format", "json"])
    assert code == 1
    payload = json.loads(out)
    assert "results" in payload
    rec = payload["results"][0]
    sym = rec["engines"]["symexec"]
    assert sym["bugs"][0]["kind"] == "broadcast_mismatch"
    assert len(sym["fingerprint"]) == 64


def test_json_fingerprint_is_deterministic(tmp_path):
    f = _write(tmp_path, "buggy.py", _BUGGY)
    _, out1 = _run(["symexec", f, "--format", "json"])
    _, out2 = _run(["symexec", f, "--format", "json"])
    d1 = json.loads(out1)["results"][0]["engines"]["symexec"]["fingerprint"]
    d2 = json.loads(out2)["results"][0]["engines"]["symexec"]["fingerprint"]
    assert d1 == d2


# -- explain & fingerprint flags ----------------------------------------


def test_explain_flag_renders_derivation(tmp_path):
    f = _write(tmp_path, "buggy.py", _BUGGY)
    code, out = _run(["symexec", f, "--explain"])
    assert "[BROADCAST_MISMATCH]" in out
    assert "confidence:" in out


def test_fingerprint_flag_prints_digest(tmp_path):
    f = _write(tmp_path, "buggy.py", _BUGGY)
    code, out = _run(["symexec", f, "--fingerprint"])
    assert "fingerprint:" in out


# -- directory traversal -------------------------------------------------


def test_directory_traversal_finds_files(tmp_path):
    _write(tmp_path, "a.py", _BUGGY)
    _write(tmp_path, "b.py", _CLEAN)
    code, out = _run(["symexec", str(tmp_path)])
    assert code == 1
    assert "analyzed 2 file(s)" in out


# -- output to file ------------------------------------------------------


def test_output_to_file(tmp_path):
    f = _write(tmp_path, "buggy.py", _BUGGY)
    outp = tmp_path / "report.txt"
    code, stdout = _run(["symexec", f, "-o", str(outp)])
    assert code == 1
    assert stdout == ""  # nothing on stdout
    assert "broadcast_mismatch" in outp.read_text(encoding="utf-8")


# -- coverage flag (Step 77) --------------------------------------------


def test_coverage_flag_prints_statement_profile(tmp_path):
    f = _write(tmp_path, "clean.py", _CLEAN)
    code, out = _run(["symexec", f, "--coverage"])
    assert "coverage:" in out
    assert "statements non-Top" in out


def test_coverage_in_json_output(tmp_path):
    f = _write(tmp_path, "clean.py", _CLEAN)
    code, out = _run(["symexec", f, "--format", "json"])
    rec = json.loads(out)["results"][0]
    cov = rec["engines"]["symexec"]["coverage"]
    assert 0.0 <= cov["coverage"] <= 1.0
    assert cov["total_statements"] >= cov["non_top_statements"]


# -- benchmark flag (Step 78) -------------------------------------------


def test_benchmark_flag_prints_profile(tmp_path):
    f = _write(tmp_path, "clean.py", _CLEAN)
    code, out = _run(["symexec", f, "--benchmark"])
    assert code == 0
    assert "performance benchmark" in out
    assert "iteration caps" in out
    assert "mean latency" in out


def test_benchmark_json_has_summary_and_files(tmp_path):
    f = _write(tmp_path, "clean.py", _CLEAN)
    code, out = _run(["symexec", f, "--benchmark", "--format", "json"])
    data = json.loads(out)
    assert "summary" in data and "files" in data
    assert data["summary"]["files"] == 1
    assert "iteration_caps" in data["summary"]


def test_budget_ms_flag_is_accepted(tmp_path):
    f = _write(tmp_path, "clean.py", _CLEAN)
    code, out = _run(["symexec", f, "--budget-ms", "5000"])
    # A generous budget changes nothing: the clean file still reports no bug.
    assert code == 0
