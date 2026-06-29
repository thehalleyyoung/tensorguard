"""Tests for the symexec documentation (roadmap Step 88).

Documentation rots silently.  These tests pin the *executable* claims in the
docs: that every public symbol the docs reference is importable, that the
worked examples produce the documented output, and that the doc files exist and
cross-link.
"""

import os
import re

from src.symexec import analyze_source, SymConfig


ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
API_MD = os.path.join(ROOT, "API.md")
ENGINE_MD = os.path.join(ROOT, "docs", "symexec", "engine.md")


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# Doc files exist and cross-link                                              #
# --------------------------------------------------------------------------- #

def test_engine_doc_exists():
    assert os.path.isfile(ENGINE_MD)


def test_api_md_has_symexec_section():
    text = _read(API_MD)
    assert "## Symbolic-Execution Engine (`src.symexec`)" in text


def test_engine_doc_links_to_api():
    assert "API.md" in _read(ENGINE_MD)


# --------------------------------------------------------------------------- #
# Every public symbol the docs reference is importable                        #
# --------------------------------------------------------------------------- #

def test_documented_public_symbols_importable():
    import src.symexec as se

    documented = [
        "analyze_source", "analyze_file", "SymResult", "SymConfig",
        "DEFAULT_CONFIG", "MODES", "analyze_package", "analyze_package_parallel",
        "IncrementalCache", "analyze_package_incremental",
        "analyze_source_incremental", "TelemetrySink", "CalibrationRecord",
        "calibration_report", "records_to_jsonl", "records_from_jsonl",
        "STUB_REGISTRY", "analyze_notebook", "parse_notebook",
        "to_publish_diagnostics", "to_lsp_diagnostics", "NotebookResult",
    ]
    for name in documented:
        assert hasattr(se, name), f"public symbol {name} missing from src.symexec"


def test_symresult_documented_methods_exist():
    r = analyze_source("x = 1\n")
    for m in (
        "fingerprint", "footprint", "explain", "to_dict", "to_sarif",
        "to_lsp_diagnostics", "to_github_annotations",
    ):
        assert hasattr(r, m), f"SymResult.{m} documented but missing"


# --------------------------------------------------------------------------- #
# Worked examples produce the documented output                               #
# --------------------------------------------------------------------------- #

_MATMUL_SRC = """
import torch

def f():
    a = torch.randn(2, 3)
    b = torch.randn(4, 5)
    return a @ b
"""

_HEURISTIC_SRC = """
import torch
def g(n):
    a = torch.randn(7)
    b = torch.randn(n)
    return a + b
"""


def test_matmul_worked_example():
    r = analyze_source(_MATMUL_SRC, filename="demo.py")
    assert len(r.bugs) == 1
    bug = r.bugs[0]
    assert bug.kind.value == "matmul_dim_mismatch"
    assert bug.line == 7 and bug.col == 11
    assert "(2, 3) @ (4, 5)" in bug.message
    assert round(bug.confidence, 2) == 0.99


def test_heuristic_worked_example():
    assert analyze_source(_HEURISTIC_SRC, config=SymConfig.balanced()).bugs == []
    heur = analyze_source(_HEURISTIC_SRC, config=SymConfig.heuristic())
    assert len(heur.bugs) == 1
    assert heur.bugs[0].kind.value == "broadcast_mismatch"
    assert round(heur.bugs[0].confidence, 2) == 0.53
    assert "suspected" in heur.bugs[0].message.lower()


def test_mode_table_knobs_match_docs():
    # The API.md / engine.md mode table claims these exact knob settings.
    assert SymConfig.balanced().min_confidence == 0.0
    assert SymConfig.balanced().require_feasibility is False
    assert SymConfig.balanced().enable_heuristics is False
    assert SymConfig.sound().min_confidence == 0.85
    assert SymConfig.sound().require_feasibility is True
    assert SymConfig.heuristic().enable_heuristics is True


def test_code_fences_balanced_in_engine_doc():
    # Guard against a malformed doc (odd number of ``` fences).
    fences = re.findall(r"^```", _read(ENGINE_MD), flags=re.MULTILINE)
    assert len(fences) % 2 == 0
