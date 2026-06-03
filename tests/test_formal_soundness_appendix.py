"""Regression tests for the generated formal soundness appendix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import src.formal_soundness_appendix as fsa


REPO = Path(__file__).resolve().parent.parent


def test_committed_appendix_tex_is_generated_from_sources():
    tex_path = REPO / "formal_soundness_appendix.tex"
    assert tex_path.exists(), "formal_soundness_appendix.tex must be committed"
    assert tex_path.read_text(encoding="utf-8") == fsa.render_latex()


def test_manifest_lean_theorems_resolve_to_namespace_qualified_declarations():
    model = fsa.build_model()
    by_name = {d.qualified_name: d for d in model.declarations}
    by_module = {}
    for decl in model.declarations:
        by_module.setdefault(decl.module, set()).add(decl.qualified_name)

    unresolved = []
    for row in model.proof_payload["operators"]:
        if row["proof_status"] != "lean_theorem":
            continue
        for module in row["lean_modules"]:
            assert module in by_module, module
        for theorem in row["lean_theorems"]:
            if theorem not in by_name:
                unresolved.append((row["operator"], theorem, row["lean_modules"]))
                continue
            assert any(theorem in by_module[module] for module in row["lean_modules"]), (
                row["operator"],
                theorem,
                row["lean_modules"],
            )
    assert not unresolved


def test_appendix_contains_live_fragment_and_proof_footprint_counts():
    model = fsa.build_model()
    tex = fsa.render_latex(model)
    assert model.grammar in tex
    for name, count in model.supported_counts.items():
        assert f"{count} {name}" in tex
    for status, count in model.proof_payload["summary"].items():
        assert status.replace("_", r"\_") in tex
        assert f"& {count} &" in tex
    assert str(model.proof_payload["total"]) in tex


def test_axiom_audit_entries_all_resolve_to_scanned_declarations():
    model = fsa.build_model()
    by_name = {d.qualified_name for d in model.declarations}
    missing = [name for name in model.audited_theorem_names if name not in by_name]
    assert not missing


def test_module_runs_as_script_and_pdf_artifact_is_present():
    out = subprocess.run(
        [sys.executable, "-m", "src.formal_soundness_appendix"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith(r"\documentclass")
    pdf_path = REPO / "formal_soundness_appendix.pdf"
    assert pdf_path.read_bytes().startswith(b"%PDF")
