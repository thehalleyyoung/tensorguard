from __future__ import annotations

from pathlib import Path

import reproducibility.camera_ready_paper as crp


def test_claim_sources_are_all_indexed_evidence_artifacts():
    data = crp.measure()
    assert data["all_claim_artifacts_indexed"]
    assert data["missing_from_evidence_index"] == []
    assert data["n_claims"] >= 8


def test_generated_claim_values_match_current_artifacts():
    data = crp.measure()
    by_id = {claim["id"]: claim["value"] for claim in data["claims"]}
    assert by_id["extended_corpus_score"] == {
        "tp": 153,
        "fp": 0,
        "tn": 74,
        "fn": 0,
        "n_total": 227,
    }
    assert by_id["differential_dispatcher"] == {
        "n_modules": 2000,
        "false_alarms": 0,
        "soundness_violations": 0,
    }
    assert by_id["fresh_machine_package"] == {
        "n_modes": 3,
        "all_modes_passed": True,
    }


def test_canonical_tool_paper_contains_generated_ledger():
    data = crp.measure()
    assert crp.validate_latex_block(data) == []
    assert crp.BEGIN in crp.render_latex_block(data)
    assert crp.END in crp.render_latex_block(data)


def test_stale_or_missing_ledger_is_rejected(tmp_path: Path):
    stale = tmp_path / "paper.tex"
    stale.write_text(r"\documentclass{article}\begin{document}no ledger\end{document}")
    errors = crp.validate_latex_block(crp.measure(), stale)
    assert errors
    assert "missing or stale" in errors[0]


def test_pdfinfo_page_count_parser():
    assert crp._page_count_from_pdfinfo_output("Title: x\nPages:          23\n") == 23
    assert crp._page_count_from_pdfinfo_output("Title: x\n") is None


def test_check_mode_is_byte_identical():
    assert crp.main(["--check"]) == 0
