"""Step 284 — third-party acceptance governance with real gates."""

from __future__ import annotations

import json
import subprocess
import sys

from reproducibility import governance_acceptance as gov


def test_policy_covers_all_submission_classes():
    classes = {cls.key: cls for cls in gov.policy()}
    assert set(classes) == {
        "verifier_backends",
        "stubs_and_plugins",
        "corpora",
        "benchmark_submissions",
    }
    for cls in classes.values():
        assert cls.gates
        assert any(g.kind == "automated" for g in cls.gates)


def test_automated_and_judgment_gates_are_distinguished():
    audit = gov.build_audit()
    assert audit["summary"]["automated_gate_count"] >= 8
    assert audit["summary"]["maintainer_judgment_gate_count"] >= 3
    for cls in audit["submission_classes"]:
        for gate in cls["gates"]:
            assert gate["kind"] in {"automated", "maintainer-judgment"}
            if gate["kind"] == "automated":
                assert gate["commands"] or gate["symbols"]


def test_every_referenced_path_command_and_symbol_resolves():
    audit = gov.build_audit()
    assert audit["summary"]["all_references_resolve"], audit["summary"]
    for cls in audit["submission_classes"]:
        assert all(cls["required_files"].values()), cls["required_files"]
        for gate in cls["gates"]:
            assert all(gate["evidence"].values()), gate
            assert all(gate["commands"].values()), gate
            assert all(gate["symbols"].values()), gate


def test_checklists_include_required_review_topics():
    text = gov.render_policy_doc(gov.build_audit())
    for phrase in (
        "Cross-backend concordance",
        "security review attestations",
        "Runtime ground truth and hash freeze",
        "Redistribution provenance audit",
        "Signed raw verdicts only",
        "Anti-overfitting disclosure",
        "maintainer-judgment",
    ):
        assert phrase in text


def test_generated_artifacts_are_byte_deterministic(tmp_path):
    first = gov.write_outputs()
    json_1 = gov.OUT_JSON.read_text(encoding="utf-8")
    md_1 = gov.OUT_MD.read_text(encoding="utf-8")
    doc_1 = gov.OUT_DOC.read_text(encoding="utf-8")
    second = gov.write_outputs()
    assert first == second
    assert gov.OUT_JSON.read_text(encoding="utf-8") == json_1
    assert gov.OUT_MD.read_text(encoding="utf-8") == md_1
    assert gov.OUT_DOC.read_text(encoding="utf-8") == doc_1


def test_cli_check_passes_against_committed_artifacts():
    proc = subprocess.run(
        [sys.executable, "reproducibility/governance_acceptance.py", "--check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(gov.OUT_JSON.read_text(encoding="utf-8"))
    assert payload["summary"]["all_references_resolve"] is True
