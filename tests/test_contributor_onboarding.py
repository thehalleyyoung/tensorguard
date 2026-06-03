"""Step 287 -- generated contributor onboarding is backed by real repo surfaces."""

from __future__ import annotations

import json
import subprocess
import sys

import reproducibility.contributor_onboarding as co


def test_good_first_issues_are_derived_from_low_confidence_registry_entries():
    issues = co.build_issues()
    assert len(issues) >= 8
    assert any(issue.confidence == "heuristic" for issue in issues)
    assert any(issue.proof_status in {"heuristic", "tested_only_rule"} for issue in issues)
    for issue in issues:
        assert "good first issue" in issue.labels
        assert "operator-coverage" in issue.labels
        assert "docs/contributing/operator_template.py" in issue.acceptance_tests
        assert issue.evidence


def test_issue_payloads_are_copyable_github_issues():
    payload = json.loads(co.OUT_ISSUES_JSON.read_text(encoding="utf-8"))
    assert payload["schema"] == "tensorguard.good_first_operators/v1"
    assert payload["summary"]["issue_count"] == len(payload["issues"])
    for issue in payload["issues"]:
        assert issue["title"].startswith("Good first operator:")
        assert "### Goal" in issue["body"]
        assert "### Acceptance checklist" in issue["body"]
        assert issue["labels"]
        assert all((co.REPO / path).exists() for path in issue["evidence"])


def test_operator_template_is_not_pytest_collected_but_contains_real_patterns():
    assert co.OUT_TEMPLATE.name == "operator_template.py"
    text = co.OUT_TEMPLATE.read_text(encoding="utf-8")
    assert "verify_architecture" in text
    assert "test_operator_transfer_accepts_real_valid_case" in text
    assert "test_operator_transfer_refutes_real_invalid_case" in text
    assert "pytest will not collect it directly" in text


def test_stub_guide_points_to_safe_declarative_and_trusted_plugin_paths():
    text = co.OUT_STUB_GUIDE.read_text(encoding="utf-8")
    for path in (
        "community_stubs/README.md",
        "src.operator_plugin_abi.OperatorTheoryContract",
        "docs/plugins/operator_plugin_abi.md",
        "docs/plugins/third_party_conformance.md",
    ):
        assert path in text


def test_lean_examples_resolve_against_committed_lean_tree():
    examples = co.build_lean_examples()
    assert len(examples) >= 5
    for example in examples:
        assert (co.REPO / example["path"]).exists(), example
        assert example["theorem_resolves"], example
        assert example["short_theorem"] in example["theorem"]


def test_audit_covers_paths_outputs_and_theorem_resolution():
    audit = co.build_audit()
    assert audit["summary"]["issue_count"] >= 8
    assert audit["summary"]["output_count"] == len(co.OUTPUTS)
    assert audit["summary"]["all_cited_paths_exist"] is True
    assert audit["summary"]["lean_examples_resolve"] is True
    assert all(audit["cited_paths"].values())


def test_generated_artifacts_are_byte_deterministic():
    first = co.write_outputs()
    snapshots = {path: path.read_text(encoding="utf-8") for path in co.OUTPUTS}
    second = co.write_outputs()
    assert first == second
    assert {path: path.read_text(encoding="utf-8") for path in co.OUTPUTS} == snapshots


def test_cli_check_passes_against_committed_artifacts():
    proc = subprocess.run(
        [sys.executable, "reproducibility/contributor_onboarding.py", "--check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
