"""Tests for the provenance-rich GitHub bug corpus expansion (Step 249)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments_v5.provenance_bug_corpus import build as corpus_build  # noqa: E402

CORPUS = REPO / "experiments_v5" / "provenance_bug_corpus" / "corpus.jsonl"
MANIFEST = REPO / "experiments_v5" / "provenance_bug_corpus" / "manifest.json"
MINED = REPO / "experiments_v5" / "github_bug_mining" / "mined_bugs_dataset.jsonl"


def _records():
    return [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]


def test_artifact_is_deterministic_and_large_enough():
    assert corpus_build.check() == 0
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["total"] >= 1000
    assert manifest["total"] == len(_records())
    assert manifest["source_dataset"]["records"] == 2704
    assert manifest["corpus_sha256"] == corpus_build._sha256_lines(_records())


def test_enriched_records_are_one_to_one_with_frozen_mined_dataset():
    base = {json.loads(line)["source_url"] for line in MINED.read_text().splitlines()}
    enriched = {record["source_url"] for record in _records()}
    assert enriched == base


def test_every_record_has_required_provenance_fields():
    required = {
        "id",
        "source_url",
        "repository",
        "owner",
        "repo_name",
        "github_kind",
        "github_number",
        "runtime_signature",
        "license_metadata",
        "commit_links",
        "commit_link_status",
        "reproducer",
        "redistribution",
    }
    for record in _records():
        assert required.issubset(record)
        assert record["source_url"].startswith("https://github.com/")
        assert record["github_kind"] in {"issue", "pull_request"}
        assert record["runtime_signature"]["matched_signature"]
        assert record["runtime_signature"]["source"] == "verbatim_pytorch_runtime_error_fragment"
        assert record["license_metadata"]["spdx_id"]
        assert record["license_metadata"]["third_party_code_redistributed"] is False
        assert record["redistribution"]["third_party_code_redistributed"] is False
        assert record["redistribution"]["stored_source_blob"] is False
        assert record["redistribution"]["stored_issue_body"] is False
        assert record["redistribution"]["stored_patch"] is False


def test_license_metadata_has_real_known_snapshot_not_only_placeholders():
    manifest = json.loads(MANIFEST.read_text())
    license_meta = manifest["license_metadata"]
    assert license_meta["known_records"] >= 200
    assert license_meta["known_repositories"] >= 10
    assert license_meta["noassertion_records"] > 0
    assert set(license_meta["by_spdx_id"]) >= {"Apache-2.0", "BSD-3-Clause", "MIT", "NOASSERTION"}


def test_commit_link_coverage_is_honest_and_measurable():
    manifest = json.loads(MANIFEST.read_text())
    coverage = manifest["commit_link_coverage"]
    statuses = coverage["by_status"]
    assert statuses["direct_pr_commits_page"] == manifest["by_github_kind"]["pull_request"]
    assert statuses["direct_pr_commits_page"] == 205
    assert coverage["records_with_any_commit_link"] >= statuses["direct_pr_commits_page"]
    assert "not_available_in_offline_snapshot" in statuses
    for record in _records():
        if record["github_kind"] == "pull_request":
            assert record["commit_links"]
            assert record["commit_links"][0]["url"].endswith("/commits")


def test_every_category_has_a_repo_authored_reproducer():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["reproducers"]["missing_categories"] == []
    paths = manifest["reproducers"]["paths"]
    assert set(paths) == set(manifest["by_category"])
    for category, rel_path in paths.items():
        path = REPO / rel_path
        assert path.exists(), category
        for record in _records():
            if record["category"] == category:
                repro = record["reproducer"]
                assert repro["path"] == rel_path
                assert repro["legally_redistributable"] is True
                assert repro["third_party_code_copied"] is False
                break


@pytest.mark.parametrize(
    "category,meta",
    sorted(corpus_build.REPRODUCERS.items()),
)
def test_minimized_reproducers_raise_real_pytorch_errors(category, meta):
    torch = pytest.importorskip("torch")
    if meta["requires_cuda"] and not torch.cuda.is_available():
        pytest.skip(f"{category} requires CUDA")
    path = REPO / meta["path"]
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert proc.returncode != 0
    assert meta["expected_error_substring"] in (proc.stdout + proc.stderr)


def test_artifact_does_not_store_third_party_code_blobs():
    forbidden = {"source_code", "patch", "diff", "issue_body", "body", "comments"}
    for record in _records():
        assert not (forbidden & set(record.keys()))
