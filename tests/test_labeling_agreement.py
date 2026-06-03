"""Tests for the Step-255 mined-bug labeling agreement artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments_v5.labeling_agreement import agreement  # noqa: E402


ARTIFACT = REPO / "experiments_v5" / "labeling_agreement" / "agreement.json"
LOG = REPO / "experiments_v5" / "labeling_agreement" / "adjudication_log.md"
RUBRIC = REPO / "experiments_v5" / "labeling_agreement" / "rubric.md"
TAXONOMY_MD = REPO / "experiments_v5" / "labeling_agreement" / "ambiguity_taxonomy.md"
PAPER = REPO / "tool_paper.tex"


def test_agreement_artifacts_are_deterministic():
    assert agreement.check() == 0
    built, log = agreement.build()
    committed = json.loads(ARTIFACT.read_text())
    assert committed == built
    assert LOG.read_text() == log


def test_sample_is_stratified_from_frozen_provenance_corpus():
    corpus, rows, manifest, _taxonomy = agreement.load_and_validate()
    by_id = {record["id"]: record for record in corpus}
    assert manifest["total"] == 2704
    assert len(rows) == 32
    assert [row["record_id"] for row in rows] == agreement.expected_sample_ids(corpus)
    assert len({row["record_id"] for row in rows}) == len(rows)
    by_category = {}
    for row in rows:
        category = by_id[row["record_id"]]["category"]
        by_category[category] = by_category.get(category, 0) + 1
    assert set(by_category.values()) == {4}
    assert set(by_category) == set(manifest["by_category"])


def test_adjudication_log_covers_exactly_disagreements_and_taxonomy():
    _corpus, rows, _manifest, taxonomy = agreement.load_and_validate()
    artifact = json.loads(ARTIFACT.read_text())
    disagreements = [row for row in rows if row["adjudication"]["disagreement_axes"]]
    assert artifact["adjudication"]["records_with_any_disagreement"] == len(disagreements)
    assert len(disagreements) >= 10
    log = LOG.read_text()
    for row in disagreements:
        assert row["record_id"] in log
        assert row["adjudication"]["rationale"] in log
    for code in taxonomy["codes"]:
        assert f"`{code}`" in TAXONOMY_MD.read_text()
        assert f"`{code}`" in log


def test_metrics_are_meaningful_judgment_axes_not_mechanical_category_claims():
    artifact = json.loads(ARTIFACT.read_text())
    provenance = artifact["annotation_provenance"]
    assert provenance["not_human_subjects_study"] is True
    assert provenance["external_independent_annotators_claimed"] is False
    assert "mechanical runtime-signature category labels are not" in provenance["measured_object"]
    include = artifact["metrics"]["include_decision"]
    root = artifact["metrics"]["root_cause_family"]
    evidence = artifact["metrics"]["evidence_strength"]
    assert 0.0 < include["cohen_kappa"] < 1.0
    assert 0.0 < root["cohen_kappa"] < 1.0
    assert 0.0 < evidence["cohen_kappa"] < 1.0
    assert include["disagreements"] > 0
    assert root["disagreements"] > 0
    assert evidence["disagreements"] > 0


def test_cohen_kappa_handles_degenerate_marginals():
    pairs = [("include", "include"), ("include", "include")]
    assert agreement.cohen_kappa(pairs, agreement.AXES["include_decision"]) is None
    mixed = [("include", "include"), ("exclude", "defer"), ("defer", "exclude")]
    kappa = agreement.cohen_kappa(mixed, agreement.AXES["include_decision"])
    assert kappa is not None
    assert -1.0 <= kappa <= 1.0


def test_public_docs_and_paper_state_the_scope_honestly():
    rubric = RUBRIC.read_text()
    paper = PAPER.read_text()
    assert "not a human-subjects study" in rubric
    assert "mechanical category" in rubric
    assert "dual-pass" in paper
    assert "not a human-subjects study" in paper
    assert "agreement.json" in paper
