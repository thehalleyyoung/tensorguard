"""Tests for Step-250 stratified precision/recall artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation import stratified_precision_recall as spr  # noqa: E402


def _committed():
    return json.loads(spr.OUT_JSON.read_text())


def test_artifact_regenerates_byte_identically():
    assert spr.run(check=True) == 0


def test_required_stratification_dimensions_are_present():
    artifact = _committed()
    assert artifact["dimensions"] == [
        "operator_family",
        "framework",
        "bug_class",
        "model_family",
        "source",
    ]
    for row in artifact["executable_rows"]:
        assert set(row["strata"]) == set(artifact["dimensions"])
        assert row["strata"]["framework"] == "pytorch_nn_module"


def test_tensorguard_overall_confusion_still_has_no_errors():
    tg = _committed()["overall_by_method"]["tensorguard"]
    assert (tg["TP"], tg["FP"], tg["TN"], tg["FN"], tg["NA"]) == (8, 0, 8, 0, 0)
    assert tg["metrics"]["precision"]["claimable"] is True
    assert tg["metrics"]["recall"]["claimable"] is True


def test_sparse_executable_strata_are_exploratory_not_claims():
    artifact = _committed()
    conv = artifact["executable_strata"]["tensorguard"]["operator_family"]["convolution"]
    assert conv["N"] == 2
    assert conv["publication_gate"]["status"] == "exploratory_only"
    assert conv["metrics"]["recall"]["claimable"] is False

    linear = artifact["executable_strata"]["tensorguard"]["bug_class"]["linear_inout_mismatch"]
    assert linear["positive_n"] == 3
    assert linear["metrics"]["recall"]["n"] == 3
    assert linear["metrics"]["recall"]["claimable"] is False


def test_framework_axis_is_honestly_degenerate():
    artifact = _committed()
    summary = artifact["executable_dimension_summary"]["framework"]
    assert summary["n_strata"] == 1
    assert summary["degenerate_axis"] is True
    assert any("framework axis is intentionally degenerate" in note
               for note in artifact["honesty_notes"])

    corpus_framework = artifact["provenance_positive_only_sample_sizes"]["framework"]
    assert corpus_framework["n_strata"] == 1
    assert corpus_framework["degenerate_axis"] is True
    assert corpus_framework["strata"]["pytorch"]["records"] == 2704


def test_wilson_intervals_contain_point_estimates_when_defined():
    artifact = _committed()
    for method in artifact["methods"]:
        for strata in artifact["executable_strata"][method].values():
            for row in strata.values():
                for metric in row["metrics"].values():
                    estimate = metric["estimate"]
                    ci = metric["ci95"]
                    if estimate is None:
                        assert ci is None
                    else:
                        assert 0.0 <= ci[0] <= estimate <= ci[1] <= 1.0


def test_positive_only_corpus_gates_use_stratum_counts_not_total_denominator():
    artifact = _committed()
    corpus = artifact["provenance_positive_only_sample_sizes"]
    assert artifact["inputs"]["provenance_positive_records"] == 2704

    rare = corpus["bug_class"]["strata"]["dtype_device_input_mismatch"]
    assert rare["records"] == 19
    assert rare["claimable"] is False
    assert rare["gate"] == "insufficient_n"

    common = corpus["operator_family"]["strata"]["matmul_linear"]
    assert common["records"] == 400
    assert common["claimable"] is True
    assert common["gate"] == "pass"
