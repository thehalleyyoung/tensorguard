"""Regression tests for the per-operator proof-footprint manifest."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from src.graph_compiler import _UNIVERSAL_TRANSFER_REGISTRY
from src.confidence_tags import ConfidenceTag
from src.proof_footprint import ProofStatus, footprint_for, proof_footprint_table, to_json


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LEAN = os.path.join(_REPO, "lean")


def _load_committed():
    with open(os.path.join(_REPO, "proof_footprint_manifest.json")) as fh:
        return json.load(fh)


def _module_path(module: str) -> str:
    parts = module.split(".")
    assert parts[0] == "TensorGuard"
    return os.path.join(_LEAN, *parts) + ".lean"


def _declared_in(module: str, theorem: str) -> bool:
    base = theorem.rsplit(".", 1)[-1]
    with open(_module_path(module)) as fh:
        src = fh.read()
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return re.search(rf"\b(?:theorem|lemma)\s+{re.escape(base)}\b", src) is not None


def test_manifest_covers_exact_live_registry():
    rows = proof_footprint_table()
    assert {row["operator"] for row in rows} == set(_UNIVERSAL_TRANSFER_REGISTRY)
    assert [row["operator"] for row in rows] == sorted(_UNIVERSAL_TRANSFER_REGISTRY)


def test_manifest_rows_are_well_formed():
    valid_statuses = {status.value for status in ProofStatus}
    valid_confidence = {tag.value for tag in ConfidenceTag}
    rows = proof_footprint_table()
    for row in rows:
        assert set(row) == {
            "operator",
            "proof_status",
            "confidence",
            "confidence_rationale",
            "rule",
            "lean_modules",
            "lean_theorems",
            "evidence",
            "rationale",
        }
        assert row["proof_status"] in valid_statuses
        assert row["confidence"] in valid_confidence
        assert row["confidence_rationale"]
        assert row["rule"]
        assert row["rationale"]
        assert row["evidence"]
        for path in row["evidence"]:
            assert os.path.exists(os.path.join(_REPO, path)), (row["operator"], path)


def test_committed_manifest_is_in_sync_with_generator():
    proc = subprocess.run(
        [sys.executable, "-m", "src.proof_footprint"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert _load_committed() == json.loads(proc.stdout)
    assert json.loads(to_json()) == json.loads(proc.stdout)


def test_summary_counts_match_rows():
    payload = _load_committed()
    assert payload["schema"] == "tensorguard.proof_footprint/v1"
    assert payload["total"] == len(_UNIVERSAL_TRANSFER_REGISTRY)
    observed = {status.value: 0 for status in ProofStatus}
    for row in payload["operators"]:
        observed[row["proof_status"]] += 1
    assert payload["summary"] == observed


def test_representative_classifications_are_independent_ground_truth():
    expected = {
        "F.relu": ProofStatus.LEAN_THEOREM,
        "torch.relu": ProofStatus.LEAN_THEOREM,
        "torch.matmul": ProofStatus.LEAN_THEOREM,
        "torch.mm": ProofStatus.LEAN_THEOREM,
        "torch.bmm": ProofStatus.LEAN_THEOREM,
        "torch.outer": ProofStatus.PEN_AND_PAPER_RULE,
        "torch.kron": ProofStatus.PEN_AND_PAPER_RULE,
        "torch.sum": ProofStatus.LEAN_THEOREM,
        "torch.prod": ProofStatus.PEN_AND_PAPER_RULE,
        "torch.gather": ProofStatus.LEAN_THEOREM,
        "torch.stack": ProofStatus.LEAN_THEOREM,
        "torch.linalg.svd": ProofStatus.TESTED_ONLY_RULE,
        "torch.fft.rfft": ProofStatus.TESTED_ONLY_RULE,
        "torch.einsum": ProofStatus.HEURISTIC,
        "torch.unique": ProofStatus.HEURISTIC,
        "torch.multinomial": ProofStatus.HEURISTIC,
    }
    for op, status in expected.items():
        assert footprint_for(op)["proof_status"] == status.value


def test_lean_rows_name_real_imported_theorems():
    imports = open(os.path.join(_LEAN, "TensorGuard.lean")).read()
    for row in proof_footprint_table():
        if row["proof_status"] != ProofStatus.LEAN_THEOREM.value:
            assert row["lean_modules"] == []
            assert row["lean_theorems"] == []
            continue
        assert row["confidence"] != ConfidenceTag.HEURISTIC.value
        assert row["lean_modules"]
        assert row["lean_theorems"]
        for module in row["lean_modules"]:
            assert os.path.exists(_module_path(module)), module
            assert f"import {module}" in imports, module
        for theorem in row["lean_theorems"]:
            assert any(_declared_in(module, theorem) for module in row["lean_modules"]), (
                row["operator"],
                theorem,
                row["lean_modules"],
            )


def test_heuristic_confidence_rows_are_heuristic_footprints():
    rows = {row["operator"]: row for row in proof_footprint_table()}
    heuristic_rows = {
        name
        for name, row in rows.items()
        if row["confidence"] == ConfidenceTag.HEURISTIC.value
    }
    assert heuristic_rows == {"torch.einsum", "torch.unique", "torch.multinomial"}
    for name in heuristic_rows:
        assert rows[name]["proof_status"] == ProofStatus.HEURISTIC.value
