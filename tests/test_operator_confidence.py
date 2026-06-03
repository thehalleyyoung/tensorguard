"""Regression tests for per-operator confidence tags (Step 6)."""

import json
import os
import subprocess
import sys

import pytest

from src.graph_compiler import _UNIVERSAL_TRANSFER_REGISTRY, get_transfer
from src.operator_confidence import (
    ConfidenceTag,
    annotate_registry,
    confidence_table,
    rationale_for,
    tag_for,
    to_json,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_registered_op_has_a_tag():
    """Every operator with a transfer function classifies to a known tag."""
    valid = {t.value for t in ConfidenceTag}
    for name in _UNIVERSAL_TRANSFER_REGISTRY:
        assert tag_for(name).value in valid, name


def test_unknown_op_defaults_to_heuristic():
    """Unclassified / unregistered operators are honestly conservative."""
    assert tag_for("torch.this_op_does_not_exist") is ConfidenceTag.HEURISTIC
    assert tag_for("F.made_up") is ConfidenceTag.HEURISTIC


def test_known_tags_are_defensible():
    """Spot-check representative operators across all three tiers."""
    # complete: shape-preserving pointwise families
    assert tag_for("F.relu") is ConfidenceTag.COMPLETE
    assert tag_for("torch.sigmoid") is ConfidenceTag.COMPLETE
    assert tag_for("torch.abs") is ConfidenceTag.COMPLETE
    assert tag_for("torch.eq") is ConfidenceTag.COMPLETE
    # sound: exact structural rules
    assert tag_for("torch.matmul") is ConfidenceTag.SOUND
    assert tag_for("torch.bmm") is ConfidenceTag.SOUND
    assert tag_for("torch.sum") is ConfidenceTag.SOUND
    assert tag_for("torch.fft.rfft") is ConfidenceTag.SOUND
    assert tag_for("torch.linalg.svd") is ConfidenceTag.SOUND
    # heuristic: data-dependent / approximated
    assert tag_for("torch.unique") is ConfidenceTag.HEURISTIC
    assert tag_for("torch.multinomial") is ConfidenceTag.HEURISTIC
    assert tag_for("torch.einsum") is ConfidenceTag.HEURISTIC
    assert tag_for("torch.linalg.lstsq") is ConfidenceTag.HEURISTIC


def test_every_op_has_a_nonempty_rationale():
    for name in _UNIVERSAL_TRANSFER_REGISTRY:
        assert rationale_for(name).strip(), name


def test_confidence_table_covers_full_registry():
    table = confidence_table()
    table_ops = {row["operator"] for row in table}
    assert table_ops == set(_UNIVERSAL_TRANSFER_REGISTRY.keys())
    # sorted and well-formed
    assert [r["operator"] for r in table] == sorted(table_ops)
    for row in table:
        assert set(row.keys()) == {"operator", "confidence", "rationale"}


def test_to_json_is_machine_readable_and_consistent():
    payload = json.loads(to_json())
    assert payload["schema"] == "tensorguard.operator_confidence/v1"
    assert payload["default_tag"] == "heuristic"
    assert payload["total"] == len(_UNIVERSAL_TRANSFER_REGISTRY)
    assert sum(payload["summary"].values()) == payload["total"]


def test_committed_table_in_sync_with_code():
    """The committed JSON artifact must match freshly generated output.

    Generated via a subprocess so the comparison reflects the canonical
    import-time registry (other tests may register extra ops into the global
    registry within this process).
    """
    path = os.path.join(_REPO, "operator_confidence_table.json")
    assert os.path.exists(path), "operator_confidence_table.json missing"
    with open(path) as f:
        committed = json.load(f)
    proc = subprocess.run(
        [sys.executable, "-m", "src.operator_confidence"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    fresh = json.loads(proc.stdout)
    assert committed == fresh, (
        "operator_confidence_table.json is stale; regenerate with "
        "`python -m src.operator_confidence > operator_confidence_table.json`"
    )


def test_annotate_registry_stamps_transfer_functions():
    n = annotate_registry()
    assert n == len(_UNIVERSAL_TRANSFER_REGISTRY)
    # the tag travels with the transfer function so consumers can surface it
    assert get_transfer("torch.matmul").confidence == "sound"
    assert get_transfer("F.relu").confidence == "complete"
    assert get_transfer("torch.unique").confidence == "heuristic"


def test_cli_operator_confidence_command():
    env = dict(os.environ)
    out = subprocess.run(
        [sys.executable, "-m", "src.cli.main", "operator-confidence", "--json"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    # The subprocess sees the canonical import-time registry; assert it matches
    # the committed artifact rather than this process's (possibly-mutated) one.
    with open(os.path.join(_REPO, "operator_confidence_table.json")) as f:
        committed = json.load(f)
    assert payload["total"] == committed["total"]
    assert payload["summary"] == committed["summary"]


def test_cli_operator_confidence_query():
    out = subprocess.run(
        [
            sys.executable, "-m", "src.cli.main", "operator-confidence",
            "torch.matmul", "torch.unique", "--json",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    rows = {r["operator"]: r["confidence"] for r in payload["operators"]}
    assert rows["torch.matmul"] == "sound"
    assert rows["torch.unique"] == "heuristic"
