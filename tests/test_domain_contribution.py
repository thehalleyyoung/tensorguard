"""Regression tests for the per-domain contribution corpus (100_STEPS.md Step 3).

These pin the honest per-domain story:
  * the base shape view refutes the shape bug on its own;
  * the device domain refutes >=1 bug the shape view misses;
  * the gradient domain refutes >=1 bug the shape view misses;
  * the phase domain contributes no refutations (diagnostic-only).

They run the curated corpus in experiments_v5/domain_corpus/ through
verify_architecture with one domain flag at a time, exactly mirroring
experiments_v5/run_domain_contribution.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.api import verify_architecture

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "experiments_v5" / "domain_corpus_manifest.json"

LEVEL_KWARGS = {
    "base": dict(check_devices=False, check_phases=False, check_gradients=False),
    "+device": dict(check_devices=True, check_phases=False, check_gradients=False),
    "+phase": dict(check_devices=False, check_phases=True, check_gradients=False),
    "+grad": dict(check_devices=False, check_phases=False, check_gradients=True),
}


def _load_manifest():
    return json.loads(MANIFEST.read_text())


def _read_shapes(src: str):
    m = re.search(r"^INPUT_SHAPES\s*=\s*(\{.*?\})", src, flags=re.MULTILINE | re.DOTALL)
    if not m:
        return {}
    return eval(m.group(1), {"__builtins__": {}}, {})


def _refuted(entry_id: str, level: str) -> bool:
    manifest = _load_manifest()
    entry = next(e for e in manifest["entries"] if e["id"] == entry_id)
    src = (REPO / entry["repro_file"]).read_text()
    shapes = _read_shapes(src)
    res = verify_architecture(
        src,
        input_shapes=shapes,
        filename="<domain-test>",
        high_confidence_only=False,
        max_cegar_iterations=0,
        **LEVEL_KWARGS[level],
    )
    return res.bug_count > 0


def test_manifest_well_formed():
    manifest = _load_manifest()
    ids = {e["id"] for e in manifest["entries"]}
    assert {"shape_01", "device_01", "device_02", "grad_01", "grad_02", "phase_01"} <= ids
    for e in manifest["entries"]:
        assert (REPO / e["repro_file"]).exists(), e["repro_file"]


def test_shape_bug_refuted_by_base_view():
    assert _refuted("shape_01", "base")


@pytest.mark.parametrize("eid", ["device_01", "device_02"])
def test_device_bug_missed_by_shape_but_caught_by_device(eid):
    assert not _refuted(eid, "base"), "device bug should NOT be caught by the shape view"
    assert _refuted(eid, "+device"), "device bug should be caught with check_devices"


@pytest.mark.parametrize("eid", ["grad_01", "grad_02"])
def test_grad_bug_missed_by_shape_but_caught_by_grad(eid):
    assert not _refuted(eid, "base"), "gradient bug should NOT be caught by the shape view"
    assert _refuted(eid, "+grad"), "gradient bug should be caught with check_gradients"


@pytest.mark.parametrize("eid", ["device_01", "device_02"])
def test_device_bug_not_caught_by_unrelated_domains(eid):
    assert not _refuted(eid, "+phase")
    assert not _refuted(eid, "+grad")


@pytest.mark.parametrize("eid", ["grad_01", "grad_02"])
def test_grad_bug_not_caught_by_unrelated_domains(eid):
    assert not _refuted(eid, "+phase")
    assert not _refuted(eid, "+device")


def test_phase_domain_is_diagnostic_only():
    """The phase domain must not flip any corpus entry to UNSAFE."""
    manifest = _load_manifest()
    for e in manifest["entries"]:
        base = _refuted(e["id"], "base")
        with_phase = _refuted(e["id"], "+phase")
        assert with_phase == base, (
            f"phase domain changed verdict for {e['id']} "
            f"(base={base}, +phase={with_phase}); it is documented diagnostic-only"
        )
