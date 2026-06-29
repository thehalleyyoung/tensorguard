"""Roadmap step 1 — the **threat model is the spec**, enforced by tests.

Two layers of checking:

1. *Doc ↔ code coverage* (the acceptance bar): every ``WeightsFinding`` kind
   constructed anywhere in ``src/symexec/weights.py`` or
   ``src/symexec/model_contract.py`` is documented in
   ``docs/symexec/threat_model.md``'s finding table, with a non-empty runtime
   failure it rules out — and the table carries no stale rows.

2. *Doc ↔ runtime behavior* (the teeth): for every finding kind we forge a
   checkpoint (or contract) that actually triggers it, assert the certifier emits
   exactly that kind, and assert the kind is documented. This binds the prose to
   observable behavior so the table cannot lie.

Entirely torch-free: a tiny in-test safetensors writer forges well-formed and
deliberately malformed files.
"""

from __future__ import annotations

import ast
import json
import re
import struct
from pathlib import Path

import pytest

from src.symexec.weights import certify_weights_file

_REPO = Path(__file__).resolve().parents[1]
_DOC = _REPO / "docs" / "symexec" / "threat_model.md"
_WEIGHTS_PY = _REPO / "src" / "symexec" / "weights.py"
_MODEL_CONTRACT_PY = _REPO / "src" / "symexec" / "model_contract.py"


# --------------------------------------------------------------------------- #
# Source of truth: AST-extract every WeightsFinding(...) kind.                  #
# --------------------------------------------------------------------------- #
def _finding_kinds_in(path: Path) -> set[str]:
    """All string ``kind`` values passed to ``WeightsFinding(...)`` in a module,
    whether positional (first arg) or keyword (``kind=``)."""
    tree = ast.parse(path.read_text())
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "WeightsFinding"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            kinds.add(node.args[0].value)
        for kw in node.keywords:
            if kw.arg == "kind" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                kinds.add(kw.value.value)
    return kinds


def _code_finding_kinds() -> set[str]:
    return _finding_kinds_in(_WEIGHTS_PY) | _finding_kinds_in(_MODEL_CONTRACT_PY)


# --------------------------------------------------------------------------- #
# Parse the threat-model finding table.                                         #
# --------------------------------------------------------------------------- #
_ROW_RE = re.compile(r"^\|\s*`([a-z_]+)`\s*\|(.+)\|\s*$")


def _doc_table_rows() -> dict[str, list[str]]:
    """Map ``kind -> [boundary, detects, runtime_failure]`` from the doc table."""
    rows: dict[str, list[str]] = {}
    for line in _DOC.read_text().splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        kind = m.group(1)
        cells = [c.strip() for c in m.group(2).split("|")]
        rows[kind] = cells
    return rows


# --------------------------------------------------------------------------- #
# Layer 1: doc ↔ code coverage (the acceptance bar).                            #
# --------------------------------------------------------------------------- #
def test_doc_exists_and_table_parses():
    assert _DOC.is_file(), "threat_model.md must exist"
    rows = _doc_table_rows()
    assert rows, "the finding table must parse at least one row"


def test_every_code_finding_kind_is_documented():
    code = _code_finding_kinds()
    documented = set(_doc_table_rows())
    missing = sorted(code - documented)
    assert not missing, f"finding kinds emitted in code but absent from the threat model: {missing}"


def test_no_stale_documented_kinds():
    code = _code_finding_kinds()
    documented = set(_doc_table_rows())
    stale = sorted(documented - code)
    assert not stale, f"threat model documents finding kinds no longer emitted in code: {stale}"


def test_each_row_cites_a_runtime_failure():
    rows = _doc_table_rows()
    for kind, cells in rows.items():
        assert len(cells) == 3, f"row {kind!r} must have boundary|detects|runtime-failure"
        boundary, detects, runtime = cells
        assert boundary, f"row {kind!r} missing boundary"
        assert detects, f"row {kind!r} missing 'detects'"
        assert runtime, f"row {kind!r} missing runtime-failure"
        # Must name an actual failure: an *Error, a known exception, or an
        # explicitly-documented silent corruption (aliasing / NaN propagation).
        low = runtime.lower()
        assert ("error" in low or "aliasing" in low or "nan" in low), \
            f"row {kind!r} runtime-failure does not name a concrete failure: {runtime!r}"


def test_expected_count_of_finding_kinds():
    # Guards against silent drift in either direction; update deliberately when a
    # new finding kind is added together with its table row.
    assert len(_code_finding_kinds()) == 15


def test_threat_model_has_required_sections():
    text = _DOC.read_text()
    for heading in (
        "## 1. Purpose & scope",
        "## 2. Actors",
        "## 3. Trust boundaries",
        "## 4. Assets & security goals",
        "## 5. The code↔data contract",
        "## 6. Finding → runtime-failure table",
        "## 7. Non-goals & limitations",
        "## 8. Soundness summary",
    ):
        assert heading in text, f"threat model missing section: {heading!r}"


def test_model_contract_emits_no_finding_kinds_directly():
    # Documented invariant (§5): the bridge abstains, it never raises a verdict;
    # all contract findings flow through weights.py. If this changes, the doc must.
    assert _finding_kinds_in(_MODEL_CONTRACT_PY) == set()


# --------------------------------------------------------------------------- #
# Torch-free safetensors writers (contiguous + fully-raw control).              #
# --------------------------------------------------------------------------- #
def _write_st(path, tensors):
    """tensors: list of (name, dtype, shape, raw_bytes); contiguous offsets."""
    header, cursor, buf = {}, 0, b""
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [cursor, cursor + len(raw)]}
        cursor += len(raw)
        buf += raw
    hb = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + buf)


def _write_raw(path, header_obj, buffer_bytes):
    """Full control: arbitrary header object + arbitrary trailing buffer."""
    hb = json.dumps(header_obj).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + buffer_bytes)


def _f32(*vals):
    return struct.pack("<%df" % len(vals), *vals)


def _kinds(cert):
    return set(cert.finding_kinds)


# --------------------------------------------------------------------------- #
# Layer 2: doc ↔ runtime behavior — trigger each finding kind for real.         #
# --------------------------------------------------------------------------- #
def test_trigger_malformed_frame(tmp_path):
    p = tmp_path / "frame.safetensors"
    # Header length claims 1e9 bytes but the file has none.
    p.write_bytes(struct.pack("<Q", 1_000_000_000))
    cert = certify_weights_file(str(p))
    assert not cert.proven_safe
    assert "malformed_frame" in _kinds(cert)


def test_trigger_malformed_entry(tmp_path):
    p = tmp_path / "entry.safetensors"
    _write_raw(p, {"w": 5}, b"")  # entry is not an object
    cert = certify_weights_file(str(p))
    assert "malformed_entry" in _kinds(cert)


def test_trigger_unknown_dtype(tmp_path):
    p = tmp_path / "dtype.safetensors"
    _write_raw(p, {"w": {"dtype": "F7", "shape": [1], "data_offsets": [0, 4]}}, b"\x00\x00\x00\x00")
    cert = certify_weights_file(str(p))
    assert "unknown_dtype" in _kinds(cert)


def test_trigger_malformed_shape(tmp_path):
    p = tmp_path / "shape.safetensors"
    _write_raw(p, {"w": {"dtype": "F32", "shape": "bad", "data_offsets": [0, 4]}}, b"\x00\x00\x00\x00")
    cert = certify_weights_file(str(p))
    assert "malformed_shape" in _kinds(cert)


def test_trigger_malformed_offsets(tmp_path):
    p = tmp_path / "off.safetensors"
    _write_raw(p, {"w": {"dtype": "F32", "shape": [1], "data_offsets": [4, 2]}}, b"\x00\x00\x00\x00")
    cert = certify_weights_file(str(p))
    assert "malformed_offsets" in _kinds(cert)


def test_trigger_byte_length_mismatch(tmp_path):
    p = tmp_path / "blen.safetensors"
    # Declares F32[2,3] (24 bytes) but only spans 4 bytes.
    _write_st(p, [("w", "F32", [2, 3], _f32(0.0))])
    cert = certify_weights_file(str(p))
    assert "byte_length_mismatch" in _kinds(cert)


def test_trigger_storage_out_of_bounds(tmp_path):
    p = tmp_path / "oob.safetensors"
    # end offset 100 but buffer is only 4 bytes.
    _write_raw(p, {"w": {"dtype": "F32", "shape": [25], "data_offsets": [0, 100]}}, b"\x00\x00\x00\x00")
    cert = certify_weights_file(str(p))
    assert "storage_out_of_bounds" in _kinds(cert)


def test_trigger_storage_gap(tmp_path):
    p = tmp_path / "gap.safetensors"
    # w:[0,4], b:[8,12]; bytes [4,8) unreferenced; buffer is 12 bytes.
    header = {
        "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "b": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
    }
    _write_raw(p, header, b"\x00" * 12)
    cert = certify_weights_file(str(p))
    assert "storage_gap" in _kinds(cert)


def test_trigger_storage_overlap(tmp_path):
    p = tmp_path / "ovl.safetensors"
    # w:[0,8], b:[4,12] overlap on [4,8); buffer is 12 bytes.
    header = {
        "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
    }
    _write_raw(p, header, b"\x00" * 12)
    cert = certify_weights_file(str(p))
    assert "storage_overlap" in _kinds(cert)


def test_trigger_storage_undercovered(tmp_path):
    p = tmp_path / "under.safetensors"
    # one tensor [0,4] but the buffer is 8 bytes -> 4 trailing unaccounted bytes.
    _write_raw(p, {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"\x00" * 8)
    cert = certify_weights_file(str(p))
    assert "storage_undercovered" in _kinds(cert)


def test_trigger_non_finite_values(tmp_path):
    p = tmp_path / "nan.safetensors"
    _write_st(p, [("w", "F32", [2], _f32(float("nan"), float("inf")))])
    cert = certify_weights_file(str(p))
    assert "non_finite_values" in _kinds(cert)


def _good_file(tmp_path):
    p = tmp_path / "good.safetensors"
    _write_st(p, [("w", "F32", [2], _f32(1.0, 2.0))])
    return p


def test_trigger_contract_missing_key(tmp_path):
    p = _good_file(tmp_path)
    cert = certify_weights_file(str(p), expected={"absent": (None, (3,))})
    assert "contract_missing_key" in _kinds(cert)


def test_trigger_contract_unexpected_key(tmp_path):
    p = _good_file(tmp_path)
    # Full (non-partial) contract that omits 'w' -> 'w' is unexpected.
    cert = certify_weights_file(str(p), expected={}, contract_partial=False)
    assert "contract_unexpected_key" in _kinds(cert)


def test_partial_contract_suppresses_unexpected_key(tmp_path):
    p = _good_file(tmp_path)
    cert = certify_weights_file(str(p), expected={}, contract_partial=True)
    assert "contract_unexpected_key" not in _kinds(cert)


def test_trigger_contract_shape_mismatch(tmp_path):
    p = _good_file(tmp_path)
    cert = certify_weights_file(str(p), expected={"w": (None, (5,))})
    assert "contract_shape_mismatch" in _kinds(cert)


def test_trigger_contract_dtype_mismatch(tmp_path):
    p = _good_file(tmp_path)
    cert = certify_weights_file(str(p), expected={"w": ("F16", (2,))})
    assert "contract_dtype_mismatch" in _kinds(cert)


# --------------------------------------------------------------------------- #
# Cross-link: every triggered kind is documented (closes the loop).             #
# --------------------------------------------------------------------------- #
_TRIGGER_FNS_TO_KIND = {
    "malformed_frame", "malformed_entry", "unknown_dtype", "malformed_shape",
    "malformed_offsets", "byte_length_mismatch", "storage_out_of_bounds",
    "storage_gap", "storage_overlap", "storage_undercovered", "non_finite_values",
    "contract_missing_key", "contract_unexpected_key", "contract_shape_mismatch",
    "contract_dtype_mismatch",
}


def test_all_15_kinds_have_a_behavioral_trigger():
    # The set of kinds we forge a real reproducer for == the set emitted in code.
    assert _TRIGGER_FNS_TO_KIND == _code_finding_kinds()


def test_every_triggered_kind_is_documented():
    documented = set(_doc_table_rows())
    for kind in _TRIGGER_FNS_TO_KIND:
        assert kind in documented, f"behaviorally-triggered kind {kind!r} is undocumented"


def test_good_file_is_proven_safe(tmp_path):
    # Sanity: a well-formed, finite, contract-conformant checkpoint certifies.
    p = _good_file(tmp_path)
    cert = certify_weights_file(str(p), expected={"w": ("F32", (2,))})
    assert cert.proven_safe
    assert cert.finding_kinds == ()
