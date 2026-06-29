"""Tests for the **weights-safety certifier** (even_more.md quantum leap): certified
well-formedness of a transformer's weights layer — data (the safetensors file) and
the code↔data contract — entirely torch-free.

A tiny in-test safetensors *writer* lets us forge both well-formed and deliberately
malformed checkpoints without depending on torch or the safetensors package."""

from __future__ import annotations

import array
import io
import json
import struct

import pytest

from src.symexec import (
    certify_weights_file,
    dumps_weights_certificate,
    loads_weights_certificate,
    render_weights_certificate,
    verify_weights_certificate,
    weights_certificate_from_dict,
    weights_certificate_to_dict,
    weights_contract_from_file,
)
from src.symexec.certify import main


# --------------------------------------------------------------------------- #
# Torch-free safetensors writer (faithful to the format).                       #
# --------------------------------------------------------------------------- #
def _write_st(path, tensors, *, raw_header=None):
    """tensors: list of (name, dtype, shape, raw_bytes). Offsets are packed
    contiguously unless ``raw_header`` overrides the JSON header verbatim."""
    header = {}
    cursor = 0
    buf = b""
    for name, dtype, shape, raw in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        cursor += len(raw)
        buf += raw
    hb = (raw_header if raw_header is not None
          else json.dumps(header).encode("utf-8"))
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(buf)


def _f32(n):
    return array.array("f", [0.0] * n).tobytes()


def _f16(n):
    return b"\x00\x00" * n


@pytest.fixture()
def good(tmp_path):
    p = tmp_path / "good.safetensors"
    _write_st(p, [("w", "F32", [2, 3], _f32(6)), ("b", "F16", [4], _f16(4))])
    return p


# --------------------------------------------------------------------------- #
# Data layer: well-formedness.                                                  #
# --------------------------------------------------------------------------- #
def test_well_formed_checkpoint_is_certified(good):
    cert = certify_weights_file(str(good))
    assert cert.proven_safe
    assert cert.num_tensors == 2
    assert cert.format == "safetensors"
    assert verify_weights_certificate(cert, str(good)).verified


def test_byte_length_mismatch_is_caught(tmp_path):
    p = tmp_path / "badlen.safetensors"
    _write_st(p, [("w", "F32", [2, 3], b"\x00" * 20)])  # needs 24 bytes
    cert = certify_weights_file(str(p))
    assert not cert.proven_safe
    assert "byte_length_mismatch" in cert.finding_kinds


def test_unknown_dtype_is_caught(tmp_path):
    p = tmp_path / "unk.safetensors"
    _write_st(p, [("w", "FOO", [1], b"\x00")])
    cert = certify_weights_file(str(p))
    assert not cert.proven_safe
    assert "unknown_dtype" in cert.finding_kinds


def test_storage_overlap_is_caught(tmp_path):
    p = tmp_path / "overlap.safetensors"
    # Hand-craft a header whose two tensors overlap in the buffer.
    header = {
        "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
    }
    raw = json.dumps(header).encode("utf-8")
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        fh.write(b"\x00" * 6)
    cert = certify_weights_file(str(p))
    assert not cert.proven_safe
    assert "storage_overlap" in cert.finding_kinds


def test_storage_gap_is_caught(tmp_path):
    p = tmp_path / "gap.safetensors"
    header = {
        "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
        "b": {"dtype": "U8", "shape": [2], "data_offsets": [4, 6]},  # gap [2,4)
    }
    raw = json.dumps(header).encode("utf-8")
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        fh.write(b"\x00" * 6)
    cert = certify_weights_file(str(p))
    assert not cert.proven_safe
    assert "storage_gap" in cert.finding_kinds


def test_truncated_frame_is_caught(tmp_path):
    p = tmp_path / "trunc.safetensors"
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", 9999))  # claims a 9999-byte header
        fh.write(b"{}")
    cert = certify_weights_file(str(p))
    assert not cert.proven_safe
    assert "malformed_frame" in cert.finding_kinds


def test_non_json_header_is_caught(tmp_path):
    p = tmp_path / "badjson.safetensors"
    body = b"not json at all!"
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(body)))
        fh.write(body)
    cert = certify_weights_file(str(p))
    assert not cert.proven_safe
    assert "malformed_frame" in cert.finding_kinds


# --------------------------------------------------------------------------- #
# Data layer: finiteness.                                                       #
# --------------------------------------------------------------------------- #
def test_inf_is_caught(tmp_path):
    p = tmp_path / "inf.safetensors"
    inf = struct.pack("<I", 0x7F800000)  # +Inf in F32
    _write_st(p, [("w", "F32", [1], inf)])
    cert = certify_weights_file(str(p))
    assert not cert.proven_safe
    assert "non_finite_values" in cert.finding_kinds


def test_finiteness_can_be_disabled(tmp_path):
    p = tmp_path / "inf.safetensors"
    nan = struct.pack("<I", 0x7FC00000)  # NaN in F32
    _write_st(p, [("w", "F32", [1], nan)])
    cert = certify_weights_file(str(p), check_finite=False)
    assert cert.proven_safe  # storage is fine; finiteness was not checked
    assert not cert.checked_finite


# --------------------------------------------------------------------------- #
# Code <-> data contract.                                                       #
# --------------------------------------------------------------------------- #
def test_contract_match_is_certified(tmp_path, good):
    contract = weights_contract_from_file(str(good))
    cert = certify_weights_file(str(good), expected=contract)
    assert cert.proven_safe
    assert cert.contract_checked


def test_contract_shape_mismatch_is_caught(tmp_path, good):
    contract = weights_contract_from_file(str(good))
    target = tmp_path / "tgt.safetensors"
    _write_st(target, [("w", "F32", [2, 4], _f32(8)), ("b", "F16", [4], _f16(4))])
    cert = certify_weights_file(str(target), expected=contract)
    assert not cert.proven_safe
    assert "contract_shape_mismatch" in cert.finding_kinds


def test_contract_missing_and_unexpected_keys(tmp_path, good):
    contract = weights_contract_from_file(str(good))
    target = tmp_path / "tgt.safetensors"
    # drop "b", add "extra"
    _write_st(target, [("w", "F32", [2, 3], _f32(6)), ("extra", "F16", [2], _f16(2))])
    cert = certify_weights_file(str(target), expected=contract)
    assert not cert.proven_safe
    assert "contract_missing_key" in cert.finding_kinds
    assert "contract_unexpected_key" in cert.finding_kinds


# --------------------------------------------------------------------------- #
# Replay / tamper-evidence / serialization.                                     #
# --------------------------------------------------------------------------- #
def test_safe_contract_certificate_self_verifies(tmp_path, good):
    contract = weights_contract_from_file(str(good))
    cert = certify_weights_file(str(good), expected=contract)
    # No reference passed: a *safe* contract cert reconstructs its own contract.
    v = verify_weights_certificate(cert, str(good))
    assert v.verified, v.reasons()


def test_verification_detects_file_tamper(tmp_path, good):
    cert = certify_weights_file(str(good))
    data = bytearray(good.read_bytes())
    data[-1] ^= 0x01  # flip a bit in the last weight byte
    good.write_bytes(bytes(data))
    v = verify_weights_certificate(cert, str(good))
    assert not v.verified
    assert any("file" in r for r in v.reasons())


def test_json_round_trip(good):
    cert = certify_weights_file(str(good))
    back = loads_weights_certificate(dumps_weights_certificate(cert))
    assert weights_certificate_to_dict(back) == weights_certificate_to_dict(cert)
    assert verify_weights_certificate(back, str(good)).verified


def test_dict_round_trip_stable(good):
    cert = certify_weights_file(str(good))
    d = weights_certificate_to_dict(cert)
    again = weights_certificate_to_dict(weights_certificate_from_dict(d))
    assert d == again


def test_render_is_deterministic(good):
    cert = certify_weights_file(str(good))
    a = render_weights_certificate(cert)
    assert a == render_weights_certificate(cert)
    assert a.startswith("# Weights-safety certificate")
    assert "Certified" in a


# --------------------------------------------------------------------------- #
# CLI.                                                                          #
# --------------------------------------------------------------------------- #
def test_cli_weights_safe_exits_zero(good):
    out = io.StringIO()
    rc = main(["weights", str(good)], out=out)
    assert rc == 0
    assert "CERTIFIED" in out.getvalue()


def test_cli_weights_bad_exits_nonzero(tmp_path):
    p = tmp_path / "bad.safetensors"
    _write_st(p, [("w", "F32", [2, 3], b"\x00" * 20)])
    out = io.StringIO()
    rc = main(["weights", str(p)], out=out)
    assert rc == 1
    assert "NOT CERTIFIED" in out.getvalue()


def test_cli_weights_emit_then_verify(tmp_path, good):
    cert_path = tmp_path / "w.cert"
    assert main(["weights", str(good), "-o", str(cert_path)], out=io.StringIO()) == 0
    out = io.StringIO()
    rc = main(["weights-verify", str(good), str(cert_path)], out=out)
    assert rc == 0
    assert "VERIFIED" in out.getvalue()


def test_cli_weights_verify_detects_tamper(tmp_path, good):
    cert_path = tmp_path / "w.cert"
    main(["weights", str(good), "-o", str(cert_path)], out=io.StringIO())
    data = bytearray(good.read_bytes())
    data[-1] ^= 0x01
    good.write_bytes(bytes(data))
    out = io.StringIO()
    rc = main(["weights-verify", str(good), str(cert_path)], out=out)
    assert rc == 1
    assert "NOT VERIFIED" in out.getvalue()


def test_cli_weights_contract(tmp_path, good):
    target = tmp_path / "tgt.safetensors"
    _write_st(target, [("w", "F32", [2, 4], _f32(8)), ("b", "F16", [4], _f16(4))])
    out = io.StringIO()
    rc = main(["weights", str(target), "--expected", str(good)], out=out)
    assert rc == 1
    assert "contract_shape_mismatch" in out.getvalue()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
