"""Roadmap step 3 — **property-based fuzzing of the safetensors parser/certifier**.

Three guarantees are stress-tested over many thousands of generated inputs:

* **Totality** — `read_safetensors_header`, `_parse_tensor_entries`,
  `_validate_storage`, `_scan_nonfinite`, and the top-level `certify_weights_file`
  *never* raise an uncaught exception on *any* byte string. The certificate always
  records the failure as a structured `WeightsFinding`.
* **No unsafe-certified invalid** — whenever the certifier returns
  ``proven_safe``, an **independent** re-implementation of the safetensors spec
  (`_oracle_safe`, written from scratch in this file) agrees the file is safe; and
  whenever the oracle rejects a file, the certifier never marks it safe.
* **Exactness** — in fact ``cert.proven_safe == _oracle_safe(data)`` for every
  generated input (both directions), so the certifier is neither unsound nor
  spuriously conservative on safetensors well-formedness.

The deterministic fuzzer runs **10,000** generated cases (mixing random noise,
mutated-good files, and structured headers); hypothesis adds derandomized
structured coverage; and a battery of targeted families pins each subtly-invalid
category (truncated prefix, header overrun, non-UTF-8 / non-JSON header, unknown
dtype, negative shape, malformed offsets, byte-length disagreement, overlap, gap,
undercoverage, NaN, Inf). Deterministic and torch-free.
"""

from __future__ import annotations

import json
import os
import random
import struct
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.symexec.weights import (
    _DTYPE_SIZE,
    _FLOAT_NONFINITE,
    _MAX_HEADER_BYTES,
    _parse_tensor_entries,
    _scan_nonfinite,
    _validate_storage,
    certify_weights_file,
    read_safetensors_header,
)

_DTYPES = sorted(_DTYPE_SIZE)


# --------------------------------------------------------------------------- #
# Independent oracle: the safetensors "safe" predicate, re-implemented here     #
# from scratch (only the dtype/finite *data* tables are shared ground truth).   #
# --------------------------------------------------------------------------- #
def _oracle_safe(data: bytes, *, check_finite: bool = True) -> bool:
    if len(data) < 8:
        return False
    header_len = int.from_bytes(data[:8], "little")
    if header_len > _MAX_HEADER_BYTES:
        return False
    if 8 + header_len > len(data):
        return False
    raw = data[8:8 + header_len]
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(header, dict):
        return False

    buffer_len = len(data) - 8 - header_len
    infos = []  # (name, dtype, shape, begin, end)
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict):
            return False
        dt = spec.get("dtype")
        shp = spec.get("shape")
        off = spec.get("data_offsets")
        if dt not in _DTYPE_SIZE:
            return False
        if not (isinstance(shp, list)
                and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0
                        for d in shp)):
            return False
        if not (isinstance(off, list) and len(off) == 2
                and all(isinstance(o, int) and not isinstance(o, bool) for o in off)):
            return False
        b, e = off
        if b < 0 or e < b:
            return False
        infos.append((name, dt, tuple(shp), b, e))

    # byte-length consistency + in-bounds
    for _name, dt, shp, b, e in infos:
        numel = 1
        for d in shp:
            numel *= d
        if (e - b) != numel * _DTYPE_SIZE[dt]:
            return False
        if e > buffer_len:
            return False

    # exact tiling of [0, buffer_len) for a non-empty tensor set
    ordered = sorted(infos, key=lambda i: (i[3], i[4]))
    cursor = 0
    for _name, _dt, _shp, b, e in ordered:
        if b > cursor or b < cursor:  # gap or overlap
            return False
        cursor = max(cursor, e)
    if ordered and cursor != buffer_len:
        return False

    # finiteness of float tensors
    if check_finite:
        for _name, dt, _shp, b, e in infos:
            spec = _FLOAT_NONFINITE.get(dt)
            if spec is None:
                continue
            bits, exp_mask, exp_shift = spec
            elem = bits // 8
            if (e - b) % elem != 0:
                continue
            seg = data[8 + header_len + b:8 + header_len + e]
            for o in range(0, len(seg), elem):
                word = int.from_bytes(seg[o:o + elem], "little")
                if (word >> exp_shift) & exp_mask == exp_mask:
                    return False
    return True


# --------------------------------------------------------------------------- #
# safetensors writers / mutators.                                              #
# --------------------------------------------------------------------------- #
def _pack(header_obj, buffer_bytes) -> bytes:
    hb = json.dumps(header_obj).encode("utf-8")
    return struct.pack("<Q", len(hb)) + hb + buffer_bytes


def _good_bytes(rng) -> bytes:
    """A well-formed, finite, contiguously-tiled checkpoint."""
    n = rng.randint(0, 4)
    header, buf, cursor = {}, b"", 0
    for i in range(n):
        dt = rng.choice(_DTYPES)
        size = _DTYPE_SIZE[dt]
        dims = [rng.randint(0, 3) for _ in range(rng.randint(0, 3))]
        numel = 1
        for d in dims:
            numel *= d
        nbytes = numel * size
        # zero bytes => finite for every float dtype.
        raw = b"\x00" * nbytes
        header[f"t{i}"] = {"dtype": dt, "shape": dims,
                           "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
        buf += raw
    if rng.random() < 0.3:
        header["__metadata__"] = {"producer": "fuzz"}
    return _pack(header, buf)


def _mutate(data: bytes, rng) -> bytes:
    b = bytearray(data)
    if not b:
        return bytes([rng.randint(0, 255)])
    op = rng.randint(0, 5)
    if op == 0:  # flip a byte
        i = rng.randrange(len(b))
        b[i] ^= 1 << rng.randint(0, 7)
    elif op == 1:  # truncate
        b = b[:rng.randint(0, len(b))]
    elif op == 2:  # extend with random bytes
        b += bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 8)))
    elif op == 3 and len(b) >= 8:  # perturb the length prefix
        newlen = rng.randint(0, len(b) + 16)
        b[0:8] = struct.pack("<Q", newlen)
    elif op == 4:  # zero a stretch
        i = rng.randrange(len(b))
        for j in range(i, min(len(b), i + rng.randint(1, 4))):
            b[j] = 0
    else:  # set a stretch to 0xFF (can create all-ones exponents => NaN/Inf)
        i = rng.randrange(len(b))
        for j in range(i, min(len(b), i + rng.randint(1, 4))):
            b[j] = 0xFF
    return bytes(b)


def _structured_bytes(rng) -> bytes:
    """A header with deliberately wild fields (often invalid)."""
    n = rng.randint(0, 4)
    header = {}
    cursor = 0
    for i in range(n):
        dt = rng.choice(_DTYPES + ["F7", "weird", 5, None])
        # shapes incl negative / non-int
        shp = rng.choice([
            [rng.randint(-2, 4) for _ in range(rng.randint(0, 3))],
            "notalist",
            [1, "x"],
            [rng.randint(0, 3)],
        ])
        begin = rng.randint(0, 12)
        end = rng.choice([begin + rng.randint(-2, 6), begin, "bad", [begin, 0]])
        off = rng.choice([[begin, end], [begin], [begin, end, 0], "nope"])
        header[f"t{i}"] = {"dtype": dt, "shape": shp, "data_offsets": off}
    buf = bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 24)))
    try:
        return _pack(header, buf)
    except (TypeError, ValueError):
        # non-JSON-serialisable -> emit raw-ish bytes instead
        return bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 40)))


# --------------------------------------------------------------------------- #
# Helpers.                                                                      #
# --------------------------------------------------------------------------- #
def _write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


def _check_invariants(path: str, data: bytes):
    """Totality + exactness for one input. Returns the certificate."""
    # Lower-level parsers never raise.
    header, hlen, fsize, err = read_safetensors_header(path)
    if header is not None:
        infos, _ = _parse_tensor_entries(header)
        buffer_len = fsize - 8 - hlen
        _validate_storage(infos, buffer_len)
        _scan_nonfinite(path, infos, hlen)

    cert = certify_weights_file(path)
    # Always a well-formed verdict object.
    assert isinstance(cert.proven_safe, bool)
    # proven_safe <=> no findings.
    assert cert.proven_safe == (len(cert.findings) == 0)
    # Exactness against the independent oracle.
    assert cert.proven_safe == _oracle_safe(data), (
        f"oracle disagreement: proven_safe={cert.proven_safe} "
        f"oracle={_oracle_safe(data)} findings={cert.finding_kinds} "
        f"data={data[:64]!r}"
    )
    return cert


# --------------------------------------------------------------------------- #
# The 10k deterministic fuzzer.                                                 #
# --------------------------------------------------------------------------- #
def test_fuzz_10k_total_and_exact(tmp_path):
    rng = random.Random(0x5AFE7E2)
    p = tmp_path / "fuzz.safetensors"
    safe_seen = invalid_seen = 0
    N = 10000
    for it in range(N):
        s = rng.random()
        if s < 0.30:
            data = bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 48)))
        elif s < 0.55:
            data = _good_bytes(rng)
        elif s < 0.80:
            data = _mutate(_good_bytes(rng), rng)
        else:
            data = _structured_bytes(rng)
        path = _write(p, data)
        cert = _check_invariants(path, data)
        if cert.proven_safe:
            safe_seen += 1
        else:
            invalid_seen += 1
    # The corpus exercised both verdicts in volume.
    assert safe_seen > 200, safe_seen
    assert invalid_seen > 2000, invalid_seen


# --------------------------------------------------------------------------- #
# Hypothesis: derandomized structured coverage.                                 #
# --------------------------------------------------------------------------- #
_offset_pair = st.lists(st.integers(min_value=-4, max_value=64), min_size=0, max_size=3)
_shape = st.one_of(
    st.lists(st.integers(min_value=-2, max_value=5), max_size=4),
    st.text(max_size=4),
)
_dtype = st.sampled_from(_DTYPES + ["F7", "??"])

_tensor_spec = st.fixed_dictionaries({
    "dtype": _dtype,
    "shape": _shape,
    "data_offsets": _offset_pair,
})
_header = st.dictionaries(st.text(min_size=1, max_size=4), _tensor_spec, max_size=4)
_buffer = st.binary(max_size=48)


@settings(max_examples=2500, derandomize=True, deadline=None, database=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture,
                                 HealthCheck.too_slow])
@given(header=_header, buf=_buffer)
def test_hypothesis_structured_headers(tmp_path, header, buf):
    try:
        data = _pack(header, buf)
    except (TypeError, ValueError):
        return
    path = _write(tmp_path / "h.safetensors", data)
    _check_invariants(path, data)


@settings(max_examples=2500, derandomize=True, deadline=None, database=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture,
                                 HealthCheck.too_slow])
@given(data=st.binary(max_size=80))
def test_hypothesis_raw_bytes(tmp_path, data):
    path = _write(tmp_path / "r.safetensors", data)
    _check_invariants(path, data)


# --------------------------------------------------------------------------- #
# Targeted subtly-invalid families: each must be rejected with a finding.       #
# --------------------------------------------------------------------------- #
def _certify(tmp_path, data):
    path = _write(tmp_path / "c.safetensors", data)
    cert = _check_invariants(path, data)
    return cert


def test_family_truncated_prefix(tmp_path):
    for data in (b"", b"\x00", b"\x00\x00\x00\x00\x00\x00\x00"):  # < 8 bytes
        cert = _certify(tmp_path, data)
        assert not cert.proven_safe
        assert "malformed_frame" in cert.finding_kinds


def test_family_header_overrun(tmp_path):
    data = struct.pack("<Q", 1000) + b"{}"  # claims 1000-byte header, has 2
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "malformed_frame" in cert.finding_kinds


def test_family_header_cap_exceeded(tmp_path):
    data = struct.pack("<Q", _MAX_HEADER_BYTES + 1)
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "malformed_frame" in cert.finding_kinds


def test_family_non_utf8_header(tmp_path):
    raw = b"\xff\xfe\xfa\x00"
    data = struct.pack("<Q", len(raw)) + raw
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "malformed_frame" in cert.finding_kinds


def test_family_non_json_header(tmp_path):
    raw = b"not json at all"
    data = struct.pack("<Q", len(raw)) + raw
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "malformed_frame" in cert.finding_kinds


def test_family_header_not_object(tmp_path):
    raw = b"[1, 2, 3]"
    data = struct.pack("<Q", len(raw)) + raw
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "malformed_frame" in cert.finding_kinds


def test_family_unknown_dtype(tmp_path):
    data = _pack({"w": {"dtype": "F7", "shape": [1], "data_offsets": [0, 4]}}, b"\x00" * 4)
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "unknown_dtype" in cert.finding_kinds


def test_family_negative_shape(tmp_path):
    data = _pack({"w": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}, b"\x00" * 4)
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "malformed_shape" in cert.finding_kinds


def test_family_malformed_offsets(tmp_path):
    for off in ([0], [4, 2], [0, 1, 2], "nope"):
        data = _pack({"w": {"dtype": "U8", "shape": [1], "data_offsets": off}}, b"\x00")
        cert = _certify(tmp_path, data)
        assert not cert.proven_safe
        assert ("malformed_offsets" in cert.finding_kinds
                or "malformed_entry" in cert.finding_kinds), off


def test_family_byte_length_offbyone(tmp_path):
    # F32[1] needs 4 bytes; declare a 5-byte span.
    data = _pack({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 5]}}, b"\x00" * 5)
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "byte_length_mismatch" in cert.finding_kinds


def test_family_overlap(tmp_path):
    header = {"a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]},
              "b": {"dtype": "U8", "shape": [8], "data_offsets": [4, 12]}}
    data = _pack(header, b"\x00" * 12)
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "storage_overlap" in cert.finding_kinds


def test_family_gap(tmp_path):
    header = {"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
              "b": {"dtype": "U8", "shape": [4], "data_offsets": [8, 12]}}
    data = _pack(header, b"\x00" * 12)
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "storage_gap" in cert.finding_kinds


def test_family_undercoverage(tmp_path):
    data = _pack({"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}, b"\x00" * 8)
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "storage_undercovered" in cert.finding_kinds


def test_family_out_of_bounds(tmp_path):
    data = _pack({"a": {"dtype": "U8", "shape": [100], "data_offsets": [0, 100]}}, b"\x00" * 4)
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "storage_out_of_bounds" in cert.finding_kinds


def test_family_nan_inf(tmp_path):
    # F32 NaN (0x7FC00000) then F32 +Inf (0x7F800000).
    raw = struct.pack("<II", 0x7FC00000, 0x7F800000)
    data = _pack({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, raw)
    cert = _certify(tmp_path, data)
    assert not cert.proven_safe and "non_finite_values" in cert.finding_kinds


def test_family_good_is_safe(tmp_path):
    rng = random.Random(7)
    for _ in range(200):
        data = _good_bytes(rng)
        cert = _certify(tmp_path, data)
        assert cert.proven_safe
        assert _oracle_safe(data)


# --------------------------------------------------------------------------- #
# Oracle self-consistency: the oracle must accept a hand-built good file.       #
# --------------------------------------------------------------------------- #
def test_oracle_accepts_known_good():
    data = _pack({"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}},
                 b"\x00" * 16)
    assert _oracle_safe(data)


def test_oracle_rejects_known_bad():
    assert not _oracle_safe(b"")
    assert not _oracle_safe(struct.pack("<Q", 5) + b"{}")  # overrun
    assert not _oracle_safe(_pack({"w": {"dtype": "F7", "shape": [1],
                                         "data_offsets": [0, 4]}}, b"\x00" * 4))
