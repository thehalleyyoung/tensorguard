"""Proof-carrying **weights-safety certificates** — certified well-formedness of a
transformer's *weights layer*, both the **data** (the checkpoint file) and the
**code↔data contract** (does the checkpoint actually fit the model that will load
it).

This is the weights-layer dual of :mod:`~src.symexec.safety_certificate`: that
module certifies a program's *shape* code never forces a failure; this one
certifies the *weights* a program will load are themselves safe to load.

Scope and soundness
-------------------
We target the modern transformer weights standard, **safetensors**, because its
header is a self-describing JSON map ``name -> {dtype, shape, data_offsets}`` — so
every claim below is decidable **torch-free** by reading bytes, with no execution
and no false positives:

* **Data well-formedness** (the file is loadable at all): the 8-byte little-endian
  header length is in range, the header is valid JSON, every tensor names a known
  dtype and a non-negative shape, and the storage offsets *tile the buffer
  exactly* — sorted, starting at 0, contiguous, non-overlapping, ending at the
  buffer length, each span equal to ``prod(shape) * dtype_size``.  A file passing
  this is exactly one a conforming safetensors loader accepts.
* **Data finiteness** (the weights are usable, not silently corrupt): every float
  tensor is scanned for ``NaN``/``Inf`` directly from its IEEE-754 bit pattern
  (the all-ones-exponent class), again torch-free.
* **Code↔data contract** (the checkpoint fits the model): against an expected
  ``name -> (dtype, shape)`` contract — what ``load_state_dict(strict=True)``
  enforces — we report missing keys, unexpected keys, dtype mismatches and shape
  mismatches.  A contract can be lifted from a *reference* checkpoint
  (:func:`weights_contract_from_file`) so "this retrain matches the known-good
  architecture" is itself certifiable.

The certificate is **replayable**: :func:`verify_weights_certificate` re-reads the
file, re-derives the structural fingerprint and findings, and confirms the
verdict — trusting nothing in the certificate but its claims (file SHA-256 +
structural fingerprint are the determinism receipt).

Torch-free; standard library only.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "WEIGHTS_CERTIFICATE_VERSION",
    "WeightTensorInfo",
    "WeightsFinding",
    "WeightsSafetyCertificate",
    "WeightsVerification",
    "read_safetensors_header",
    "certify_weights_file",
    "verify_weights_certificate",
    "weights_contract_from_file",
    "weights_certificate_to_dict",
    "weights_certificate_from_dict",
    "dumps_weights_certificate",
    "loads_weights_certificate",
    "render_weights_certificate",
]

WEIGHTS_CERTIFICATE_VERSION = 1

# Canonical safetensors dtype -> element size in bytes.
_DTYPE_SIZE: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

# Float dtypes whose IEEE bit pattern we can scan for NaN/Inf, with the
# (exponent mask, exponent-shift) pair identifying the all-ones-exponent class.
# F8 variants are skipped (no portable struct unpack; rarely the dominant store).
_FLOAT_NONFINITE: Dict[str, Tuple[int, int, int]] = {
    # dtype: (total_bits, exponent_mask_after_shift, exponent_shift)
    "F16": (16, 0x1F, 10),
    "BF16": (16, 0xFF, 7),
    "F32": (32, 0xFF, 23),
    "F64": (64, 0x7FF, 52),
}

_MAX_HEADER_BYTES = 100 * 1000 * 1000  # safetensors spec hard cap.


# --------------------------------------------------------------------------- #
# Data model.                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WeightTensorInfo:
    """One tensor entry from a safetensors header (the *declared* data layout)."""

    name: str
    dtype: str
    shape: Tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class WeightsFinding:
    """A reason the weights layer is *not* certified safe.

    ``kind`` is a stable machine code (e.g. ``"storage_overlap"``,
    ``"byte_length_mismatch"``, ``"non_finite_values"``, ``"contract_missing_key"``);
    ``name`` is the offending tensor (or ``None`` for whole-file findings)."""

    kind: str
    name: Optional[str]
    detail: str


@dataclass(frozen=True)
class WeightsSafetyCertificate:
    """A self-contained, replayable certificate that a safetensors checkpoint is
    safe to load: well-formed storage, finite float weights, and (optionally)
    conformant to an expected contract.

    ``proven_safe`` is ``True`` exactly when there are no findings.  ``file_sha256``
    and ``structural_fingerprint`` bind the certificate to an exact file and an
    exact structural reading, so :func:`verify_weights_certificate` can re-derive
    the verdict from the file alone."""

    version: int
    filename: str
    file_sha256: str
    structural_fingerprint: str
    proven_safe: bool
    num_tensors: int
    total_tensor_bytes: int
    format: str
    checked_finite: bool
    contract_checked: bool
    tensors: Tuple[WeightTensorInfo, ...]
    findings: Tuple[WeightsFinding, ...]

    @property
    def finding_kinds(self) -> Tuple[str, ...]:
        return tuple(sorted({f.kind for f in self.findings}))


@dataclass(frozen=True)
class WeightsVerification:
    """The result of independently re-checking a :class:`WeightsSafetyCertificate`."""

    verified: bool
    checks: Tuple[Tuple[str, bool, str], ...]

    def reasons(self) -> List[str]:
        return [detail for _name, ok, detail in self.checks if not ok]


# --------------------------------------------------------------------------- #
# Torch-free safetensors reading.                                               #
# --------------------------------------------------------------------------- #
def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors_header(
    path: str,
) -> Tuple[Optional[dict], int, int, Optional[str]]:
    """Read a safetensors header torch-free.

    Returns ``(header, header_len, file_size, error)``.  On a malformed *frame*
    (too short, bad length prefix, non-JSON header) ``header`` is ``None`` and
    ``error`` is a human-readable reason; per-tensor validation happens later."""
    import os

    file_size = os.path.getsize(path)
    if file_size < 8:
        return None, 0, file_size, "file shorter than the 8-byte header-length prefix"
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > _MAX_HEADER_BYTES:
            return None, header_len, file_size, (
                f"header length {header_len} exceeds the {_MAX_HEADER_BYTES}-byte cap"
            )
        if 8 + header_len > file_size:
            return None, header_len, file_size, (
                f"header length {header_len} overruns the file "
                f"(size {file_size})"
            )
        raw = fh.read(header_len)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, header_len, file_size, f"header is not valid JSON: {exc}"
    if not isinstance(header, dict):
        return None, header_len, file_size, "header JSON is not an object"
    return header, header_len, file_size, None


def _parse_tensor_entries(
    header: dict,
) -> Tuple[List[WeightTensorInfo], List[WeightsFinding]]:
    """Turn a parsed header into typed tensor infos, collecting malformed entries
    as findings (never raising — the certificate records the failure)."""
    infos: List[WeightTensorInfo] = []
    findings: List[WeightsFinding] = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict):
            findings.append(WeightsFinding("malformed_entry", name, "entry is not an object"))
            continue
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if dtype not in _DTYPE_SIZE:
            findings.append(WeightsFinding("unknown_dtype", name, f"dtype {dtype!r} is not a known safetensors dtype"))
            continue
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            findings.append(WeightsFinding("malformed_shape", name, f"shape {shape!r} is not a list of non-negative ints"))
            continue
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            findings.append(WeightsFinding("malformed_offsets", name, f"data_offsets {offsets!r} is not a [begin, end] int pair"))
            continue
        begin, end = offsets
        if begin < 0 or end < begin:
            findings.append(WeightsFinding("malformed_offsets", name, f"data_offsets {offsets!r} are not 0 <= begin <= end"))
            continue
        infos.append(WeightTensorInfo(name=name, dtype=dtype, shape=tuple(shape), begin=begin, end=end))
    return infos, findings


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _validate_storage(
    infos: List[WeightTensorInfo], buffer_len: int
) -> List[WeightsFinding]:
    """Check byte-length consistency and that offsets tile ``[0, buffer_len)``."""
    findings: List[WeightsFinding] = []
    for info in infos:
        expected = _numel(info.shape) * _DTYPE_SIZE[info.dtype]
        if info.nbytes != expected:
            findings.append(WeightsFinding(
                "byte_length_mismatch", info.name,
                f"declared span {info.nbytes} bytes but {info.dtype}{list(info.shape)} "
                f"needs {expected}",
            ))
        if info.end > buffer_len:
            findings.append(WeightsFinding(
                "storage_out_of_bounds", info.name,
                f"end offset {info.end} exceeds the {buffer_len}-byte data buffer",
            ))

    ordered = sorted(infos, key=lambda i: (i.begin, i.end))
    cursor = 0
    for info in ordered:
        if info.begin > cursor:
            findings.append(WeightsFinding(
                "storage_gap", info.name,
                f"unreferenced gap in bytes [{cursor}, {info.begin})",
            ))
        elif info.begin < cursor:
            findings.append(WeightsFinding(
                "storage_overlap", info.name,
                f"storage overlaps the previous tensor at byte {info.begin} "
                f"(cursor {cursor})",
            ))
        cursor = max(cursor, info.end)
    if ordered and cursor != buffer_len:
        findings.append(WeightsFinding(
            "storage_undercovered", None,
            f"tensors cover {cursor} bytes but the data buffer is {buffer_len}",
        ))
    return findings


def _scan_nonfinite(
    path: str, infos: List[WeightTensorInfo], header_len: int
) -> List[WeightsFinding]:
    """Scan float tensors for NaN/Inf using their IEEE all-ones-exponent class."""
    findings: List[WeightsFinding] = []
    data_start = 8 + header_len
    with open(path, "rb") as fh:
        for info in infos:
            spec = _FLOAT_NONFINITE.get(info.dtype)
            if spec is None:
                continue
            bits, exp_mask, exp_shift = spec
            elem = bits // 8
            if info.nbytes % elem != 0:
                continue
            fh.seek(data_start + info.begin)
            raw = fh.read(info.nbytes)
            count = 0
            for off in range(0, len(raw), elem):
                word = int.from_bytes(raw[off:off + elem], "little")
                if (word >> exp_shift) & exp_mask == exp_mask:
                    count += 1
            if count:
                findings.append(WeightsFinding(
                    "non_finite_values", info.name,
                    f"{count} NaN/Inf element(s) in {info.dtype} tensor",
                ))
    return findings


# --------------------------------------------------------------------------- #
# Contract (code <-> data) checking.                                            #
# --------------------------------------------------------------------------- #
def weights_contract_from_file(path: str) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """Lift an expected ``name -> (dtype, shape)`` contract from a *reference*
    safetensors checkpoint (e.g. a known-good architecture)."""
    header, _hlen, _fsize, err = read_safetensors_header(path)
    if header is None:
        raise ValueError(f"cannot read reference checkpoint {path!r}: {err}")
    infos, bad = _parse_tensor_entries(header)
    if bad:
        raise ValueError(f"reference checkpoint {path!r} is malformed: {bad[0].detail}")
    return {i.name: (i.dtype, i.shape) for i in infos}


def _check_contract(
    infos: List[WeightTensorInfo],
    expected: Dict[str, Tuple[Optional[str], Tuple[int, ...]]],
    *,
    partial: bool = False,
) -> List[WeightsFinding]:
    """Cross-check the checkpoint against an expected ``name -> (dtype, shape)``
    contract (``dtype`` may be ``None`` to skip the dtype check).

    A ``partial`` contract (e.g. one derived from model *code* that could not
    resolve every layer) only ever asserts *positive* obligations — a derived
    tensor must be present with the derived shape — and never reports
    ``contract_unexpected_key``, since the checkpoint may legitimately contain
    parameters the partial contract could not account for."""
    findings: List[WeightsFinding] = []
    have = {i.name: i for i in infos}
    for name in sorted(set(expected) - set(have)):
        findings.append(WeightsFinding("contract_missing_key", name, "expected tensor is absent from the checkpoint"))
    if not partial:
        for name in sorted(set(have) - set(expected)):
            findings.append(WeightsFinding("contract_unexpected_key", name, "checkpoint tensor is not in the contract"))
    for name in sorted(set(have) & set(expected)):
        exp_dtype, exp_shape = expected[name]
        got = have[name]
        if tuple(got.shape) != tuple(exp_shape):
            findings.append(WeightsFinding(
                "contract_shape_mismatch", name,
                f"expected shape {list(exp_shape)} but checkpoint has {list(got.shape)}",
            ))
        if exp_dtype is not None and got.dtype != exp_dtype:
            findings.append(WeightsFinding(
                "contract_dtype_mismatch", name,
                f"expected dtype {exp_dtype} but checkpoint has {got.dtype}",
            ))
    return findings


# --------------------------------------------------------------------------- #
# Certification.                                                                #
# --------------------------------------------------------------------------- #
def _structural_fingerprint(
    infos: List[WeightTensorInfo],
    findings: List[WeightsFinding],
    *,
    checked_finite: bool,
    contract_checked: bool,
) -> str:
    """A deterministic digest of the structural reading (tensors + findings +
    which checks ran), independent of file path — the replay receipt."""
    payload = {
        "tensors": sorted(
            [i.name, i.dtype, list(i.shape), i.begin, i.end] for i in infos
        ),
        "findings": sorted([f.kind, f.name or "", f.detail] for f in findings),
        "checked_finite": checked_finite,
        "contract_checked": contract_checked,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def certify_weights_file(
    path: str,
    *,
    check_finite: bool = True,
    expected: Optional[Dict[str, Tuple[Optional[str], Tuple[int, ...]]]] = None,
    contract_partial: bool = False,
) -> WeightsSafetyCertificate:
    """Certify a safetensors checkpoint's weights layer (data + optional contract).

    ``proven_safe`` is true iff there are *no* findings: well-formed storage,
    finite float weights (when ``check_finite``), and contract conformance (when
    ``expected`` is given).  ``contract_partial`` marks a contract (e.g. one
    derived from model code) that does not enumerate every parameter, so only
    positive obligations are asserted (no ``contract_unexpected_key``)."""
    header, header_len, file_size, frame_err = read_safetensors_header(path)
    findings: List[WeightsFinding] = []
    infos: List[WeightTensorInfo] = []

    if header is None:
        findings.append(WeightsFinding("malformed_frame", None, frame_err or "unreadable safetensors frame"))
    else:
        infos, entry_findings = _parse_tensor_entries(header)
        findings.extend(entry_findings)
        buffer_len = file_size - 8 - header_len
        findings.extend(_validate_storage(infos, buffer_len))
        # Finiteness is only meaningful once storage is in-bounds.
        storage_bad = any(
            f.kind in ("storage_out_of_bounds", "byte_length_mismatch")
            for f in findings
        )
        if check_finite and not storage_bad:
            findings.extend(_scan_nonfinite(path, infos, header_len))
        if expected is not None:
            findings.extend(_check_contract(infos, expected, partial=contract_partial))

    findings = sorted(findings, key=lambda f: (f.kind, f.name or "", f.detail))
    total_bytes = sum(i.nbytes for i in infos)
    return WeightsSafetyCertificate(
        version=WEIGHTS_CERTIFICATE_VERSION,
        filename=path,
        file_sha256=_file_sha256(path),
        structural_fingerprint=_structural_fingerprint(
            infos, findings, checked_finite=check_finite,
            contract_checked=expected is not None,
        ),
        proven_safe=not findings,
        num_tensors=len(infos),
        total_tensor_bytes=total_bytes,
        format="safetensors",
        checked_finite=check_finite,
        contract_checked=expected is not None,
        tensors=tuple(sorted(infos, key=lambda i: i.begin)),
        findings=tuple(findings),
    )


def verify_weights_certificate(
    cert: WeightsSafetyCertificate,
    path: str,
    *,
    expected: Optional[Dict[str, Tuple[str, Tuple[int, ...]]]] = None,
) -> WeightsVerification:
    """Independently re-derive the certificate's verdict from ``path`` alone.

    Re-hash the file, re-run the (deterministic) structural reading, and confirm
    (a) the file matches, (b) the structural fingerprint matches, and (c) the
    proven-safe verdict matches a fresh certification under the same options.

    A *safe* contract certificate is fully self-verifying: its expected contract
    is exactly ``{name: (dtype, shape)}`` over its own tensors (safety ⇒ every key
    matched, none missing/extra), so it is reconstructed and re-checked, and the
    fingerprint must reproduce *exactly*.  For an *unsafe* contract certificate the
    original contract is not recoverable; pass ``expected`` to fully re-verify, or
    the data-layer core is re-verified and the contract findings are trusted."""
    checks: List[Tuple[str, bool, str]] = []

    file_ok = _file_sha256(path) == cert.file_sha256
    checks.append((
        "file_sha256", file_ok,
        "ok" if file_ok else "file does not match the certified checkpoint",
    ))

    # Choose the contract to re-check against.
    replay_expected = expected
    exact_fingerprint = True
    if cert.contract_checked and replay_expected is None:
        if cert.proven_safe:
            replay_expected = {t.name: (t.dtype, t.shape) for t in cert.tensors}
        else:
            exact_fingerprint = False  # cannot reconstruct an unsafe contract
    elif not cert.contract_checked:
        replay_expected = None

    fresh = certify_weights_file(
        path, check_finite=cert.checked_finite, expected=replay_expected,
    )

    if exact_fingerprint:
        fp_ok = fresh.structural_fingerprint == cert.structural_fingerprint
        fp_detail = "structural fingerprint reproduced exactly"
    else:
        fp_ok = _structural_core_matches(cert, fresh)
        fp_detail = "structural core (tensors + data findings) reproduced"
    checks.append((
        "structural_fingerprint", fp_ok,
        "ok" if fp_ok else f"{fp_detail}: MISMATCH",
    ))

    if cert.proven_safe:
        verdict_ok = fresh.proven_safe
        verdict_detail = "fresh certification is also safe"
    else:
        # A non-safe verdict must reproduce at least one finding (or, for an
        # unrecoverable contract, the data core already matched above).
        verdict_ok = (not fresh.proven_safe) or not exact_fingerprint
        verdict_detail = "fresh certification reproduces an unsafe verdict"
    checks.append((
        "verdict", verdict_ok,
        "ok" if verdict_ok else f"{verdict_detail}: MISMATCH",
    ))

    verified = all(ok for _n, ok, _d in checks)
    return WeightsVerification(verified=verified, checks=tuple(checks))


def _structural_core_matches(
    cert: WeightsSafetyCertificate, fresh: WeightsSafetyCertificate
) -> bool:
    """Compare the contract-independent structural core of two certificates."""
    def core(c: WeightsSafetyCertificate):
        tensors = sorted((t.name, t.dtype, t.shape, t.begin, t.end) for t in c.tensors)
        data_findings = sorted(
            (f.kind, f.name or "", f.detail)
            for f in c.findings
            if not f.kind.startswith("contract_")
        )
        return tensors, data_findings

    return core(cert) == core(fresh)


# --------------------------------------------------------------------------- #
# Serialization.                                                                #
# --------------------------------------------------------------------------- #
def weights_certificate_to_dict(cert: WeightsSafetyCertificate) -> dict:
    return {
        "version": cert.version,
        "filename": cert.filename,
        "file_sha256": cert.file_sha256,
        "structural_fingerprint": cert.structural_fingerprint,
        "proven_safe": cert.proven_safe,
        "num_tensors": cert.num_tensors,
        "total_tensor_bytes": cert.total_tensor_bytes,
        "format": cert.format,
        "checked_finite": cert.checked_finite,
        "contract_checked": cert.contract_checked,
        "tensors": [
            {
                "name": t.name,
                "dtype": t.dtype,
                "shape": list(t.shape),
                "begin": t.begin,
                "end": t.end,
            }
            for t in cert.tensors
        ],
        "findings": [
            {"kind": f.kind, "name": f.name, "detail": f.detail}
            for f in cert.findings
        ],
    }


def weights_certificate_from_dict(d: dict) -> WeightsSafetyCertificate:
    return WeightsSafetyCertificate(
        version=d["version"],
        filename=d["filename"],
        file_sha256=d["file_sha256"],
        structural_fingerprint=d["structural_fingerprint"],
        proven_safe=d["proven_safe"],
        num_tensors=d["num_tensors"],
        total_tensor_bytes=d["total_tensor_bytes"],
        format=d.get("format", "safetensors"),
        checked_finite=d["checked_finite"],
        contract_checked=d["contract_checked"],
        tensors=tuple(
            WeightTensorInfo(
                name=t["name"], dtype=t["dtype"], shape=tuple(t["shape"]),
                begin=t["begin"], end=t["end"],
            )
            for t in d["tensors"]
        ),
        findings=tuple(
            WeightsFinding(kind=f["kind"], name=f["name"], detail=f["detail"])
            for f in d["findings"]
        ),
    )


def dumps_weights_certificate(cert: WeightsSafetyCertificate, *, indent: int = 2) -> str:
    return json.dumps(weights_certificate_to_dict(cert), indent=indent, sort_keys=True)


def loads_weights_certificate(text: str) -> WeightsSafetyCertificate:
    return weights_certificate_from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# Rendering.                                                                    #
# --------------------------------------------------------------------------- #
def render_weights_certificate(cert: WeightsSafetyCertificate) -> str:
    lines: List[str] = []
    lines.append(f"# Weights-safety certificate for `{cert.filename}`")
    lines.append("")
    if cert.proven_safe:
        lines.append(
            "✅ **Certified: the weights layer is safe to load** — well-formed "
            "storage"
            + (", finite float weights" if cert.checked_finite else "")
            + (", and contract-conformant." if cert.contract_checked else ".")
        )
    else:
        lines.append(
            f"❌ **Not certified** — {len(cert.findings)} weights-safety "
            "finding(s) below."
        )
    lines.append("")
    lines.append(f"- File SHA-256: `{cert.file_sha256}`")
    lines.append(f"- Structural fingerprint: `{cert.structural_fingerprint}`")
    lines.append(
        f"- Format: **{cert.format}**; tensors: **{cert.num_tensors}**; "
        f"tensor bytes: **{cert.total_tensor_bytes}**."
    )
    lines.append(
        f"- Checks: storage well-formedness"
        + (" · finiteness" if cert.checked_finite else "")
        + (" · contract" if cert.contract_checked else "")
        + "."
    )
    lines.append("")

    if cert.findings:
        lines.append("## Findings")
        lines.append("")
        lines.append("| Kind | Tensor | Detail |")
        lines.append("| --- | --- | --- |")
        for f in cert.findings:
            lines.append(f"| `{f.kind}` | `{f.name or '—'}` | {f.detail} |")
        lines.append("")

    lines.append("## What this certifies")
    lines.append("")
    lines.append(
        "Read torch-free from the safetensors header: every tensor names a known "
        "dtype and non-negative shape, the storage offsets tile the data buffer "
        "exactly (sorted, contiguous, non-overlapping, byte-length = "
        "`prod(shape)·dtype_size`)"
        + (
            ", and no float tensor contains a NaN/Inf bit pattern"
            if cert.checked_finite else ""
        )
        + (
            ", and the checkpoint conforms to the expected name→(dtype,shape) "
            "contract (what `load_state_dict(strict=True)` enforces)"
            if cert.contract_checked else ""
        )
        + ". The certificate is replayable: re-reading the SHA-256-pinned file "
        "must reproduce the structural fingerprint and verdict."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
