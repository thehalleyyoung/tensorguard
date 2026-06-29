"""Proof-carrying bug certificates (SYMEXEC_100_STEPS Step 94).

Every forced-failure report the engine emits is *refutation sound*: it fires
only when a runtime precondition is violated on operands the abstract state pins
to concrete values (see the machine-checked ``refute``/``witness`` lemmas in
``lean/TensorGuard/Symexec/``).  This module distils that fact into a compact,
**self-contained, replayable certificate**: the bug's identity, the *named
runtime precondition* that must hold for the operation to be well-typed, and the
concrete witness operands on which it is violated.

The certificate is deliberately independent of the engine: it carries enough to
let a third party (``src.symexec.replay``, Step 95) **re-derive** the verdict by
re-evaluating the precondition on the witness — without re-running the analysis.
The precondition vocabulary mirrors the Lean ``Ok`` predicates one-for-one:

    dims_equal(a, b)        a == b                      (matmul / cat / einsum)
    feature_match(a, b)     a == b                      (nn.Linear in-features)
    broadcast_compat(a, b)  a == b or a == 1 or b == 1  (elementwise broadcast)
    numel_match(a, b)       a == b                      (reshape)
    index_in_range(i, n)    0 <= i < n                  (axis / index OOB)
    arity_match(a, b)       a == b                      (unpack / return arity)
    divisor_nonzero(d)      d != 0                      (division by zero)
    dim_nonneg(d)           d >= 0                      (negative dimension)

A certificate is built by parsing the report's **fingerprinted** message (which
is contractually stable — it is part of the proof footprint, so a format change
trips the corpus fingerprint test).  When a kind's operands cannot be recovered,
the certificate is *claim-only* (``operands is None``): it still names the
violated precondition, but replay reports it ``unchecked`` rather than pretending
to verify it.

Torch-free; standard library only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CERTIFICATE_VERSION",
    "PRECONDITIONS",
    "BugCertificate",
    "certify",
    "certify_result",
    "certificate_to_dict",
    "certificate_from_dict",
    "dumps_certificates",
    "loads_certificates",
]

CERTIFICATE_VERSION = 1


# --------------------------------------------------------------------------- #
# The runtime-precondition vocabulary (mirrors the Lean ``Ok`` predicates).    #
# Each is a total, pure function of the integer witness operands; the          #
# certificate asserts it is *violated* on the witness.                         #
# --------------------------------------------------------------------------- #
def _dims_equal(ops: Tuple[int, ...]) -> bool:
    return ops[0] == ops[1]


def _broadcast_compat(ops: Tuple[int, ...]) -> bool:
    a, b = ops[0], ops[1]
    return a == b or a == 1 or b == 1


def _numel_match(ops: Tuple[int, ...]) -> bool:
    return ops[0] == ops[1]


def _index_in_range(ops: Tuple[int, ...]) -> bool:
    i, n = ops[0], ops[1]
    return 0 <= i < n


def _arity_match(ops: Tuple[int, ...]) -> bool:
    return ops[0] == ops[1]


def _divisor_nonzero(ops: Tuple[int, ...]) -> bool:
    return ops[0] != 0


def _dim_nonneg(ops: Tuple[int, ...]) -> bool:
    return ops[0] >= 0


def _not_none(ops: Tuple[int, ...]) -> bool:
    # Operand is a 1/0 flag: 1 == the value is the abstract None, 0 == not None.
    # The precondition (dereference/unpack target is non-None) holds iff the flag
    # is 0; a fired bug carries the witness flag 1 (precondition violated).
    return ops[0] == 0


#: predicate name -> (the precondition, its arity, a human description).
PRECONDITIONS: Dict[str, Tuple[Callable[[Tuple[int, ...]], bool], int, str]] = {
    "dims_equal": (_dims_equal, 2, "the two dimensions must be equal"),
    "feature_match": (_dims_equal, 2, "the input's last dim must equal in_features"),
    "broadcast_compat": (_broadcast_compat, 2,
                         "aligned dims must be equal or one must be 1"),
    "numel_match": (_numel_match, 2,
                    "input element count must equal the target element count"),
    "index_in_range": (_index_in_range, 2,
                       "the index must satisfy 0 <= index < length"),
    "arity_match": (_arity_match, 2,
                    "the produced arity must equal the target count"),
    "divisor_nonzero": (_divisor_nonzero, 1, "the divisor must be non-zero"),
    "dim_nonneg": (_dim_nonneg, 1, "a constructed dimension must be >= 0"),
    "not_none": (_not_none, 1,
                 "a dereferenced or unpacked value must not be None"),
}


def precondition_holds(predicate: str, operands: Sequence[int]) -> bool:
    """Evaluate a named runtime precondition on concrete operands.

    Raises ``KeyError`` for an unknown predicate and ``ValueError`` on an
    operand-count mismatch, so a malformed certificate cannot silently pass."""
    fn, arity, _ = PRECONDITIONS[predicate]
    ops = tuple(int(x) for x in operands)
    if len(ops) != arity:
        raise ValueError(
            f"predicate {predicate!r} expects {arity} operands, got {len(ops)}"
        )
    return fn(ops)


# --------------------------------------------------------------------------- #
# Per-kind extraction: (default predicate, operand parser).                    #
# Parsers read the fingerprinted message and return the witness operands, or   #
# ``None`` when this kind carries no recoverable numeric witness.              #
# --------------------------------------------------------------------------- #
def _re_pair(pattern: str) -> Callable[[str], Optional[Tuple[int, ...]]]:
    rx = re.compile(pattern)

    def parse(msg: str) -> Optional[Tuple[int, ...]]:
        m = rx.search(msg)
        if not m:
            return None
        return (int(m.group(1)), int(m.group(2)))

    return parse


def _parse_reshape(msg: str) -> Optional[Tuple[int, ...]]:
    # "reshape target (5, 5) is incompatible with a tensor of 6 elements"
    tgt = re.search(r"target \(([\d,\s]+)\)", msg)
    numel = re.search(r"of (\d+) element", msg)
    if not tgt or not numel:
        return None
    dims = [int(x) for x in re.findall(r"\d+", tgt.group(1))]
    if not dims:
        return None
    prod = 1
    for d in dims:
        prod *= d
    return (int(numel.group(1)), prod)


def _parse_divzero(_msg: str) -> Optional[Tuple[int, ...]]:
    # The detector fires only when the divisor is the known constant 0.
    return (0,)


#: kind value -> (predicate name, message->operands parser).
_EXTRACTORS: Dict[str, Tuple[str, Callable[[str], Optional[Tuple[int, ...]]]]] = {
    "matmul_dim_mismatch": ("dims_equal", _re_pair(r"\((\d+) vs (\d+)\)")),
    "broadcast_mismatch": ("broadcast_compat", _re_pair(r"\((\d+) vs (\d+)\)")),
    "cat_shape_mismatch": ("dims_equal", _re_pair(r"\(\[(\d+),\s*(\d+)\]\)")),
    "einsum_dim_mismatch": ("dims_equal", _re_pair(r"sizes (\d+) vs (\d+)")),
    "layer_dim_mismatch": ("feature_match",
                           _re_pair(r"last input dim (\d+) but received (\d+)")),
    "reshape_size_mismatch": ("numel_match", _parse_reshape),
    "axis_out_of_range": ("index_in_range",
                          _re_pair(r"dim (\d+) but the tensor has rank (\d+)")),
    "tensor_index_oob": ("index_in_range",
                         _re_pair(r"index (\d+) .*length (\d+)")),
    "rank_index_error": ("index_in_range",
                         _re_pair(r"index (\d+) .*length (\d+)")),
    "division_by_zero": ("divisor_nonzero", _parse_divzero),
    "negative_dimension": ("dim_nonneg", lambda _m: None),
    "unpack_arity_mismatch": ("arity_match",
                              _re_pair(r"(\d+)-tuple.*?(\d+)")),
    "return_arity_contract": ("arity_match",
                              _re_pair(r"expected (\d+) values?, got (\d+)")),
}


@dataclass(frozen=True)
class BugCertificate:
    """A self-contained, replayable certificate for one report.

    ``operands is None`` marks a *claim-only* certificate (the violated
    precondition is named, but no numeric witness was recoverable)."""

    version: int
    kind: str
    line: int
    col: int
    function: str
    message: str
    predicate: str
    claim: str
    operands: Optional[Tuple[int, ...]]
    filename: Optional[str] = None

    @property
    def is_claim_only(self) -> bool:
        return self.operands is None


def certify(bug, filename: Optional[str] = None) -> BugCertificate:
    """Build a proof-carrying certificate for a single :class:`SymBug`."""
    kind = getattr(bug.kind, "value", str(bug.kind))
    message = getattr(bug, "message", "") or ""
    predicate = "dims_equal"
    operands: Optional[Tuple[int, ...]] = None
    extractor = _EXTRACTORS.get(kind)
    if extractor is not None:
        predicate, parser = extractor
        try:
            operands = parser(message)
        except Exception:
            operands = None
    # Validate recovered operands against the predicate's arity; drop on mismatch
    # so a certificate never carries an internally inconsistent witness.
    if operands is not None:
        _, arity, _ = PRECONDITIONS.get(predicate, (None, -1, ""))
        if len(operands) != arity:
            operands = None
    claim = PRECONDITIONS.get(predicate, (None, 0, "(unknown precondition)"))[2]
    return BugCertificate(
        version=CERTIFICATE_VERSION,
        kind=kind,
        line=int(getattr(bug, "line", 0)),
        col=int(getattr(bug, "col", 0)),
        function=getattr(bug, "function", "") or "",
        message=message,
        predicate=predicate,
        claim=claim,
        operands=operands,
        filename=filename,
    )


def certify_result(result, filename: str = "<unknown>") -> List[BugCertificate]:
    """Certificates for every report in a :class:`SymResult`."""
    return [certify(b, filename) for b in result.bugs]


# --------------------------------------------------------------------------- #
# Serialization (the certificate is the on-the-wire proof artifact).           #
# --------------------------------------------------------------------------- #
def certificate_to_dict(cert: BugCertificate) -> dict:
    return {
        "version": cert.version,
        "kind": cert.kind,
        "line": cert.line,
        "col": cert.col,
        "function": cert.function,
        "message": cert.message,
        "predicate": cert.predicate,
        "claim": cert.claim,
        "operands": list(cert.operands) if cert.operands is not None else None,
        "filename": cert.filename,
    }


def certificate_from_dict(d: dict) -> BugCertificate:
    ops = d.get("operands")
    return BugCertificate(
        version=int(d.get("version", CERTIFICATE_VERSION)),
        kind=d["kind"],
        line=int(d.get("line", 0)),
        col=int(d.get("col", 0)),
        function=d.get("function", "") or "",
        message=d.get("message", ""),
        predicate=d["predicate"],
        claim=d.get("claim", ""),
        operands=tuple(int(x) for x in ops) if ops is not None else None,
        filename=d.get("filename"),
    )


def dumps_certificates(certs: Sequence[BugCertificate], *, indent: int = 2) -> str:
    payload = {
        "certificate_version": CERTIFICATE_VERSION,
        "certificates": [certificate_to_dict(c) for c in certs],
    }
    return json.dumps(payload, indent=indent, sort_keys=True)


def loads_certificates(text: str) -> List[BugCertificate]:
    payload = json.loads(text)
    return [certificate_from_dict(d) for d in payload.get("certificates", [])]
