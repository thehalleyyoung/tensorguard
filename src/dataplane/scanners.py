"""Unifying layer that connects every structural scanner family to the typed
core's obligation -> proof-packet -> TensorGuard hand-off pipeline.

Each scanner module (``pipeline_leakage``, ``split_contracts``,
``temporal_leakage``, ``group_leakage``, ``dataloader_determinism``,
``join_cardinality``, ``value_domain``) historically returned its own bespoke
``*Finding`` dataclass and *discarded* the :class:`~datarefine.obligations.Obligation`
it built internally.  Findings therefore never reached
``export_proof_packet`` / ``export_tensorguard_refinement_context`` -- the
value-domain refinement that complements ``thehalleyyoung/tensorguard``'s
shape verifier sat isolated from the rest of the certifier.

This module is the connective tissue:

* a single :data:`CONSTRAINT_TO_KIND` source of truth (validated at import
  against the certifier's own kind<->constraint defaults);
* an ordered :data:`SCANNER_FAMILIES` registry so a new family is declared in
  exactly one place;
* :func:`finding_to_obligation`, which lifts *any* scanner finding back into a
  first-class obligation -- faithfully reconstructing the structural lowering
  payload from the z3 witness where the witness carries enough to reproduce the
  same ``rejected`` verdict (value_domain, join, group, sampling, temporal),
  and recording a verdict-preserving obligation otherwise;
* :func:`findings_to_obligation_packet` / :func:`findings_to_proof_packets`,
  which fold a blind source scan into a typed obligation packet plus genuine,
  independently re-checkable proof packets.

No scanner module is edited; faithfulness is *self-validated* by re-certifying
the reconstructed obligation, so a proof packet is only ever emitted when the
reconstruction provably reproduces the scanner's rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .certification import (
    STRUCTURAL_CONSTRAINTS,
    StructuralCertifier,
    _constraint_of,
    lower_obligation,
)
from .formal_verification import export_proof_packet
from .obligations import (
    OBLIGATION_KINDS,
    OBLIGATION_STATUSES,
    Obligation,
    ObligationPacket,
    obligation,
    obligation_packet,
)

SCANNERS_SCHEMA_VERSION = "datarefine.scanners.v1"

_CERT = StructuralCertifier()


# ---------------------------------------------------------------------------
# Finding protocol (the fields every scanner family shares)
# ---------------------------------------------------------------------------
@runtime_checkable
class Finding(Protocol):
    """The common surface of every scanner ``*Finding`` dataclass."""

    file: str
    line: int
    pattern: str
    constraint: str
    verdict: str


def _detail(finding: object) -> str:
    # ``pipeline_leakage.LeakageFinding`` uses ``explanation`` instead of ``detail``.
    value = getattr(finding, "detail", None)
    if value is None:
        value = getattr(finding, "explanation", "")
    return str(value or "")


def _witness(finding: object) -> Mapping[str, object]:
    value = getattr(finding, "witness", None)
    return value if isinstance(value, Mapping) else {}


# ---------------------------------------------------------------------------
# constraint -> obligation kind (single source of truth)
# ---------------------------------------------------------------------------
CONSTRAINT_TO_KIND: dict[str, str] = {
    "schema_consistency": "schema",
    "role_constraint": "role",
    "split_disjointness": "split",
    "column_lineage": "lineage",
    "join_safety": "join",
    "bounds": "schema",
    "no_outcome_in_feature": "role",
    "fit_transform_isolation": "provenance",
    "partition_lengths": "split",
    "temporal_causality": "temporal",
    "group_disjointness": "group",
    "sampling_independence": "sampling",
    "join_cardinality": "join",
    "value_domain": "domain",
}


def _validate_constraint_map() -> None:
    for constraint, kind in CONSTRAINT_TO_KIND.items():
        if constraint not in STRUCTURAL_CONSTRAINTS:
            raise AssertionError(f"unknown structural constraint in map: {constraint!r}")
        if kind not in OBLIGATION_KINDS:
            raise AssertionError(f"unknown obligation kind in map: {kind!r}")
    # The map must agree with the certifier's own kind -> default-constraint
    # inversion: for every kind that has a canonical default constraint, that
    # constraint must map back to the same kind here.
    for kind in sorted(OBLIGATION_KINDS):
        try:
            probe = obligation(kind, "probe", "probe")
        except Exception:  # pragma: no cover - kind always valid here
            continue
        default_constraint = _constraint_of(probe)
        if default_constraint is None:
            continue
        mapped = CONSTRAINT_TO_KIND.get(default_constraint)
        if mapped is not None and mapped != kind:
            raise AssertionError(
                "CONSTRAINT_TO_KIND disagrees with certifier default for "
                f"{default_constraint!r}: {mapped!r} vs {kind!r}"
            )


_validate_constraint_map()


# ---------------------------------------------------------------------------
# Scanner family registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScannerFamily:
    """One structural scanner family, declared in exactly one place."""

    key: str
    report_field: str
    module: str
    finding_class: str
    obligation_kind: str

    def scan_source(self, source: str, filename: str = "<string>") -> list[object]:
        mod = import_module(f".{self.module}", __package__)
        return list(mod.scan_source(source, filename))


SCANNER_FAMILIES: tuple[ScannerFamily, ...] = (
    ScannerFamily("leakage", "leakage", "pipeline_leakage", "LeakageFinding", "provenance"),
    ScannerFamily("contracts", "contracts", "split_contracts", "ContractFinding", "split"),
    ScannerFamily("temporal", "temporal", "temporal_leakage", "TemporalFinding", "temporal"),
    ScannerFamily("group", "group", "group_leakage", "GroupFinding", "group"),
    ScannerFamily("sampling", "sampling", "dataloader_determinism", "SamplingFinding", "sampling"),
    ScannerFamily("joins", "joins", "join_cardinality", "JoinFinding", "join"),
    ScannerFamily("domains", "domains", "value_domain", "ValueDomainFinding", "domain"),
)

_FAMILY_BY_KEY: dict[str, ScannerFamily] = {fam.key: fam for fam in SCANNER_FAMILIES}


def family(key: str) -> ScannerFamily:
    return _FAMILY_BY_KEY[key]


def scan_all(
    source: str,
    filename: str = "<string>",
    *,
    families: Iterable[str] | None = None,
) -> dict[str, list[object]]:
    """Run every (or a selected subset of) scanner family on ``source``."""

    selected = SCANNER_FAMILIES if families is None else [family(k) for k in families]
    out: dict[str, list[object]] = {}
    for fam in selected:
        try:
            out[fam.key] = fam.scan_source(source, filename)
        except SyntaxError:
            out[fam.key] = []
    return out


def all_findings(scan: Mapping[str, Sequence[object]]) -> list[object]:
    """Flatten a :func:`scan_all` result into a single ordered finding list."""

    findings: list[object] = []
    for fam in SCANNER_FAMILIES:
        findings.extend(scan.get(fam.key, ()))
    return findings


# ---------------------------------------------------------------------------
# Faithful witness -> lowering-payload reconstruction
# ---------------------------------------------------------------------------
def _reconstruct_value_domain(f: object, w: Mapping[str, object]) -> dict[str, object] | None:
    lo = w.get("required_lo")
    hi = w.get("required_hi")
    if lo is None or hi is None:
        return None
    return {
        "constraint": "value_domain",
        "op": str(w.get("op", getattr(f, "op", "bce"))),
        "loss": str(w.get("loss", getattr(f, "loss", ""))),
        "producer": str(w.get("producer", getattr(f, "producer", "unknown"))),
        "domain_established": False,
        "required_lo": lo,
        "required_hi": hi,
    }


def _reconstruct_join(f: object, w: Mapping[str, object]) -> dict[str, object]:
    return {
        "constraint": "join_cardinality",
        "validated": False,
        "right_key_unique": False,
        "cardinality_consumed": True,
        "join_key": str(w.get("join_key", getattr(f, "join_key", "key"))),
        "how": str(w.get("how", getattr(f, "how", "inner"))),
        "left_rows": int(float(w.get("left_rows", 100))),
    }


def _reconstruct_group(f: object, w: Mapping[str, object]) -> dict[str, object]:
    return {
        "constraint": "group_disjointness",
        "group_aware": False,
        "group_size": int(float(w.get("group_size", 2))),
        "partitions": int(float(w.get("partitions", 2))),
        "group_key": str(w.get("group_key", getattr(f, "group_key", "group"))),
    }


def _reconstruct_sampling(f: object, w: Mapping[str, object]) -> dict[str, object] | None:
    violation = w.get("violation")
    if violation == "eval_nondeterminism":
        payload: dict[str, object] = {"is_eval": True, "stochastic_eval": True}
    elif violation == "correlated_worker_rng":
        workers = w.get("num_workers", getattr(f, "num_workers", None))
        payload = {
            "is_eval": False,
            "global_rng": True,
            "worker_init_fn": False,
            "num_workers": int(float(workers if workers is not None else 2)),
        }
    else:
        return None
    payload["constraint"] = "sampling_independence"
    return payload


def _reconstruct_temporal(f: object, w: Mapping[str, object]) -> dict[str, object]:
    reach = w.get("forward_reach", getattr(f, "forward_reach", 1))
    return {
        "constraint": "temporal_causality",
        "forward_reach": int(float(reach if reach is not None else 1)),
        "cut": int(float(w.get("feature_row", 0))),
    }


_RECONSTRUCTORS: dict[str, Callable[[object, Mapping[str, object]], dict[str, object] | None]] = {
    "value_domain": _reconstruct_value_domain,
    "join_cardinality": _reconstruct_join,
    "group_disjointness": _reconstruct_group,
    "sampling_independence": _reconstruct_sampling,
    "temporal_causality": _reconstruct_temporal,
}


def _faithful(ob: Obligation, expected_status: str) -> bool:
    """A reconstruction is faithful iff it re-lowers and re-certifies to the
    *same* verdict the scanner reported -- never trust it blindly."""

    try:
        lower_obligation(ob)
        return _CERT.certify(ob).status == expected_status
    except Exception:
        return False


def _record_obligation(finding: object, kind: str, status: str, constraint: str) -> Obligation:
    """A verdict-preserving record of a scanner finding (no faithful structural
    payload available).  Packet-/handoff-able, but not proof-packet-able."""

    target = f"{getattr(finding, 'file', '<string>')}:{int(getattr(finding, 'line', 0))}"
    predicate = f"{constraint}: {getattr(finding, 'pattern', '')}"
    return obligation(
        kind,
        target,
        predicate,
        status=status,
        scanner_constraint=constraint,
        scanner_pattern=str(getattr(finding, "pattern", "")),
        scanner_line=int(getattr(finding, "line", 0)),
        scanner_detail=_detail(finding),
        scanner_witness=dict(_witness(finding)),
        scanner_recheckable=False,
    )


def finding_to_obligation(finding: object) -> Obligation:
    """Lift a scanner finding back into a first-class obligation.

    Returns a *faithful* structural obligation (carrying the reconstructed
    lowering payload, so it round-trips through ``export_proof_packet``) when
    the witness reproduces the scanner's verdict; otherwise a verdict-preserving
    record obligation.
    """

    constraint = str(getattr(finding, "constraint", ""))
    kind = CONSTRAINT_TO_KIND.get(constraint, "lineage")
    raw_status = str(getattr(finding, "verdict", "unknown"))
    status = raw_status if raw_status in OBLIGATION_STATUSES else "unknown"

    reconstruct = _RECONSTRUCTORS.get(constraint)
    if reconstruct is not None:
        payload = reconstruct(finding, _witness(finding))
        if payload is not None:
            target = f"{getattr(finding, 'file', '<string>')}:{int(getattr(finding, 'line', 0))}"
            predicate = f"{constraint}: {getattr(finding, 'pattern', '')}"
            candidate = obligation(kind, target, predicate, status=status, **payload)
            if _faithful(candidate, status):
                return candidate
    return _record_obligation(finding, kind, status, constraint)


def is_proof_packetable(finding: object) -> bool:
    """True iff a genuine, independently re-checkable proof packet can be built."""

    if str(getattr(finding, "verdict", "")) != "rejected":
        return False
    return "scanner_constraint" not in finding_to_obligation(finding).payload


def proof_packet_for(finding: object) -> dict[str, object] | None:
    """A genuine proof packet for a finding, or ``None`` if not reconstructable."""

    if str(getattr(finding, "verdict", "")) != "rejected":
        return None
    ob = finding_to_obligation(finding)
    if "scanner_constraint" in ob.payload:  # record-only obligation
        return None
    try:
        verdict = _CERT.certify(ob)
        if verdict.status != "rejected":
            return None
        return export_proof_packet(ob, verdict)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Findings -> obligation packet / proof packets
# ---------------------------------------------------------------------------
def findings_to_obligations(findings: Iterable[object]) -> list[Obligation]:
    return [finding_to_obligation(f) for f in findings]


def findings_to_obligation_packet(
    findings: Iterable[object],
    *,
    source_id: str | None = None,
) -> ObligationPacket:
    obligations = findings_to_obligations(findings)
    metadata: dict[str, object] = {
        "schema_version": SCANNERS_SCHEMA_VERSION,
        "producer": "datarefine",
        "families": [fam.key for fam in SCANNER_FAMILIES],
    }
    if source_id is not None:
        metadata["source_id"] = source_id
    return obligation_packet(*obligations, metadata=metadata)


def findings_to_proof_packets(findings: Iterable[object]) -> list[dict[str, object]]:
    packets: list[dict[str, object]] = []
    for finding in findings:
        packet = proof_packet_for(finding)
        if packet is not None:
            packets.append(packet)
    return packets


# ---------------------------------------------------------------------------
# One-shot source/path/tree -> packet helpers
# ---------------------------------------------------------------------------
def scan_source_to_packet(
    source: str,
    filename: str = "<string>",
    *,
    families: Iterable[str] | None = None,
) -> ObligationPacket:
    findings = all_findings(scan_all(source, filename, families=families))
    return findings_to_obligation_packet(findings, source_id=filename)


def scan_path_to_packet(
    path: str | Path,
    *,
    families: Iterable[str] | None = None,
) -> ObligationPacket:
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        source = ""
    return scan_source_to_packet(source, str(p), families=families)


def scan_tree_to_packet(
    root: str | Path,
    *,
    families: Iterable[str] | None = None,
) -> ObligationPacket:
    root = Path(root)
    paths: Iterable[Path] = [root] if root.is_file() else sorted(root.rglob("*.py"))
    obligations: list[Obligation] = []
    for p in paths:
        try:
            source = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        findings = all_findings(scan_all(source, str(p), families=families))
        obligations.extend(findings_to_obligations(findings))
    metadata = {
        "schema_version": SCANNERS_SCHEMA_VERSION,
        "producer": "datarefine",
        "families": [fam.key for fam in SCANNER_FAMILIES],
        "source_id": str(root),
    }
    return obligation_packet(*obligations, metadata=metadata)


__all__ = [
    "SCANNERS_SCHEMA_VERSION",
    "CONSTRAINT_TO_KIND",
    "Finding",
    "ScannerFamily",
    "SCANNER_FAMILIES",
    "family",
    "scan_all",
    "all_findings",
    "finding_to_obligation",
    "findings_to_obligations",
    "findings_to_obligation_packet",
    "findings_to_proof_packets",
    "is_proof_packetable",
    "proof_packet_for",
    "scan_source_to_packet",
    "scan_path_to_packet",
    "scan_tree_to_packet",
]
