from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .provenance import stable_json_hash

OBLIGATION_SCHEMA_VERSION = "datarefine.obligation.v1"
OBLIGATION_PACKET_SCHEMA_VERSION = "datarefine.obligations.v1"
_RESERVED_OBLIGATION_FIELDS = {
    "content_hash",
    "evidence",
    "id",
    "kind",
    "limitations",
    "parent_ids",
    "predicate",
    "schema_version",
    "status",
    "target",
}

OBLIGATION_KINDS = {
    "schema",
    "role",
    "split",
    "provenance",
    "temporal",
    "uncertainty",
    "differentiability",
    "privacy",
    "lineage",
    "claim_scope",
    "group",
    "sampling",
    "join",
    "domain",
}

OBLIGATION_STATUSES = {
    "unknown",
    "admitted",
    "rejected",
    "empirical-required",
    "enforced",
    "preserved",
    "blocked",
    "limited",
}


@dataclass(frozen=True)
class Obligation:
    kind: str
    target: str
    predicate: str
    status: str = "unknown"
    evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    parent_ids: tuple[str, ...] = ()
    payload: Mapping[str, object] = field(default_factory=dict)

    def __init__(
        self,
        kind: str,
        target: str,
        predicate: str,
        *,
        status: str = "unknown",
        evidence: Sequence[str] = (),
        limitations: Sequence[str] = (),
        parent_ids: Sequence[str] = (),
        payload: Mapping[str, object] | None = None,
    ) -> None:
        _validate_kind(kind)
        _validate_non_empty("target", target)
        _validate_non_empty("predicate", predicate)
        _validate_status(status)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence", _strings("evidence", evidence, allow_empty=True))
        object.__setattr__(self, "limitations", _strings("limitations", limitations, allow_empty=True))
        object.__setattr__(self, "parent_ids", _strings("parent_ids", parent_ids, allow_empty=True))
        _validate_payload_keys({} if payload is None else payload)
        object.__setattr__(self, "payload", _stable_mapping({} if payload is None else payload))

    @property
    def obligation_id(self) -> str:
        return f"obl:{self.content_hash[:24]}"

    @property
    def id(self) -> str:
        return self.obligation_id

    def canonical_content(self) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": OBLIGATION_SCHEMA_VERSION,
            "kind": self.kind,
            "target": self.target,
            "predicate": self.predicate,
            "status": self.status,
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
            "parent_ids": list(self.parent_ids),
        }
        out.update(dict(self.payload))
        return out

    def as_dict(self) -> dict[str, object]:
        out = {
            "schema_version": OBLIGATION_SCHEMA_VERSION,
            "id": self.obligation_id,
            "content_hash": self.content_hash,
            "kind": self.kind,
            "target": self.target,
            "predicate": self.predicate,
            "status": self.status,
            "evidence": list(self.evidence),
            "limitations": list(self.limitations),
            "parent_ids": list(self.parent_ids),
        }
        out.update(dict(self.payload))
        return out

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"

    @property
    def content_hash(self) -> str:
        return stable_json_hash(self.canonical_content())


@dataclass(frozen=True)
class ObligationPacket:
    obligations: tuple[Obligation, ...]
    schema_version: str = OBLIGATION_PACKET_SCHEMA_VERSION
    producer: str = "datarefine"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __init__(
        self,
        obligations: Sequence[Mapping[str, object] | Obligation],
        *,
        schema_version: str = OBLIGATION_PACKET_SCHEMA_VERSION,
        producer: str = "datarefine",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        normalized = tuple(sorted((_coerce_obligation(obligation) for obligation in obligations), key=lambda item: item.obligation_id))
        object.__setattr__(self, "obligations", normalized)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "metadata", _stable_mapping(metadata or {}))

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "obligations": [obligation.as_dict() for obligation in self.obligations],
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"

    @property
    def content_hash(self) -> str:
        return stable_json_hash(self.as_dict())


def _validate_kind(kind: str) -> None:
    if kind not in OBLIGATION_KINDS:
        raise ValueError(f"unsupported obligation kind {kind!r}")


def _validate_status(status: str) -> None:
    if status not in OBLIGATION_STATUSES:
        raise ValueError(f"unsupported obligation status {status!r}")


def _validate_non_empty(field: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")


def _strings(field: str, values: Sequence[str], *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of strings")
    out = tuple(values)
    if not allow_empty and not out:
        raise ValueError(f"{field} must not be empty")
    if any(not isinstance(value, str) or not value for value in out):
        raise ValueError(f"{field} must contain non-empty strings")
    return out


def _stable_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("payload metadata must be a mapping")
    return {str(key): value[key] for key in sorted(value)}


def _validate_payload_keys(value: Mapping[str, object]) -> None:
    overlap = _RESERVED_OBLIGATION_FIELDS.intersection(str(key) for key in value)
    if overlap:
        raise ValueError(f"payload metadata uses reserved obligation fields: {', '.join(sorted(overlap))}")


def _coerce_obligation(obligation: Mapping[str, object] | Obligation) -> Obligation:
    if isinstance(obligation, Obligation):
        return obligation
    kind = str(obligation.get("kind", ""))
    _validate_kind(kind)
    payload = dict(obligation)
    payload.pop("kind", None)
    supplied_id = payload.pop("id", None)
    supplied_hash = payload.pop("content_hash", None)
    payload.pop("schema_version", None)
    target = str(_pop_first(payload, ("target", "field", "claim"), kind))
    predicate = str(_pop_first(payload, ("predicate", "description"), f"{kind} obligation"))
    status = str(payload.pop("status", "unknown"))
    evidence = payload.pop("evidence", ())
    limitations = payload.pop("limitations", ())
    parent_ids = payload.pop("parent_ids", ())
    coerced = Obligation(
        kind,
        target,
        predicate,
        status=status,
        evidence=tuple(str(item) for item in evidence),
        limitations=tuple(str(item) for item in limitations),
        parent_ids=tuple(str(item) for item in parent_ids),
        payload=payload,
    )
    if supplied_hash is not None and supplied_hash != coerced.content_hash:
        raise ValueError("content_hash does not match canonical obligation content")
    if supplied_id is not None and supplied_id != coerced.obligation_id:
        raise ValueError("id does not match canonical obligation content")
    return coerced


def _pop_first(payload: dict[str, object], keys: tuple[str, ...], default: object) -> object:
    for key in keys:
        if key in payload:
            return payload.pop(key)
    return default


def obligation(
    kind: str,
    target: str,
    predicate: str,
    *,
    status: str = "unknown",
    evidence: Sequence[str] = (),
    limitations: Sequence[str] = (),
    parent_ids: Sequence[str] = (),
    **payload: object,
) -> Obligation:
    return Obligation(
        kind,
        target,
        predicate,
        status=status,
        evidence=evidence,
        limitations=limitations,
        parent_ids=parent_ids,
        payload=payload,
    )


def obligation_packet(*obligations: Mapping[str, object] | Obligation, metadata: Mapping[str, object] | None = None) -> ObligationPacket:
    return ObligationPacket(obligations, metadata=metadata)


__all__ = [
    "OBLIGATION_KINDS",
    "OBLIGATION_PACKET_SCHEMA_VERSION",
    "OBLIGATION_SCHEMA_VERSION",
    "OBLIGATION_STATUSES",
    "Obligation",
    "ObligationPacket",
    "obligation",
    "obligation_packet",
]
