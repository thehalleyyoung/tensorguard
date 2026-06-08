from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .certification import CertifierVerdict
from .obligations import Obligation
from .provenance import stable_json_hash


PRIVACY_COMPOSITION_SCHEMA_VERSION = "datarefine.privacy_composition.v1"


@dataclass(frozen=True)
class PrivacyDischarge:
    transform: str
    epsilon: float
    delta: float = 0.0
    mechanism: str = "redaction"
    reference: str = ""

    def __post_init__(self) -> None:
        if not self.transform:
            raise ValueError("privacy discharge transform must be non-empty")
        if self.epsilon < 0 or self.delta < 0:
            raise ValueError("privacy discharge epsilon/delta must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "transform": self.transform,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "mechanism": self.mechanism,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class PrivacyCompositionResult:
    epsilon_bound: float
    delta_bound: float
    discharges: tuple[PrivacyDischarge, ...]

    @property
    def total_epsilon(self) -> float:
        return sum(item.epsilon for item in self.discharges)

    @property
    def total_delta(self) -> float:
        return sum(item.delta for item in self.discharges)

    @property
    def all_pass(self) -> bool:
        return self.total_epsilon <= self.epsilon_bound and self.total_delta <= self.delta_bound

    @property
    def status(self) -> str:
        return "admitted" if self.all_pass else "rejected"

    @property
    def failing_aspects(self) -> tuple[str, ...]:
        aspects: list[str] = []
        if self.total_epsilon > self.epsilon_bound:
            aspects.append("epsilon")
        if self.total_delta > self.delta_bound:
            aspects.append("delta")
        return tuple(aspects)

    @property
    def failure_summary(self) -> str:
        details = []
        if self.total_epsilon > self.epsilon_bound:
            details.append(f"epsilon {self.total_epsilon} exceeds bound {self.epsilon_bound}")
        if self.total_delta > self.delta_bound:
            details.append(f"delta {self.total_delta} exceeds bound {self.delta_bound}")
        return "; ".join(details)

    def as_dict(self) -> dict[str, object]:
        rows = [item.as_dict() for item in self.discharges]
        return {
            "schema_version": PRIVACY_COMPOSITION_SCHEMA_VERSION,
            "status": self.status,
            "all_pass": self.all_pass,
            "epsilon_bound": self.epsilon_bound,
            "delta_bound": self.delta_bound,
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "failing_aspects": list(self.failing_aspects),
            "failure_summary": self.failure_summary,
            "discharges": rows,
            "composition_hash": stable_json_hash({"bounds": [self.epsilon_bound, self.delta_bound], "discharges": rows}),
        }


@dataclass(frozen=True)
class PrivacyBudgetCertifier:
    name: str = "privacy_budget"
    kind: str = "privacy"

    def handles(self, obligation: Obligation) -> bool:
        return obligation.kind == "privacy" and obligation.payload.get("constraint") == "privacy_budget"

    def certify(self, obligation: Obligation, context: Mapping[str, object] | None = None) -> CertifierVerdict:
        context = context or {}
        try:
            result = privacy_composition_from_payload(obligation.payload, context=context)
        except ValueError as exc:
            return CertifierVerdict(
                certifier=self.name,
                kind=self.kind,
                status="unknown",
                obligation_kind=obligation.kind,
                obligation_id=obligation.obligation_id,
                explanation=str(exc),
                input_hashes={"obligation": obligation.content_hash},
            )
        return CertifierVerdict(
            certifier=self.name,
            kind=self.kind,
            status=result.status,
            obligation_kind=obligation.kind,
            obligation_id=obligation.obligation_id,
            explanation="privacy budget composition admitted" if result.all_pass else result.failure_summary,
            diagnostics=(result.as_dict(),),
            input_hashes={"obligation": obligation.content_hash, "privacy_composition": result.as_dict()["composition_hash"]},  # type: ignore[index]
        )


def compose_privacy_budget(
    discharges: Sequence[PrivacyDischarge | Mapping[str, object]],
    *,
    epsilon_bound: float,
    delta_bound: float = 0.0,
) -> PrivacyCompositionResult:
    if epsilon_bound < 0 or delta_bound < 0:
        raise ValueError("privacy budget bounds must be non-negative")
    rows = tuple(_coerce_discharge(item) for item in discharges)
    return PrivacyCompositionResult(float(epsilon_bound), float(delta_bound), rows)


def privacy_composition_from_payload(
    payload: Mapping[str, object],
    *,
    context: Mapping[str, object] | None = None,
) -> PrivacyCompositionResult:
    context = context or {}
    epsilon_bound = _number(payload.get("epsilon_bound", payload.get("epsilon")), "epsilon_bound")
    delta_bound = _number(payload.get("delta_bound", payload.get("delta", 0.0)), "delta_bound")
    discharges = payload.get("privacy_discharges")
    if discharges is None:
        discharges = context.get("privacy_discharges")
    if discharges is None:
        raise ValueError("privacy budget obligation requires privacy_discharges")
    return compose_privacy_budget(_as_sequence(discharges), epsilon_bound=epsilon_bound, delta_bound=delta_bound)


def _coerce_discharge(value: PrivacyDischarge | Mapping[str, object]) -> PrivacyDischarge:
    if isinstance(value, PrivacyDischarge):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("privacy discharge must be a mapping")
    return PrivacyDischarge(
        transform=str(value.get("transform", "")),
        epsilon=_number(value.get("epsilon"), "epsilon"),
        delta=_number(value.get("delta", 0.0), "delta"),
        mechanism=str(value.get("mechanism", "redaction")),
        reference=str(value.get("reference", "")),
    )


def _number(value: object, label: str) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if out < 0:
        raise ValueError(f"{label} must be non-negative")
    return out


def _as_sequence(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


__all__ = [
    "PRIVACY_COMPOSITION_SCHEMA_VERSION",
    "PrivacyBudgetCertifier",
    "PrivacyCompositionResult",
    "PrivacyDischarge",
    "compose_privacy_budget",
    "privacy_composition_from_payload",
]
