from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .provenance import stable_json_hash


WORKFLOW_BMC_SCHEMA_VERSION = "datarefine.workflow_bmc.v1"
INVARIANTS = (
    "split_disjointness",
    "role_consistency",
    "no_leakage",
    "provenance_continuity",
    "claim_scope_monotonicity",
)


@dataclass(frozen=True)
class WorkflowInvariantViolation:
    invariant: str
    step_index: int
    step_id: str
    message: str
    counterexample: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "invariant": self.invariant,
            "step_index": self.step_index,
            "step_id": self.step_id,
            "message": self.message,
            "counterexample": dict(self.counterexample),
        }


@dataclass(frozen=True)
class BoundedModelCheckResult:
    checked_depth: int
    invariants: tuple[str, ...]
    first_violation: WorkflowInvariantViolation | None
    trace: tuple[Mapping[str, object], ...]

    @property
    def all_pass(self) -> bool:
        return self.first_violation is None

    @property
    def status(self) -> str:
        return "admitted" if self.all_pass else "rejected"

    @property
    def failing_aspects(self) -> tuple[str, ...]:
        if self.first_violation is None:
            return ()
        return (self.first_violation.invariant,)

    @property
    def failure_summary(self) -> str:
        if self.first_violation is None:
            return ""
        return self.first_violation.message

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": WORKFLOW_BMC_SCHEMA_VERSION,
            "status": self.status,
            "all_pass": self.all_pass,
            "checked_depth": self.checked_depth,
            "invariants": list(self.invariants),
            "failing_aspects": list(self.failing_aspects),
            "failure_summary": self.failure_summary,
            "first_violation": self.first_violation.as_dict() if self.first_violation else None,
            "trace": [dict(row) for row in self.trace],
            "trace_hash": stable_json_hash({"trace": [dict(row) for row in self.trace]}),
        }


def bounded_model_check_workflow(
    manifest: Mapping[str, object],
    *,
    max_depth: int | None = None,
) -> BoundedModelCheckResult:
    steps = tuple(row for row in _as_sequence(manifest.get("transforms")) if isinstance(row, Mapping))
    depth = min(len(steps), max_depth if max_depth is not None else len(steps))
    role_by_column = _role_map(manifest)
    known_columns = set(role_by_column)
    split_violation = _split_violation(manifest)
    trace: list[Mapping[str, object]] = []

    for index in range(depth + 1):
        if index == 0:
            step = {"id": "<splits>", "reads": [], "writes": []}
        else:
            step = dict(steps[index - 1])
        step_id = str(step.get("id", f"step-{index}"))

        if split_violation is not None:
            violation = WorkflowInvariantViolation(
                "split_disjointness",
                index,
                step_id,
                f"split partitions overlap at row {split_violation['row_id']}",
                split_violation,
            )
            return BoundedModelCheckResult(depth, INVARIANTS, violation, tuple(trace))

        reads = [str(item) for item in _as_sequence(step.get("reads"))]
        writes = [str(item) for item in _as_sequence(step.get("writes"))]
        features = [str(item) for item in _as_sequence(step.get("features") or reads)]
        unknown_reads = sorted(col for col in reads if col not in known_columns)
        if unknown_reads:
            violation = WorkflowInvariantViolation(
                "provenance_continuity",
                index,
                step_id,
                f"step {step_id} reads columns with no prior schema or transform source",
                {"unknown_reads": unknown_reads, "known_columns": sorted(known_columns)},
            )
            return BoundedModelCheckResult(depth, INVARIANTS, violation, tuple(trace))
        bad_roles = sorted(col for col in reads if role_by_column.get(col) == "id" and str(step.get("allow_id_reads", "")).lower() != "true")
        if bad_roles:
            violation = WorkflowInvariantViolation(
                "role_consistency",
                index,
                step_id,
                f"step {step_id} consumes id columns as ordinary transform inputs",
                {"columns": bad_roles, "roles": {col: role_by_column[col] for col in bad_roles}},
            )
            return BoundedModelCheckResult(depth, INVARIANTS, violation, tuple(trace))
        leaked = sorted(col for col in features if role_by_column.get(col) == "outcome")
        if leaked:
            violation = WorkflowInvariantViolation(
                "no_leakage",
                index,
                step_id,
                f"step {step_id} exposes outcome columns as model features",
                {"outcome_features": leaked, "features": features},
            )
            return BoundedModelCheckResult(depth, INVARIANTS, violation, tuple(trace))

        for col in writes:
            known_columns.add(col)
            role_by_column.setdefault(col, str(step.get("role", "feature")))
        trace.append({"step_index": index, "step_id": step_id, "known_columns": sorted(known_columns)})

    claim_violation = _claim_scope_violation(manifest)
    if claim_violation is not None:
        violation = WorkflowInvariantViolation(
            "claim_scope_monotonicity",
            depth,
            "<claims>",
            "claims reference metrics outside the manifest metric scope",
            claim_violation,
        )
        return BoundedModelCheckResult(depth, INVARIANTS, violation, tuple(trace))
    return BoundedModelCheckResult(depth, INVARIANTS, None, tuple(trace))


def _role_map(manifest: Mapping[str, object]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for row in _as_sequence(manifest.get("schemas")):
        if isinstance(row, Mapping):
            name = str(row.get("name", ""))
            if name:
                roles[name] = str(row.get("role", "feature"))
    return roles


def _split_violation(manifest: Mapping[str, object]) -> Mapping[str, object] | None:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        return None
    partitions = splits.get("partitions")
    if not isinstance(partitions, Mapping):
        return None
    owner: dict[str, str] = {}
    for name, ids in partitions.items():
        for row_id in _as_sequence(ids):
            rid = str(row_id)
            part = str(name)
            if rid in owner and owner[rid] != part:
                return {"row_id": rid, "partitions": sorted([owner[rid], part])}
            owner[rid] = part
    return None


def _claim_scope_violation(manifest: Mapping[str, object]) -> Mapping[str, object] | None:
    metric_ids = {str(row.get("id", row.get("kind", ""))) for row in _as_sequence(manifest.get("metrics")) if isinstance(row, Mapping)}
    for claim in _as_sequence(manifest.get("claims")):
        if not isinstance(claim, Mapping):
            continue
        used = {str(item) for item in _as_sequence(claim.get("metric_ids"))}
        unknown = sorted(used - metric_ids)
        if unknown:
            return {"claim_id": str(claim.get("id", "")), "unknown_metric_ids": unknown, "declared_metric_ids": sorted(metric_ids)}
    return None


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
    "WORKFLOW_BMC_SCHEMA_VERSION",
    "INVARIANTS",
    "BoundedModelCheckResult",
    "WorkflowInvariantViolation",
    "bounded_model_check_workflow",
]
