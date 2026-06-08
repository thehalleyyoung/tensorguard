from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .certification import (
    CausalGraph,
    CausalGraphCertifier,
    CertifierVerdict,
    ProbabilisticCertifier,
    RuntimeReplayCertifier,
    StructuralCertifier,
    certifier_packet,
    combine_verdicts,
)
from .model_checking import BoundedModelCheckResult, bounded_model_check_workflow
from .obligations import Obligation, ObligationPacket, obligation_packet
from .privacy import PrivacyBudgetCertifier
from .provenance import stable_json_hash


WORKFLOW_NODE_KINDS = (
    "data_source",
    "transform",
    "model_fit",
    "metric",
    "artifact",
    "claim",
)

WORKFLOW_SCHEMA_VERSION = "datarefine.workflow.v1"
WORKFLOW_TEMPLATE_KINDS = (
    "tabular_classification",
    "regression",
    "temporal",
    "causal",
    "multi_table",
    "uncertainty",
    "llm_eval",
)
WORKFLOW_REQUIRED_SECTIONS = (
    "sources",
    "schemas",
    "splits",
    "transforms",
    "models",
    "metrics",
    "baselines",
    "ablations",
    "controls",
    "artifacts",
    "claims",
    "obligations",
)
WORKFLOW_OUTPUT_FILES = (
    "obligation_graph.json",
    "certifier_packet.json",
    "schema_summary.json",
    "lineage_graph.json",
    "metrics.json",
    "uncertainty_report.json",
    "controls.json",
    "claim_packet.json",
    "summary.md",
    "ledger.json",
)
WORKFLOW_FAILURE_CATEGORIES = (
    "unsafe-to-train",
    "unsafe-to-evaluate",
    "unsupported-claim",
    "empirical-required",
    "privacy-risk",
    "interop-warning",
)
LLM_EVAL_CONTRACT_SCHEMA_VERSION = "datarefine.llm_eval_contract.v1"

# A failure mode -> actionable repair hint table (step 260).
_REPAIR_HINTS: dict[str, dict[str, str]] = {
    "role_mismatch": {
        "diagnosis": "a column's causal role conflicts with how a transform uses it",
        "obligation_kind": "role",
        "suggested_fix": "set the column's SchemaField.role correctly or re-scope the transform's source columns",
    },
    "split_leakage": {
        "diagnosis": "rows or groups appear in more than one split",
        "obligation_kind": "split",
        "suggested_fix": "use grouped_split_refinement on the entity id and re-split with a recorded seed",
    },
    "missing_provenance": {
        "diagnosis": "a transform output has no lineage tokens back to its sources",
        "obligation_kind": "provenance",
        "suggested_fix": "apply provenance_effect with preserves_lineage=True, or name a discharge reference",
    },
    "unsafe_join": {
        "diagnosis": "a join can fan out rows or import held-out columns",
        "obligation_kind": "provenance",
        "suggested_fix": "declare join keys, assert one-to-one/one-to-many cardinality, and exclude outcome columns",
    },
    "uncertainty_drop": {
        "diagnosis": "an operation discards interval/calibration uncertainty silently",
        "obligation_kind": "uncertainty",
        "suggested_fix": "pass an uncertainty_transform or call widen_uncertainty/downgrade_uncertainty explicitly",
    },
    "nondifferentiable_transform": {
        "diagnosis": "a transform on the gradient path is nondifferentiable",
        "obligation_kind": "differentiability",
        "suggested_fix": "move the transform out of the gradient path or attach a differentiability discharge reference",
    },
    "unsupported_claim": {
        "diagnosis": "a claim cites a metric or scope that no admitted obligation covers",
        "obligation_kind": "claim_scope",
        "suggested_fix": "narrow the claim scope to admitted obligations or mark the claim partial/conjectural",
    },
}
REPAIR_FAILURE_KINDS = frozenset(_REPAIR_HINTS)


@dataclass(frozen=True)
class WorkflowManifest:
    workflow_id: str
    dataset_id: str
    split_policy: str
    steps: tuple[Mapping[str, object], ...]
    obligations: ObligationPacket | None = None

    def __init__(
        self,
        workflow_id: str,
        dataset_id: str,
        split_policy: str,
        steps: Sequence[Mapping[str, object]],
        obligations: ObligationPacket | None = None,
    ) -> None:
        object.__setattr__(self, "workflow_id", workflow_id)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "split_policy", split_policy)
        object.__setattr__(self, "steps", tuple(dict(step) for step in steps))
        object.__setattr__(self, "obligations", obligations)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "workflow_id": self.workflow_id,
            "dataset_id": self.dataset_id,
            "split_policy": self.split_policy,
            "steps": [dict(step) for step in self.steps],
            "obligations": self.obligations.as_dict() if self.obligations is not None else None,
        }


@dataclass(frozen=True)
class WorkflowValidationReport:
    schema_version: str
    workflow_id: str
    required_sections: tuple[str, ...]
    present_sections: tuple[str, ...]
    missing_sections: tuple[str, ...]
    download_policy: Mapping[str, object]
    offline_ready: bool

    @property
    def valid(self) -> bool:
        return not self.missing_sections and self.schema_version == WORKFLOW_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "required_sections": list(self.required_sections),
            "present_sections": list(self.present_sections),
            "missing_sections": list(self.missing_sections),
            "valid": self.valid,
            "download_policy": dict(self.download_policy),
            "offline_ready": self.offline_ready,
        }


@dataclass(frozen=True)
class WorkflowVerificationResult:
    manifest: Mapping[str, object]
    validation: WorkflowValidationReport
    obligations: ObligationPacket
    verdicts: tuple[CertifierVerdict, ...]
    combined: tuple[object, ...]
    output_dir: Path
    artifacts: Mapping[str, str]
    failure_groups: Mapping[str, tuple[Mapping[str, object], ...]]
    ledger: Mapping[str, object]
    model_check: BoundedModelCheckResult | None = None

    @property
    def admitted(self) -> bool:
        return not self.failing_aspects

    @property
    def failing_aspects(self) -> tuple[str, ...]:
        return tuple(category for category, rows in self.failure_groups.items() if rows)

    @property
    def exit_code(self) -> int:
        return 0 if self.admitted else 1

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "datarefine.workflow_result.v1",
            "workflow_id": str(self.manifest.get("workflow_id", "")),
            "admitted": self.admitted,
            "failing_aspects": list(self.failing_aspects),
            "artifacts": dict(self.artifacts),
            "failure_groups": {key: [dict(row) for row in rows] for key, rows in self.failure_groups.items()},
            "model_check": self.model_check.as_dict() if self.model_check else None,
            "ledger": dict(self.ledger),
        }


@dataclass(frozen=True)
class WorkflowObligationSummary:
    """Aggregated obligation status rolled up by workflow node kind (step 259)."""

    by_node: tuple[tuple[str, dict[str, object]], ...]

    @property
    def failing_aspects(self) -> tuple[str, ...]:
        return tuple(node for node, summary in self.by_node if summary.get("has_blocking"))

    @property
    def all_admitted(self) -> bool:
        return not self.failing_aspects

    def as_dict(self) -> dict[str, object]:
        nodes = {node: dict(summary) for node, summary in self.by_node}
        total = sum(int(summary["count"]) for _, summary in self.by_node)
        failing = list(self.failing_aspects)
        return {
            "schema_version": "datarefine.workflow_summary.v1",
            "total_obligations": total,
            "any_blocking": bool(failing),
            "failing_aspects": failing,
            "nodes": nodes,
        }


@dataclass(frozen=True)
class LLMEvalContractReport:
    """DataRefine-owned contract checks for an LLM evaluation dataset."""

    spec: Mapping[str, object]
    obligations: ObligationPacket

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return tuple(item.obligation_id for item in self.obligations.obligations)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": LLM_EVAL_CONTRACT_SCHEMA_VERSION,
            "spec_hash": stable_json_hash(self.spec),
            "obligation_ids": list(self.obligation_ids),
            "obligations": self.obligations.as_dict(),
        }


_BLOCKING_STATUSES = {"rejected", "blocked"}


def summarize_workflow_obligations(
    assignments: Mapping[str, Sequence[Obligation | Mapping[str, object]]],
) -> WorkflowObligationSummary:
    """Roll up obligations grouped by workflow node kind.

    ``assignments`` maps each node kind (``data_source``, ``transform``, ``model_fit``,
    ``metric``, ``artifact``, ``claim``) to the obligations attached to that node. The
    summary reports, per node, how many obligations it carries, the status histogram,
    the distinct obligation kinds, and whether any obligation is blocking. If anything
    is blocking, ``failing_aspects`` names exactly which node kinds caused it so no
    reader is left with a bare boolean.
    """
    unknown = sorted(set(assignments) - set(WORKFLOW_NODE_KINDS))
    if unknown:
        raise ValueError(f"unsupported workflow node kind(s): {', '.join(unknown)}")
    rows: list[tuple[str, dict[str, object]]] = []
    for node in WORKFLOW_NODE_KINDS:
        obligations = tuple(assignments.get(node) or ())
        status_hist: dict[str, int] = {}
        kinds: set[str] = set()
        has_blocking = False
        for item in obligations:
            status = str(item.status if isinstance(item, Obligation) else item.get("status", "unknown"))
            kind = str(item.kind if isinstance(item, Obligation) else item.get("kind", "schema"))
            status_hist[status] = status_hist.get(status, 0) + 1
            kinds.add(kind)
            if status in _BLOCKING_STATUSES:
                has_blocking = True
        rows.append(
            (
                node,
                {
                    "count": len(obligations),
                    "status_histogram": dict(sorted(status_hist.items())),
                    "kinds": sorted(kinds),
                    "has_blocking": has_blocking,
                },
            )
        )
    return WorkflowObligationSummary(tuple(rows))


def load_workflow_manifest(path: str | Path) -> dict[str, object]:
    """Load a DataRefine workflow manifest from YAML or JSON."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        if manifest_path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("workflow manifest must be a mapping")
    return {str(key): payload[key] for key in payload}


def validate_workflow_manifest(manifest: Mapping[str, object]) -> WorkflowValidationReport:
    """Validate the stable workflow.yaml schema surface for steps 271 and 277."""

    schema_version = str(manifest.get("schema_version", ""))
    present = tuple(sorted(str(key) for key in manifest if key in WORKFLOW_REQUIRED_SECTIONS))
    missing = tuple(section for section in WORKFLOW_REQUIRED_SECTIONS if section not in manifest)
    sources = _as_sequence(manifest.get("sources"))
    downloads = [_download_record(source) for source in sources if isinstance(source, Mapping)]
    policy = {
        "mode": str(manifest.get("download_policy", "offline")),
        "cache_dir": str(manifest.get("cache_dir", ".datarefine/cache")),
        "requires_opt_in": any(bool(row["download_bytes"]) for row in downloads),
        "sources": downloads,
    }
    offline_ready = all(bool(source.get("fixture") or source.get("hash")) for source in sources if isinstance(source, Mapping))
    return WorkflowValidationReport(
        schema_version=schema_version,
        workflow_id=str(manifest.get("workflow_id", "")),
        required_sections=WORKFLOW_REQUIRED_SECTIONS,
        present_sections=present,
        missing_sections=missing,
        download_policy=policy,
        offline_ready=offline_ready,
    )


def workflow_template(kind: str = "tabular_classification") -> dict[str, object]:
    """Return a complete, offline-safe starter workflow manifest."""

    if kind not in WORKFLOW_TEMPLATE_KINDS:
        raise ValueError(f"unsupported workflow template {kind!r}")
    task = kind.replace("_", "-")
    manifest = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "workflow_id": f"{task}-fixture",
        "description": f"Offline fixture workflow for {task}",
        "fixture_mode": True,
        "download_policy": "offline",
        "cache_dir": ".datarefine/cache",
        "sources": [
            {
                "id": "fixture-table",
                "kind": "inline_csv",
                "fixture": True,
                "download_bytes": 0,
                "cache": ".datarefine/cache/fixture-table.csv",
                "license": "local-fixture",
                "hash": "fixture:stable",
            }
        ],
        "schemas": [
            {"name": "x", "role": "feature", "dtype": "float32"},
            {"name": "y", "role": "outcome", "dtype": "int64"},
        ],
        "splits": {"policy": "fixture-holdout", "seed": 0, "partitions": {"train": ["r0"], "test": ["r1"]}},
        "transforms": [{"id": "scale-x", "reads": ["x"], "writes": ["x_scaled"]}],
        "models": [{"id": "fixture-model", "kind": kind}],
        "metrics": [{"id": "accuracy", "kind": "classification"}],
        "baselines": [{"id": "metadata-stripped", "description": "plain tensor without DataRefine obligations"}],
        "ablations": [{"id": "no-role-guard", "removes": "role obligation"}],
        "controls": [{"id": "safe-scale", "expected": "admitted"}],
        "artifacts": [{"id": "summary", "path": "summary.md"}],
        "claims": [{"id": "fixture-claim", "scope": "offline tutorial only", "status": "partial"}],
        "required_obligations": ["schema-columns", "split-disjoint", "no-outcome-feature"],
        "obligations": [
            {
                "id_hint": "schema-columns",
                "kind": "schema",
                "target": "fixture-table",
                "predicate": "observed columns match declared schema",
                "status": "unknown",
                "constraint": "schema_consistency",
                "declared_columns": ["x", "y"],
                "observed_columns": ["x", "y"],
                "failure_category": "unsafe-to-train",
            },
            {
                "id_hint": "split-disjoint",
                "kind": "split",
                "target": "fixture-table",
                "predicate": "train and test row ids are disjoint",
                "status": "unknown",
                "constraint": "split_disjointness",
                "partitions": {"train": ["r0"], "test": ["r1"]},
                "failure_category": "unsafe-to-evaluate",
            },
            {
                "id_hint": "no-outcome-feature",
                "kind": "role",
                "target": "scale-x",
                "predicate": "model features do not read outcome columns",
                "status": "unknown",
                "constraint": "no_outcome_in_feature",
                "feature_sources": ["x"],
                "outcome_columns": ["y"],
                "failure_category": "unsafe-to-train",
            },
            {
                "id_hint": "runtime-replay",
                "kind": "provenance",
                "target": "fixture-run",
                "predicate": "fixture verification replay is recorded",
                "status": "unknown",
                "constraint": "runtime_replay",
                "failure_category": "interop-warning",
            },
        ],
        "replay": {
            "command": "python -m datarefine.cli verify workflow.yaml --fixture-mode",
            "fixture_hashes": {"fixture-table": "fixture:stable"},
            "observed_violations": [],
            "metric_keys": ["accuracy"],
            "acceptance": True,
            "environment": {"mode": "fixture"},
        },
    }
    if kind == "llm_eval":
        llm_eval = _fixture_llm_eval_spec()
        report = llm_eval_contract_report(llm_eval)
        manifest.update(
            {
                "llm_eval": llm_eval,
                "obligations": [item.as_dict() for item in report.obligations.obligations],
                "required_obligations": [
                    "llm-eval-split-disjoint",
                    "llm-eval-prompt-label-roles",
                    "llm-eval-rubric-provenance",
                    "llm-eval-contamination-check",
                    "llm-eval-metric-claim-scope",
                ],
                "claims": [
                    {
                        "id": "fixture-llm-eval-claim",
                        "scope": "offline LLM-eval fixture only",
                        "status": "partial",
                        "metric_ids": ["exact_match"],
                    }
                ],
                "metrics": [{"id": "exact_match", "kind": "llm_eval"}],
                "models": [{"id": "fixture-model", "kind": "llm_eval"}],
            }
        )
    return manifest


def write_workflow_template(kind: str, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(workflow_template(kind), handle, sort_keys=False)
    return target


def verify_workflow(
    manifest_or_path: Mapping[str, object] | str | Path,
    *,
    output_dir: str | Path | None = None,
    fixture_mode: bool = False,
    offline: bool = False,
    allow_download: bool = False,
) -> WorkflowVerificationResult:
    """Verify a workflow manifest and write standardized product artifacts."""

    manifest = load_workflow_manifest(manifest_or_path) if isinstance(manifest_or_path, (str, Path)) else dict(manifest_or_path)
    validation = validate_workflow_manifest(manifest)
    if validation.schema_version != WORKFLOW_SCHEMA_VERSION:
        raise ValueError(f"unsupported workflow schema_version {validation.schema_version!r}")
    if validation.missing_sections:
        raise ValueError(f"workflow manifest missing sections: {', '.join(validation.missing_sections)}")
    if offline and not validation.offline_ready:
        raise ValueError("offline verification requires fixture or hash metadata for every source")
    if not allow_download and _requires_download(manifest):
        raise ValueError("workflow declares downloadable sources; pass allow_download=True after reviewing sizes/licenses")

    obligations = obligation_packet(*_manifest_obligations(manifest), metadata={"workflow_id": validation.workflow_id})
    context = _certifier_context(manifest, fixture_mode=fixture_mode or bool(manifest.get("fixture_mode")))
    certifiers = (
        StructuralCertifier(),
        CausalGraphCertifier(),
        PrivacyBudgetCertifier(),
        ProbabilisticCertifier(),
        RuntimeReplayCertifier(),
    )
    verdicts: list[CertifierVerdict] = []
    combined = []
    for item in obligations.obligations:
        item_verdicts = [certifier.certify(item, context) for certifier in certifiers if certifier.handles(item)]
        if not item_verdicts:
            item_verdicts = [StructuralCertifier().certify(item, context)]
        verdicts.extend(item_verdicts)
        combined.append(combine_verdicts(item_verdicts))

    out_dir = Path(output_dir or manifest.get("output_dir") or "datarefine_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_check = bounded_model_check_workflow(manifest)
    failure_groups = _failure_groups(manifest, obligations, combined, model_check=model_check)
    ledger = _workflow_ledger(manifest, obligations, combined, failure_groups, model_check=model_check)
    artifacts = _write_workflow_artifacts(out_dir, manifest, validation, obligations, verdicts, combined, failure_groups, ledger, model_check)
    return WorkflowVerificationResult(
        manifest=manifest,
        validation=validation,
        obligations=obligations,
        verdicts=tuple(verdicts),
        combined=tuple(combined),
        output_dir=out_dir,
        artifacts=artifacts,
        failure_groups=failure_groups,
        ledger=ledger,
        model_check=model_check,
    )


def llm_eval_contract_report(spec: Mapping[str, object]) -> LLMEvalContractReport:
    """Build the DataRefine obligations an LLM-eval dataset must satisfy."""

    normalized = _normalize_llm_eval_spec(spec)
    return LLMEvalContractReport(
        spec=normalized,
        obligations=obligation_packet(*_llm_eval_obligation_rows(normalized), metadata={"contract": "llm_eval"}),
    )


def verify_llm_eval_contract(
    manifest_or_spec: Mapping[str, object],
    *,
    output_dir: str | Path | None = None,
    fixture_mode: bool = True,
    offline: bool = True,
) -> WorkflowVerificationResult:
    """Verify an LLM-eval data contract before downstream PromptABI consumption."""

    if manifest_or_spec.get("schema_version") == WORKFLOW_SCHEMA_VERSION:
        manifest = dict(manifest_or_spec)
        spec = manifest.get("llm_eval")
        if not isinstance(spec, Mapping):
            raise ValueError("LLM-eval workflow manifest must include an llm_eval mapping")
    else:
        spec = manifest_or_spec
        manifest = workflow_template("llm_eval")
    report = llm_eval_contract_report(spec)
    manifest["llm_eval"] = dict(report.spec)
    manifest["obligations"] = [item.as_dict() for item in report.obligations.obligations]
    manifest["required_obligations"] = [str(item.payload["id_hint"]) for item in report.obligations.obligations]
    return verify_workflow(manifest, output_dir=output_dir, fixture_mode=fixture_mode, offline=offline)


def inspect_packet(path: str | Path) -> dict[str, object]:
    """Summarize verifier packets for CLI inspection."""

    packet_path = Path(path)
    with packet_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("packet must contain a JSON object")
    summary: dict[str, object] = {"path": str(packet_path), "schema_version": payload.get("schema_version")}
    if "obligations" in payload:
        obligations = _as_sequence(payload.get("obligations"))
        summary.update(
            {
                "packet_kind": "obligation_graph",
                "obligation_count": len(obligations),
                "statuses": _histogram(str(row.get("status", "unknown")) for row in obligations if isinstance(row, Mapping)),
                "hashes": [str(row.get("content_hash", "")) for row in obligations if isinstance(row, Mapping)],
                "limitations": sorted({str(limit) for row in obligations if isinstance(row, Mapping) for limit in _as_sequence(row.get("limitations"))}),
            }
        )
    elif "combined" in payload:
        combined = _as_sequence(payload.get("combined"))
        summary.update(
            {
                "packet_kind": "certifier_packet",
                "verdict_count": len(_as_sequence(payload.get("verdicts"))),
                "combined_statuses": _histogram(str(row.get("status", "unknown")) for row in combined if isinstance(row, Mapping)),
                "replay_commands": _replay_commands(payload),
                "hashes": _packet_hashes(payload),
            }
        )
    elif "claims" in payload:
        claims = _as_sequence(payload.get("claims"))
        summary.update(
            {
                "packet_kind": "claim_packet",
                "claim_count": len(claims),
                "claim_scopes": [str(row.get("scope", "")) for row in claims if isinstance(row, Mapping)],
                "limitations": list(_as_sequence(payload.get("limitations"))),
            }
        )
    else:
        summary["packet_kind"] = "unknown"
    return summary


@dataclass(frozen=True)
class RepairHint:
    failure_kind: str
    diagnosis: str
    suggested_fix: str
    obligation_kind: str

    def as_dict(self) -> dict[str, object]:
        return {
            "failure_kind": self.failure_kind,
            "diagnosis": self.diagnosis,
            "suggested_fix": self.suggested_fix,
            "obligation_kind": self.obligation_kind,
        }


def repair_hint(failure_kind: str) -> RepairHint:
    """Return an actionable repair hint for a common assumption failure (step 260)."""
    if failure_kind not in _REPAIR_HINTS:
        raise ValueError(f"unknown failure kind {failure_kind!r}")
    spec = _REPAIR_HINTS[failure_kind]
    return RepairHint(
        failure_kind=failure_kind,
        diagnosis=spec["diagnosis"],
        suggested_fix=spec["suggested_fix"],
        obligation_kind=spec["obligation_kind"],
    )


def repair_hints() -> tuple[RepairHint, ...]:
    return tuple(repair_hint(kind) for kind in _REPAIR_HINTS)


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


def _download_record(source: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": str(source.get("id", "")),
        "url": source.get("url"),
        "download_bytes": int(source.get("download_bytes") or 0),
        "cache": str(source.get("cache", "")),
        "license": str(source.get("license", "")),
        "fixture": bool(source.get("fixture", False)),
    }


def _requires_download(manifest: Mapping[str, object]) -> bool:
    if str(manifest.get("download_policy", "offline")) == "offline":
        return False
    return any(isinstance(source, Mapping) and int(source.get("download_bytes") or 0) > 0 for source in _as_sequence(manifest.get("sources")))


def _manifest_obligations(manifest: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    rows = _as_sequence(manifest.get("obligations"))
    if not rows:
        raise ValueError("workflow manifest must declare at least one obligation")
    out = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("workflow obligations must be mappings")
        payload = dict(row)
        out.append(payload)
    return tuple(out)


def _certifier_context(manifest: Mapping[str, object], *, fixture_mode: bool) -> dict[str, object]:
    context: dict[str, object] = {
        "evidence": manifest.get("evidence"),
        "replay": manifest.get("replay") or _fixture_replay(manifest) if fixture_mode else manifest.get("replay"),
        "privacy_discharges": manifest.get("privacy_discharges"),
    }
    graph = manifest.get("causal_graph")
    if isinstance(graph, Mapping):
        context["causal_graph"] = CausalGraph(
            roles={str(k): str(v) for k, v in dict(graph.get("roles") or {}).items()},
            edges=tuple((str(src), str(dst)) for src, dst in _as_sequence(graph.get("edges"))),
            adjustment_set=tuple(str(item) for item in _as_sequence(graph.get("adjustment_set"))),
        )
    return context


def _fixture_replay(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "command": "python -m datarefine.cli verify workflow.yaml --fixture-mode",
        "fixture_hashes": {str(source.get("id", "fixture")): str(source.get("hash", "fixture:stable")) for source in _as_sequence(manifest.get("sources")) if isinstance(source, Mapping)},
        "observed_violations": [],
        "metric_keys": [str(metric.get("id", metric.get("kind", "metric"))) for metric in _as_sequence(manifest.get("metrics")) if isinstance(metric, Mapping)],
        "acceptance": True,
        "environment": {"mode": "fixture"},
    }


def _fixture_llm_eval_spec() -> dict[str, object]:
    return {
        "prompt_column": "prompt_text",
        "label_column": "expected_label",
        "prompt_role": "prompt",
        "label_role": "label",
        "train_prompt_hashes": ["prompt:train:0", "prompt:train:1"],
        "eval_prompt_hashes": ["prompt:eval:0", "prompt:eval:1"],
        "training_corpus_hashes": ["prompt:train:0", "prompt:train:1", "prompt:background:0"],
        "rubric_id": "rubric:helpfulness-v1",
        "rubric_source_hash": "rubric:sha256:fixture",
        "metric_ids": ["exact_match"],
        "claim_metric_ids": ["exact_match"],
    }


def _normalize_llm_eval_spec(spec: Mapping[str, object]) -> dict[str, object]:
    normalized = {
        "prompt_column": str(spec.get("prompt_column", "prompt")),
        "label_column": str(spec.get("label_column", "label")),
        "prompt_role": str(spec.get("prompt_role", "prompt")),
        "label_role": str(spec.get("label_role", "label")),
        "train_prompt_hashes": [str(item) for item in _as_sequence(spec.get("train_prompt_hashes"))],
        "eval_prompt_hashes": [str(item) for item in _as_sequence(spec.get("eval_prompt_hashes"))],
        "training_corpus_hashes": [str(item) for item in _as_sequence(spec.get("training_corpus_hashes"))],
        "rubric_id": str(spec.get("rubric_id", "rubric")),
        "rubric_source_hash": str(spec.get("rubric_source_hash", "")),
        "metric_ids": [str(item) for item in _as_sequence(spec.get("metric_ids"))],
        "claim_metric_ids": [str(item) for item in _as_sequence(spec.get("claim_metric_ids"))],
    }
    if not normalized["training_corpus_hashes"]:
        normalized["training_corpus_hashes"] = list(normalized["train_prompt_hashes"])
    return normalized


def _llm_eval_obligation_rows(spec: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    train_hashes = list(_as_sequence(spec.get("train_prompt_hashes")))
    eval_hashes = list(_as_sequence(spec.get("eval_prompt_hashes")))
    training_corpus_hashes = list(_as_sequence(spec.get("training_corpus_hashes")))
    metric_ids = list(_as_sequence(spec.get("metric_ids")))
    claim_metric_ids = list(_as_sequence(spec.get("claim_metric_ids")))
    return (
        {
            "id_hint": "llm-eval-split-disjoint",
            "kind": "split",
            "target": "llm_eval.prompt_splits",
            "predicate": "training prompts and evaluation prompts are disjoint",
            "constraint": "split_disjointness",
            "partitions": {"train": train_hashes, "eval": eval_hashes},
            "failure_category": "unsafe-to-evaluate",
        },
        {
            "id_hint": "llm-eval-prompt-label-roles",
            "kind": "role",
            "target": "llm_eval.prompt_labels",
            "predicate": "prompt text and expected labels have declared LLM-eval roles",
            "constraint": "role_constraint",
            "roles": [spec.get("prompt_role"), spec.get("label_role")],
            "allowed_roles": ["prompt", "label"],
            "failure_category": "unsafe-to-evaluate",
        },
        {
            "id_hint": "llm-eval-rubric-provenance",
            "kind": "lineage",
            "target": "llm_eval.rubric",
            "predicate": "rubric has source provenance before PromptABI references it",
            "constraint": "column_lineage",
            "lineage": {str(spec.get("rubric_id")): [str(spec.get("rubric_source_hash", ""))] if spec.get("rubric_source_hash") else []},
            "failure_category": "interop-warning",
        },
        {
            "id_hint": "llm-eval-contamination-check",
            "kind": "split",
            "target": "llm_eval.contamination",
            "predicate": "evaluation prompt hashes are absent from the declared training corpus",
            "constraint": "split_disjointness",
            "partitions": {"training_corpus": training_corpus_hashes, "eval_prompts": eval_hashes},
            "failure_category": "unsafe-to-evaluate",
        },
        {
            "id_hint": "llm-eval-metric-claim-scope",
            "kind": "claim_scope",
            "target": "llm_eval.metric_claims",
            "predicate": "metric claims cite only metrics declared for the LLM-eval workflow",
            "constraint": "role_constraint",
            "roles": claim_metric_ids,
            "allowed_roles": metric_ids,
            "failure_category": "unsupported-claim",
        },
    )


def _required_hints(manifest: Mapping[str, object]) -> set[str]:
    return {str(item) for item in _as_sequence(manifest.get("required_obligations"))}


def _failure_groups(
    manifest: Mapping[str, object],
    obligations: ObligationPacket,
    combined: Sequence[object],
    *,
    model_check: BoundedModelCheckResult | None = None,
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    hints = _required_hints(manifest)
    rows: dict[str, list[Mapping[str, object]]] = {category: [] for category in WORKFLOW_FAILURE_CATEGORIES}
    by_id = {item.obligation_id: item for item in obligations.obligations}
    for verdict in combined:
        status = str(getattr(verdict, "status"))
        obligation_id = str(getattr(verdict, "obligation_id"))
        obligation = by_id[obligation_id]
        hint = str(obligation.payload.get("id_hint", ""))
        is_required = not hints or hint in hints or obligation_id in hints
        if not is_required or status == "admitted":
            continue
        category = _failure_category(obligation, status)
        rows[category].append(
            {
                "obligation_id": obligation_id,
                "id_hint": hint,
                "kind": obligation.kind,
                "target": obligation.target,
                "status": status,
                "explanation": str(getattr(verdict, "explanation")),
            }
        )
    if model_check is not None and not model_check.all_pass and model_check.first_violation is not None:
        violation = model_check.first_violation
        category = "unsafe-to-evaluate" if violation.invariant == "split_disjointness" else "unsafe-to-train"
        rows[category].append(
            {
                "obligation_id": "workflow-bmc",
                "id_hint": "workflow-bmc",
                "kind": "workflow",
                "target": violation.step_id,
                "status": "rejected",
                "explanation": violation.message,
                "invariant": violation.invariant,
                "counterexample": dict(violation.counterexample),
            }
        )
    return {category: tuple(items) for category, items in rows.items()}


def _failure_category(obligation: Obligation, status: str) -> str:
    explicit = str(obligation.payload.get("failure_category", ""))
    if explicit in WORKFLOW_FAILURE_CATEGORIES:
        return explicit
    if status == "empirical-required":
        return "empirical-required"
    if obligation.kind == "claim_scope":
        return "unsupported-claim"
    if obligation.kind == "privacy":
        return "privacy-risk"
    if obligation.kind in {"schema", "role", "lineage"}:
        return "unsafe-to-train"
    if obligation.kind == "split":
        return "unsafe-to-evaluate"
    return "interop-warning"


def _workflow_ledger(
    manifest: Mapping[str, object],
    obligations: ObligationPacket,
    combined: Sequence[object],
    failure_groups: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    model_check: BoundedModelCheckResult | None = None,
) -> dict[str, object]:
    failing = [category for category, rows in failure_groups.items() if rows]
    input_hashes = {"manifest": stable_json_hash(manifest), "obligations": obligations.content_hash}
    return {
        "_provenance": {
            "generator": "datarefine.workflows.verify_workflow",
            "input_hashes": input_hashes,
            "env_hash": stable_json_hash({"python": sys.version.split()[0], "platform": platform.platform()}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "workflow": {
            "schema_defined": True,
            "verify_command": True,
            "inspect_command": True,
            "templates": list(WORKFLOW_TEMPLATE_KINDS),
            "llm_eval_data_contract": "llm_eval" in manifest,
            "failure_categories": list(WORKFLOW_FAILURE_CATEGORIES),
            "standard_artifacts": list(WORKFLOW_OUTPUT_FILES),
            "offline_fixture_mode": True,
            "download_policy": True,
            "ledger_provenance": True,
            "failure_ledger": bool(failing),
            "all_pass": not failing,
            "failing_aspects": failing,
            "combined_statuses": _histogram(str(getattr(item, "status")) for item in combined),
            "bounded_model_check": model_check.as_dict() if model_check else None,
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_workflow_artifacts(
    out_dir: Path,
    manifest: Mapping[str, object],
    validation: WorkflowValidationReport,
    obligations: ObligationPacket,
    verdicts: Sequence[CertifierVerdict],
    combined: Sequence[object],
    failure_groups: Mapping[str, Sequence[Mapping[str, object]]],
    ledger: Mapping[str, object],
    model_check: BoundedModelCheckResult | None = None,
) -> Mapping[str, str]:
    artifacts: dict[str, str] = {}
    payloads: dict[str, Mapping[str, object]] = {
        "obligation_graph.json": obligations.as_dict(),
        "certifier_packet.json": certifier_packet(verdicts, combined=combined),  # type: ignore[arg-type]
        "schema_summary.json": {"schema_version": "datarefine.schema_summary.v1", "validation": validation.as_dict(), "schemas": list(_as_sequence(manifest.get("schemas")))},
        "lineage_graph.json": {"schema_version": "datarefine.lineage_graph.v1", "sources": list(_as_sequence(manifest.get("sources"))), "transforms": list(_as_sequence(manifest.get("transforms"))), "artifacts": list(_as_sequence(manifest.get("artifacts")))},
        "metrics.json": {"schema_version": "datarefine.metrics.v1", "metrics": list(_as_sequence(manifest.get("metrics"))), "baselines": list(_as_sequence(manifest.get("baselines"))), "ablations": list(_as_sequence(manifest.get("ablations")))},
        "uncertainty_report.json": {"schema_version": "datarefine.uncertainty_report.v1", "uncertainty": manifest.get("uncertainty", {}), "empirical_required": list(failure_groups.get("empirical-required", ()))},
        "controls.json": {"schema_version": "datarefine.controls.v1", "controls": list(_as_sequence(manifest.get("controls")))},
        "claim_packet.json": {"schema_version": "datarefine.claim_packet.v1", "claims": list(_as_sequence(manifest.get("claims"))), "limitations": list(_as_sequence(manifest.get("limitations")))},
        "ledger.json": ledger,
    }
    if model_check is not None:
        payloads["model_check.json"] = model_check.as_dict()
    for name, payload in payloads.items():
        path = out_dir / name
        _write_json(path, payload)
        artifacts[name] = str(path)
    summary = _summary_markdown(manifest, combined, failure_groups, artifacts)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    artifacts["summary.md"] = str(out_dir / "summary.md")
    if any(failure_groups.values()):
        failure_ledger = {
            "_provenance": dict(ledger["_provenance"]),  # type: ignore[index]
            "failures": {key: [dict(row) for row in rows] for key, rows in failure_groups.items() if rows},
            "failure_summary": "; ".join(f"{key}: {len(rows)}" for key, rows in failure_groups.items() if rows),
        }
        _write_json(out_dir / "failure_ledger.json", failure_ledger)
        artifacts["failure_ledger.json"] = str(out_dir / "failure_ledger.json")
    return artifacts


def _summary_markdown(
    manifest: Mapping[str, object],
    combined: Sequence[object],
    failure_groups: Mapping[str, Sequence[Mapping[str, object]]],
    artifacts: Mapping[str, str],
) -> str:
    failing = [category for category, rows in failure_groups.items() if rows]
    lines = [
        f"# DataRefine workflow summary: {manifest.get('workflow_id', 'workflow')}",
        "",
        f"admitted: {not failing}",
        f"failing_aspects: {', '.join(failing) if failing else 'none'}",
        "",
        "## Required obligation results",
    ]
    for verdict in combined:
        lines.append(f"- {getattr(verdict, 'obligation_id')}: {getattr(verdict, 'status')} ({getattr(verdict, 'decided_by')})")
    lines.extend(["", "## Failure groups"])
    for category in WORKFLOW_FAILURE_CATEGORIES:
        rows = failure_groups.get(category, ())
        lines.append(f"- {category}: {len(rows)}")
    lines.extend(["", "## Artifacts"])
    for name in WORKFLOW_OUTPUT_FILES:
        if name in artifacts:
            lines.append(f"- {name}: {os.path.basename(artifacts[name])}")
    return "\n".join(lines) + "\n"


def _histogram(values: Sequence[str] | object) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:  # type: ignore[union-attr]
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def _replay_commands(payload: Mapping[str, object]) -> list[str]:
    commands: list[str] = []
    for verdict in _as_sequence(payload.get("verdicts")):
        if not isinstance(verdict, Mapping):
            continue
        for diagnostic in _as_sequence(verdict.get("diagnostics")):
            if isinstance(diagnostic, Mapping) and diagnostic.get("command"):
                commands.append(str(diagnostic["command"]))
    return sorted(set(commands))


def _packet_hashes(payload: Mapping[str, object]) -> list[str]:
    hashes: set[str] = set()
    for verdict in _as_sequence(payload.get("verdicts")):
        if isinstance(verdict, Mapping):
            hashes.update(str(value) for value in dict(verdict.get("input_hashes") or {}).values())
    return sorted(hashes)


__all__ = [
    "REPAIR_FAILURE_KINDS",
    "WORKFLOW_FAILURE_CATEGORIES",
    "LLM_EVAL_CONTRACT_SCHEMA_VERSION",
    "WORKFLOW_NODE_KINDS",
    "WORKFLOW_OUTPUT_FILES",
    "WORKFLOW_REQUIRED_SECTIONS",
    "WORKFLOW_SCHEMA_VERSION",
    "WORKFLOW_TEMPLATE_KINDS",
    "LLMEvalContractReport",
    "RepairHint",
    "WorkflowManifest",
    "WorkflowObligationSummary",
    "WorkflowValidationReport",
    "WorkflowVerificationResult",
    "inspect_packet",
    "llm_eval_contract_report",
    "load_workflow_manifest",
    "repair_hint",
    "repair_hints",
    "summarize_workflow_obligations",
    "validate_workflow_manifest",
    "verify_workflow",
    "verify_llm_eval_contract",
    "workflow_template",
    "write_workflow_template",
]
