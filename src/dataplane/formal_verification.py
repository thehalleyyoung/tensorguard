from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Mapping, Sequence

from .certification import (
    CertifierVerdict,
    StructuralCertifier,
    _decide_structural,
    lower_obligation,
    solver_diagnostics,
)
from .obligations import Obligation
from .provenance import stable_json_hash
from .smt_backend import recheck
from .workflows import verify_workflow

FORMAL_BOUNDARY_SCHEMA_VERSION = "datarefine.formal_boundary.v1"
DETERMINISM_SCHEMA_VERSION = "datarefine.solver_determinism.v1"
OVERHEAD_SCHEMA_VERSION = "datarefine.formal_overhead.v1"
PROOF_PACKET_SCHEMA_VERSION = "datarefine.proof_packet.v1"
PROOF_PACKET_REQUIRED_KEYS = (
    "schema_version",
    "producer",
    "obligation_id",
    "obligation_hash",
    "lowered_formula",
    "verdict",
    "proof_object",
    "proof_recheck",
    "solver_identity",
    "env_hash",
    "admitted_scope",
    "packet_hash",
)

DECIDABLE_COMPLETE_FAMILIES = (
    "schema_consistency",
    "role_constraint",
    "split_disjointness",
    "column_lineage",
    "join_safety",
    "bounds",
    "no_outcome_in_feature",
    "fit_transform_isolation",
    "partition_lengths",
)
SOUND_INCOMPLETE_FAMILIES = (
    "z3_differential_cross_check",
    "bounded_workflow_model_check",
    "causal_graph_warning",
)
EMPIRICAL_REQUIRED_FAMILIES = (
    "statistical_significance",
    "calibration",
    "causal_effect",
    "benchmark_utility",
    "coverage",
    "bootstrap_interval",
    "empirical_confidence",
)
OUT_OF_SCOPE_FAMILIES = (
    "arbitrary_model_code_semantics",
    "causal_validity_from_metadata_alone",
    "external_prompt_or_tensor_contract_analysis",
)


def formal_verification_boundary() -> dict[str, object]:
    """Return the admitted, unknown, empirical, and out-of-scope boundary."""

    return {
        "schema_version": FORMAL_BOUNDARY_SCHEMA_VERSION,
        "sound_and_complete": list(DECIDABLE_COMPLETE_FAMILIES),
        "sound_but_incomplete": list(SOUND_INCOMPLETE_FAMILIES),
        "empirical_required": list(EMPIRICAL_REQUIRED_FAMILIES),
        "out_of_scope": list(OUT_OF_SCOPE_FAMILIES),
        "solver": solver_diagnostics(),
        "follow_up_steps": {
            "sound_but_incomplete": [
                "tighten bounded workflow model checking depth policy",
                "surface z3 unknown reasons in all workflow packets",
            ],
            "empirical_required": [
                "route calibration and coverage claims through registered studies",
                "keep benchmark-utility claims ledger-backed",
            ],
            "out_of_scope": [
                "delegate PromptABI and TensorGuard owned checks to those tools",
            ],
        },
    }


def _solver_diagnostic(verdict: CertifierVerdict) -> Mapping[str, object]:
    for diagnostic in verdict.diagnostics:
        if isinstance(diagnostic, Mapping) and "formula_hash" in diagnostic:
            return diagnostic
    return {}


def _determinism_signature(verdict: CertifierVerdict) -> dict[str, object]:
    diagnostic = _solver_diagnostic(verdict)
    return {
        "status": verdict.status,
        "obligation_id": verdict.obligation_id,
        "model": diagnostic.get("model"),
        "unsat_core": diagnostic.get("unsat_core"),
        "formula_hash": diagnostic.get("formula_hash"),
        "env_hash": diagnostic.get("env_hash"),
    }


def rerun_solver_determinism(
    obligation: Obligation,
    *,
    runs: int = 2,
    certifier: StructuralCertifier | None = None,
) -> dict[str, object]:
    """Re-run admission and fail if verdict/model/core changes."""

    if runs < 2:
        raise ValueError("determinism rerun requires at least two runs")
    certifier = certifier or StructuralCertifier()
    verdicts = [certifier.certify(obligation) for _ in range(runs)]
    signatures = [_determinism_signature(verdict) for verdict in verdicts]
    baseline = signatures[0]
    mismatches = [
        {"run": index, "baseline": baseline, "observed": signature}
        for index, signature in enumerate(signatures[1:], start=1)
        if signature != baseline
    ]
    return {
        "schema_version": DETERMINISM_SCHEMA_VERSION,
        "all_pass": not mismatches,
        "failing_aspects": ["solver_determinism"] if mismatches else [],
        "failure_summary": "solver verdict/model/core changed between reruns" if mismatches else "",
        "runs": runs,
        "signature": baseline,
        "mismatches": mismatches,
    }


@dataclass(frozen=True)
class _TimedResult:
    elapsed_s: float
    peak_bytes: int
    value: object

    def as_dict(self) -> dict[str, object]:
        return {"elapsed_s": self.elapsed_s, "peak_bytes": self.peak_bytes}


def _time_call(fn) -> _TimedResult:
    tracemalloc.start()
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _TimedResult(elapsed_s=elapsed, peak_bytes=int(peak), value=value)


def benchmark_formal_overhead(obligations: Sequence[Obligation]) -> dict[str, object]:
    """Measure validation, lowering, single-solver, differential, and re-check overhead."""

    rows: list[dict[str, object]] = []
    for obligation in obligations:
        lowered = _time_call(lambda: lower_obligation(obligation))
        formula = lowered.value
        validation = _time_call(lambda: _decide_structural(formula.constraint, formula.payload))  # type: ignore[attr-defined]
        single = _time_call(lambda: StructuralCertifier(differential=False).certify(obligation))
        dual = _time_call(lambda: StructuralCertifier(differential=True).certify(obligation))
        proof = _solver_diagnostic(dual.value)  # type: ignore[arg-type]
        proof_check = _time_call(lambda: recheck(formula.constraint, formula.payload, proof))  # type: ignore[attr-defined]
        rows.append(
            {
                "obligation_id": obligation.obligation_id,
                "constraint": formula.constraint,  # type: ignore[attr-defined]
                "input_hash": obligation.content_hash,
                "formula_hash": formula.formula_hash,  # type: ignore[attr-defined]
                "plain_validation": validation.as_dict(),
                "lowering": lowered.as_dict(),
                "single_solver": single.as_dict(),
                "dual_solver_differential": dual.as_dict(),
                "proof_recheck": proof_check.as_dict(),
                "single_status": single.value.status,  # type: ignore[attr-defined]
                "dual_status": dual.value.status,  # type: ignore[attr-defined]
                "proof_recheck_ok": bool(proof_check.value[0]) if isinstance(proof_check.value, tuple) else False,
            }
        )
    failing = [
        str(row["obligation_id"])
        for row in rows
        if row["single_status"] != row["dual_status"] or not row["proof_recheck_ok"]
    ]
    return {
        "schema_version": OVERHEAD_SCHEMA_VERSION,
        "all_pass": not failing,
        "failing_aspects": ["status_or_recheck"] if failing else [],
        "failure_summary": "status mismatch or proof re-check failure: " + ", ".join(failing) if failing else "",
        "sample_size": len(rows),
        "solver": solver_diagnostics(),
        "rows": rows,
        "benchmark_hash": stable_json_hash({"rows": rows}),
    }


def export_proof_packet(
    obligation: Obligation,
    verdict: CertifierVerdict | None = None,
    *,
    admitted_scope: Sequence[str] = (),
) -> dict[str, object]:
    """Export a versioned proof-carrying packet for downstream contract boundaries."""

    verdict = verdict or StructuralCertifier().certify(obligation)
    formula = lower_obligation(obligation)
    proof = _solver_diagnostic(verdict)
    ok, detail = recheck(formula.constraint, formula.payload, proof)
    packet = {
        "schema_version": PROOF_PACKET_SCHEMA_VERSION,
        "producer": "datarefine",
        "obligation_id": obligation.obligation_id,
        "obligation_hash": obligation.content_hash,
        "lowered_formula": formula.as_dict(),
        "verdict": verdict.status,
        "proof_object": proof,
        "proof_recheck": {
            "all_pass": ok,
            "failing_aspects": [] if ok else ["proof_recheck"],
            "failure_summary": "" if ok else detail,
            "detail": detail,
        },
        "solver_identity": solver_diagnostics(),
        "env_hash": str(proof.get("env_hash") or solver_diagnostics()["env_hash"]),
        "admitted_scope": list(admitted_scope),
    }
    packet["packet_hash"] = stable_json_hash(packet)
    return packet


def proof_packet_schema() -> dict[str, object]:
    """Return the stable, JSON-compatible proof-packet contract."""

    return {
        "schema_version": PROOF_PACKET_SCHEMA_VERSION,
        "type": "object",
        "required": list(PROOF_PACKET_REQUIRED_KEYS),
        "properties": {
            "schema_version": {"const": PROOF_PACKET_SCHEMA_VERSION},
            "producer": {"const": "datarefine"},
            "obligation_id": {"type": "string"},
            "obligation_hash": {"type": "string"},
            "lowered_formula": {"type": "object"},
            "verdict": {"enum": ["admitted", "rejected", "unknown", "empirical-required", "skipped"]},
            "proof_object": {"type": "object"},
            "proof_recheck": {"type": "object"},
            "solver_identity": {"type": "object"},
            "env_hash": {"type": "string"},
            "admitted_scope": {"type": "array", "items": {"type": "string"}},
            "packet_hash": {"type": "string"},
        },
        "recheck_api": "datarefine.recheck_proof_packet(packet)",
    }


def recheck_proof_packet(packet: Mapping[str, object]) -> dict[str, object]:
    """Re-check packet integrity and its embedded proof object."""

    missing = [key for key in PROOF_PACKET_REQUIRED_KEYS if key not in packet]
    if missing:
        return {
            "all_pass": False,
            "failing_aspects": ["missing_fields"],
            "failure_summary": "missing proof packet fields: " + ", ".join(missing),
        }
    expected_hash = stable_json_hash({key: value for key, value in packet.items() if key != "packet_hash"})
    if packet.get("packet_hash") != expected_hash:
        return {
            "all_pass": False,
            "failing_aspects": ["packet_hash"],
            "failure_summary": "proof packet hash mismatch",
        }
    formula = packet.get("lowered_formula")
    proof = packet.get("proof_object")
    if not isinstance(formula, Mapping) or not isinstance(proof, Mapping):
        return {
            "all_pass": False,
            "failing_aspects": ["packet_shape"],
            "failure_summary": "lowered_formula and proof_object must be mappings",
        }
    constraint = str(formula.get("constraint", ""))
    payload = formula.get("payload")
    if not constraint or not isinstance(payload, Mapping):
        return {
            "all_pass": False,
            "failing_aspects": ["lowered_formula"],
            "failure_summary": "lowered formula is missing constraint or payload",
        }
    ok, detail = recheck(constraint, payload, proof)
    return {
        "all_pass": ok,
        "failing_aspects": [] if ok else ["proof_recheck"],
        "failure_summary": "" if ok else detail,
        "detail": detail,
    }


def audit_formal_claim_scope() -> dict[str, object]:
    """Summarize the formal-verification claims that must stay ledger-bounded."""

    boundary = formal_verification_boundary()
    non_admitted = {
        "unknown": list(boundary["sound_but_incomplete"]),
        "empirical_required": list(boundary["empirical_required"]),
        "out_of_scope": list(boundary["out_of_scope"]),
    }
    return {
        "schema_version": "datarefine.formal_claim_scope_audit.v1",
        "all_pass": all(bool(values) for values in non_admitted.values()),
        "failing_aspects": [],
        "failure_summary": "",
        "required_prose_rule": "Stage F formal claims must cite .ledgers/formal_verification.json and annotate unknown, empirical-required, and out-of-scope obligations as non-admissions.",
        "ledger": ".ledgers/formal_verification.json",
        "non_admitted": non_admitted,
    }


def workflow_proof_manifest(
    manifest: Mapping[str, object] | str,
    *,
    output_dir: str | None = None,
    fixture_mode: bool = True,
    offline: bool = True,
) -> dict[str, object]:
    """Verify a workflow and attach re-checkable proof packets to each structural obligation.

    The public workflow manifest schema is left unchanged. Proof material is emitted
    as a downstream reviewer artifact that references the verifier's obligation ids
    and certifier packet.
    """

    result = verify_workflow(manifest, output_dir=output_dir, fixture_mode=fixture_mode, offline=offline)
    rows: list[dict[str, object]] = []
    for item, verdict in zip(result.obligations.obligations, result.combined):
        constraint = str(item.payload.get("constraint", ""))
        if constraint not in DECIDABLE_COMPLETE_FAMILIES:
            continue
        proof = export_proof_packet(item, admitted_scope=[str(result.manifest.get("workflow_id", "workflow"))])
        rechecked = recheck_proof_packet(proof)
        rows.append(
            {
                "obligation_id": item.obligation_id,
                "id_hint": str(item.payload.get("id_hint", "")),
                "constraint": constraint,
                "combined_status": str(getattr(verdict, "status")),
                "lowered_formula": proof["lowered_formula"],
                "solver_report": proof["proof_object"],
                "proof_object": proof,
                "independent_recheck": rechecked,
            }
        )
    failing = [
        row["obligation_id"]
        for row in rows
        if not bool(row["independent_recheck"].get("all_pass"))  # type: ignore[union-attr]
    ]
    packet = {
        "schema_version": "datarefine.workflow_proof_manifest.v1",
        "workflow_id": str(result.manifest.get("workflow_id", "")),
        "manifest_schema_unchanged": result.manifest.get("schema_version") == "datarefine.workflow.v1",
        "certifier_packet": result.artifacts.get("certifier_packet.json", ""),
        "proof_count": len(rows),
        "proofs": rows,
        "all_pass": not failing,
        "failing_aspects": ["proof_recheck"] if failing else [],
        "failure_summary": "proof re-check failed for: " + ", ".join(str(item) for item in failing) if failing else "",
    }
    packet["manifest_hash"] = stable_json_hash(packet)
    return packet


__all__ = [
    "FORMAL_BOUNDARY_SCHEMA_VERSION",
    "DETERMINISM_SCHEMA_VERSION",
    "OVERHEAD_SCHEMA_VERSION",
    "PROOF_PACKET_SCHEMA_VERSION",
    "PROOF_PACKET_REQUIRED_KEYS",
    "DECIDABLE_COMPLETE_FAMILIES",
    "SOUND_INCOMPLETE_FAMILIES",
    "EMPIRICAL_REQUIRED_FAMILIES",
    "OUT_OF_SCOPE_FAMILIES",
    "formal_verification_boundary",
    "rerun_solver_determinism",
    "benchmark_formal_overhead",
    "export_proof_packet",
    "proof_packet_schema",
    "recheck_proof_packet",
    "audit_formal_claim_scope",
    "workflow_proof_manifest",
]
