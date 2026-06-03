"""Standalone signed-certificate checker for CI.

This module is deliberately independent of the TensorGuard verifier and proof
classes: it imports only the Python standard library and checks the signed JSON
artifact directly.  That keeps the trusted replay core small enough to audit and
cross-check against ``src.signed_certificate.verify_signed_certificate``.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union


SCHEMA_VERSION = "tensorguard.signed-safety-certificate.v1"
SIGNATURE_ALGORITHM = "HMAC-SHA256"
_DRIFT_FIELDS = (
    "source_sha256",
    "config_sha256",
    "dependency_sha256",
    "soundness_contract_sha256",
)
_LEAF_RULES = {"asserted", "refl", "hypothesis"}
_PREMISE_OPTIONAL_RULES = {
    "asserted",
    "refl",
    "hypothesis",
    "rewrite",
    "def-intro",
    "commutativity",
    "unit-resolution",
    "th-lemma",
    "iff-true",
    "iff-false",
    "elim-unused",
    "der",
    "sk",
}


@dataclass(frozen=True)
class CertificateCheckResult:
    ok: bool
    checked: bool
    reason: str = ""
    model_name: Optional[str] = None
    proof_steps: int = 0
    payload_sha256: Optional[str] = None


def check_signed_certificate_artifact(
    artifact: Union[str, bytes, Mapping[str, Any]],
    secret: Union[str, bytes],
    *,
    current_drift_context: Optional[Mapping[str, Any]] = None,
    require_proof: bool = True,
    require_drift_context: bool = False,
) -> CertificateCheckResult:
    """Authenticate and structurally replay a signed certificate artifact."""

    parsed = _parse_artifact(artifact)
    payload = parsed.get("payload")
    if not isinstance(payload, Mapping):
        return CertificateCheckResult(False, True, "missing payload")

    payload_bytes = _canonical_json(payload)
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if parsed.get("payload_sha256") != payload_sha256:
        return CertificateCheckResult(
            False,
            True,
            "payload SHA-256 mismatch",
            payload_sha256=payload_sha256,
        )

    expected = _mac(payload_bytes, _secret_bytes(secret))
    actual = parsed.get("signature")
    if not isinstance(actual, str) or not hmac.compare_digest(actual, expected):
        return CertificateCheckResult(
            False,
            True,
            "signature mismatch",
            payload_sha256=payload_sha256,
        )

    payload_result = _check_payload(
        payload,
        current_drift_context=current_drift_context,
        require_proof=require_proof,
        require_drift_context=require_drift_context,
    )
    return CertificateCheckResult(
        payload_result.ok,
        True,
        payload_result.reason,
        model_name=payload_result.model_name,
        proof_steps=payload_result.proof_steps,
        payload_sha256=payload_sha256,
    )


def _check_payload(
    payload: Mapping[str, Any],
    *,
    current_drift_context: Optional[Mapping[str, Any]],
    require_proof: bool,
    require_drift_context: bool,
) -> CertificateCheckResult:
    if payload.get("schema") != SCHEMA_VERSION:
        return CertificateCheckResult(False, True, "unsupported schema")
    if payload.get("algorithm") != SIGNATURE_ALGORITHM:
        return CertificateCheckResult(False, True, "unsupported signature algorithm")
    if payload.get("verdict") != "SAFE":
        return CertificateCheckResult(False, True, "certificate verdict is not SAFE")

    model_name = payload.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        return CertificateCheckResult(False, True, "missing model_name")
    if not _nonempty_string_list(payload.get("properties")):
        return CertificateCheckResult(
            False,
            True,
            "certificate proves no properties",
            model_name=model_name,
        )
    for field_name in ("k", "checked_steps"):
        if not _nonnegative_int(payload.get(field_name)):
            return CertificateCheckResult(
                False,
                True,
                f"{field_name} must be a non-negative integer",
                model_name=model_name,
            )

    drift = payload.get("drift")
    if drift is None:
        if require_drift_context:
            return CertificateCheckResult(
                False,
                True,
                "signed certificate missing drift context",
                model_name=model_name,
            )
    else:
        drift_result = _check_drift(
            drift,
            current_drift_context=current_drift_context,
            require_drift_context=require_drift_context,
            model_name=model_name,
        )
        if not drift_result.ok:
            return drift_result

    proof = payload.get("proof")
    if proof is None:
        if require_proof:
            return CertificateCheckResult(
                False,
                True,
                "signed certificate missing embedded proof",
                model_name=model_name,
            )
        return CertificateCheckResult(
            True,
            True,
            "signed certificate accepted",
            model_name=model_name,
        )
    if not isinstance(proof, Mapping):
        return CertificateCheckResult(
            False,
            True,
            "proof payload is not an object",
            model_name=model_name,
        )
    proof_result = _check_proof(proof)
    if not proof_result.ok:
        return proof_result
    return CertificateCheckResult(
        True,
        True,
        "signed certificate and embedded proof replayed",
        model_name,
        proof_steps=proof_result.proof_steps,
    )


def _check_drift(
    drift: Any,
    *,
    current_drift_context: Optional[Mapping[str, Any]],
    require_drift_context: bool,
    model_name: str,
) -> CertificateCheckResult:
    if not isinstance(drift, Mapping):
        return CertificateCheckResult(
            False,
            True,
            "drift context is not an object",
            model_name=model_name,
        )
    signed = _normalise_drift(drift)
    if signed is None:
        return CertificateCheckResult(
            False,
            True,
            "invalid drift context",
            model_name=model_name,
        )
    if current_drift_context is None:
        if require_drift_context:
            return CertificateCheckResult(
                False,
                True,
                "current drift context required for CI verification",
                model_name,
            )
        return CertificateCheckResult(True, True, model_name=model_name)
    current = _normalise_drift(current_drift_context)
    if current is None:
        return CertificateCheckResult(
            False,
            True,
            "invalid current drift context",
            model_name=model_name,
        )
    for field_name in _DRIFT_FIELDS:
        if signed[field_name] != current[field_name]:
            drift_name = field_name.removesuffix("_sha256").replace("_", "-")
            return CertificateCheckResult(
                False,
                True,
                f"{drift_name} drift detected",
                model_name,
            )
    return CertificateCheckResult(True, True, model_name=model_name)


def _check_proof(proof: Mapping[str, Any]) -> CertificateCheckResult:
    claimed_proof_hash = proof.get("proof_sha256")
    proof_without_hash = dict(proof)
    proof_without_hash.pop("proof_sha256", None)
    actual_proof_hash = hashlib.sha256(_canonical_json(proof_without_hash)).hexdigest()
    if claimed_proof_hash != actual_proof_hash:
        return CertificateCheckResult(False, True, "proof SHA-256 mismatch")

    steps = proof.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return CertificateCheckResult(False, True, "proof steps must be a list")
    if not _nonnegative_int(proof.get("root_step")):
        return CertificateCheckResult(False, True, "proof root_step is invalid")
    root_step = int(proof["root_step"])
    if root_step >= len(steps):
        return CertificateCheckResult(False, True, "proof root_step is out of range")
    if not _nonempty_string_list(proof.get("properties")):
        return CertificateCheckResult(False, True, "proof has no properties")

    normalised_steps = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            return CertificateCheckResult(False, True, "proof step is not an object")
        rule = raw_step.get("rule")
        conclusion = raw_step.get("conclusion")
        premises = raw_step.get("premises")
        if not isinstance(rule, str) or not isinstance(conclusion, str):
            return CertificateCheckResult(False, True, "proof step missing rule/conclusion")
        if not isinstance(premises, Sequence) or isinstance(premises, (str, bytes)):
            return CertificateCheckResult(False, True, "proof step premises must be a list")
        premise_list = []
        for premise in premises:
            if not _nonnegative_int(premise):
                return CertificateCheckResult(False, True, "proof premise must be non-negative")
            if premise >= len(steps) or premise >= index:
                return CertificateCheckResult(False, True, "proof premise is not earlier")
            premise_list.append(int(premise))
        if rule in _LEAF_RULES and premise_list:
            return CertificateCheckResult(False, True, "proof leaf rule has premises")
        if rule not in _PREMISE_OPTIONAL_RULES and not premise_list:
            return CertificateCheckResult(False, True, "proof non-leaf rule has no premises")
        normalised_steps.append({
            "rule": rule,
            "conclusion": conclusion,
            "premises": premise_list,
        })

    if proof.get("certificate_hash") != _legacy_proof_hash(normalised_steps):
        return CertificateCheckResult(False, True, "proof certificate_hash mismatch")
    return CertificateCheckResult(
        True,
        True,
        "embedded proof replayed",
        model_name=proof.get("model_name") if isinstance(proof.get("model_name"), str) else None,
        proof_steps=len(normalised_steps),
    )


def _parse_artifact(artifact: Union[str, bytes, Mapping[str, Any]]) -> Mapping[str, Any]:
    if isinstance(artifact, bytes):
        parsed = json.loads(artifact.decode("utf-8"))
    elif isinstance(artifact, str):
        parsed = json.loads(artifact)
    elif isinstance(artifact, Mapping):
        parsed = artifact
    else:
        raise TypeError("artifact must be JSON text or a mapping")
    if not isinstance(parsed, Mapping):
        raise ValueError("artifact must be a JSON object")
    return parsed


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _mac(data: bytes, secret: bytes) -> str:
    return base64.urlsafe_b64encode(
        hmac.new(secret, data, hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")


def _secret_bytes(secret: Union[str, bytes]) -> bytes:
    if isinstance(secret, str):
        data = secret.encode("utf-8")
    elif isinstance(secret, bytes):
        data = secret
    else:
        raise TypeError("secret must be str or bytes")
    if not data:
        raise ValueError("secret must not be empty")
    return data


def _normalise_drift(drift: Mapping[str, Any]) -> Optional[dict[str, str]]:
    result: dict[str, str] = {}
    for field_name in _DRIFT_FIELDS:
        value = drift.get(field_name)
        if not isinstance(value, str) or len(value) != 64:
            return None
        try:
            int(value, 16)
        except ValueError:
            return None
        result[field_name] = value.lower()
    return result


def _legacy_proof_hash(steps: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for step in steps:
        digest.update(str(step["rule"]).encode("utf-8"))
        digest.update(str(step["conclusion"]).encode("utf-8"))
        for premise in step.get("premises", []):
            digest.update(int(premise).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonempty_string_list(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) > 0
        and all(isinstance(item, str) and item for item in value)
    )
