"""Compact signed certificate artifacts for CI replay.

The helpers in this module intentionally avoid solver calls.  They sign a
deterministic payload extracted from a top-level ``SafetyCertificate`` and
verify the embedded proof DAG structurally, so a CI job can authenticate and
replay a SAFE claim without re-running Z3.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from src.proof_certificate import CertificateStrategy, ProofCertificate, ProofStep


SCHEMA_VERSION = "tensorguard.signed-safety-certificate.v1"
SIGNATURE_ALGORITHM = "HMAC-SHA256"

JsonArtifact = Union[str, bytes, Mapping[str, Any]]
Secret = Union[str, bytes]
Blob = Union[str, bytes, Mapping[str, Any], Sequence[Any]]

_DRIFT_FIELDS = (
    "source_sha256",
    "config_sha256",
    "dependency_sha256",
    "soundness_contract_sha256",
)


@dataclass(frozen=True)
class SignedCertificateVerification:
    """Result of verifying a signed TensorGuard certificate artifact."""

    ok: bool
    checked: bool
    reason: str = ""
    payload_sha256: Optional[str] = None
    proof_steps: int = 0
    model_name: Optional[str] = None


@dataclass(frozen=True)
class CertificateDriftContext:
    """Fingerprints that make a signed certificate stale when CI inputs drift."""

    source_sha256: str
    config_sha256: str
    dependency_sha256: str
    soundness_contract_sha256: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "dependency_sha256": self.dependency_sha256,
            "soundness_contract_sha256": self.soundness_contract_sha256,
        }


def build_certificate_drift_context(
    *,
    source: Blob,
    config: Blob,
    dependencies: Blob,
    soundness_contract: Blob,
) -> CertificateDriftContext:
    """Build the four drift fingerprints bound into a signed certificate."""

    return CertificateDriftContext(
        source_sha256=_hash_blob(source),
        config_sha256=_hash_blob(config),
        dependency_sha256=_hash_blob(dependencies),
        soundness_contract_sha256=_hash_blob(soundness_contract),
    )


def sign_safety_certificate(
    certificate: Any,
    secret: Secret,
    *,
    issued_at: Optional[str] = None,
    issuer: str = "tensorguard",
    key_id: Optional[str] = None,
    require_proof: bool = True,
    drift_context: Optional[Union[CertificateDriftContext, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return a compact MAC-signed artifact for a SAFE ``SafetyCertificate``.

    The signed payload excludes timing fields and the legacy pretty-print hash
    because both are unsuitable as deterministic integrity anchors.  When
    ``require_proof`` is true, the embedded ``ProofCertificate`` is replayed
    structurally before signing and included in the payload for CI replay.
    """

    payload = _certificate_payload(
        certificate,
        issued_at=issued_at or _utc_now_iso(),
        issuer=issuer,
        key_id=key_id,
        require_proof=require_proof,
        drift_context=drift_context,
    )
    canonical = _canonical_json(payload)
    payload_sha256 = hashlib.sha256(canonical).hexdigest()
    signature = _sign_bytes(canonical, _normalise_secret(secret))
    return {
        "payload": payload,
        "payload_sha256": payload_sha256,
        "signature": signature,
    }


def dumps_signed_certificate(artifact: Mapping[str, Any]) -> str:
    """Serialize a signed certificate artifact in deterministic JSON form."""

    return json.dumps(artifact, sort_keys=True, separators=(",", ":"))


def verify_signed_certificate(
    artifact: JsonArtifact,
    secret: Secret,
    *,
    require_proof: bool = True,
    current_drift_context: Optional[
        Union[CertificateDriftContext, Mapping[str, Any]]
    ] = None,
    require_drift_context: bool = False,
) -> SignedCertificateVerification:
    """Verify a signed certificate artifact without invoking an SMT solver."""

    parsed = _parse_artifact(artifact)
    payload = parsed.get("payload")
    if not isinstance(payload, Mapping):
        return SignedCertificateVerification(
            ok=False,
            checked=True,
            reason="signed certificate missing payload",
        )

    canonical = _canonical_json(payload)
    payload_sha256 = hashlib.sha256(canonical).hexdigest()
    if parsed.get("payload_sha256") != payload_sha256:
        return SignedCertificateVerification(
            ok=False,
            checked=True,
            reason="payload SHA-256 mismatch",
            payload_sha256=payload_sha256,
        )

    expected_signature = _sign_bytes(canonical, _normalise_secret(secret))
    actual_signature = parsed.get("signature")
    if not isinstance(actual_signature, str) or not hmac.compare_digest(
        actual_signature, expected_signature
    ):
        return SignedCertificateVerification(
            ok=False,
            checked=True,
            reason="signature mismatch",
            payload_sha256=payload_sha256,
        )

    payload_check = _verify_payload(
        payload,
        require_proof=require_proof,
        current_drift_context=current_drift_context,
        require_drift_context=require_drift_context,
    )
    return SignedCertificateVerification(
        ok=payload_check.ok,
        checked=True,
        reason=payload_check.reason,
        payload_sha256=payload_sha256,
        proof_steps=payload_check.proof_steps,
        model_name=payload_check.model_name,
    )


def _certificate_payload(
    certificate: Any,
    *,
    issued_at: str,
    issuer: str,
    key_id: Optional[str],
    require_proof: bool,
    drift_context: Optional[Union[CertificateDriftContext, Mapping[str, Any]]],
) -> Dict[str, Any]:
    if certificate is None:
        raise ValueError("cannot sign a missing SafetyCertificate")

    model_name = _require_str(getattr(certificate, "model_name", None), "model_name")
    properties = _require_str_list(getattr(certificate, "properties", None), "properties")
    if not properties:
        raise ValueError("cannot sign a SafetyCertificate with no proved properties")

    k = _require_nonnegative_int(getattr(certificate, "k", None), "k")
    checked_steps = _require_nonnegative_int(
        getattr(certificate, "checked_steps", None),
        "checked_steps",
    )

    proof = getattr(certificate, "proof_certificate", None)
    if proof is None and require_proof:
        raise ValueError("SafetyCertificate has no embedded ProofCertificate")

    payload: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "algorithm": SIGNATURE_ALGORITHM,
        "verdict": "SAFE",
        "issuer": _require_str(issuer, "issuer"),
        "issued_at": _require_str(issued_at, "issued_at"),
        "model_name": model_name,
        "properties": properties,
        "k": k,
        "checked_steps": checked_steps,
        "symbolic_bindings": _string_dict(
            getattr(certificate, "symbolic_bindings", {}) or {},
            "symbolic_bindings",
        ),
        "theories_used": _require_str_list(
            getattr(certificate, "theories_used", []) or [],
            "theories_used",
            allow_empty=True,
        ),
        "product_domains": _require_str_list(
            getattr(certificate, "product_domains", []) or [],
            "product_domains",
            allow_empty=True,
        ),
    }
    if key_id is not None:
        payload["key_id"] = _require_str(key_id, "key_id")
    if drift_context is not None:
        payload["drift"] = _normalise_drift_context(drift_context)
    if proof is not None:
        payload["proof"] = _proof_payload(proof)
    return payload


def _proof_payload(proof: Any) -> Dict[str, Any]:
    if not bool(proof.verify_locally()):
        raise ValueError("embedded ProofCertificate failed local replay")

    proof_steps = [
        {
            "rule": _require_str(step.rule, "proof.steps[].rule"),
            "conclusion": _require_str(step.conclusion, "proof.steps[].conclusion"),
            "premises": [
                _require_nonnegative_int(p, "proof.steps[].premises[]")
                for p in step.premises
            ],
            **({"theory": _require_str(step.theory, "proof.steps[].theory")}
               if step.theory is not None else {}),
        }
        for step in proof.steps
    ]
    payload: Dict[str, Any] = {
        "model_name": _require_str(proof.model_name, "proof.model_name"),
        "properties": _require_str_list(proof.properties, "proof.properties"),
        "steps": proof_steps,
        "root_step": _require_nonnegative_int(proof.root_step, "proof.root_step"),
        "theories_used": _require_str_list(
            proof.theories_used,
            "proof.theories_used",
            allow_empty=True,
        ),
        "proof_source": _require_str(proof.proof_source, "proof.proof_source"),
        "certificate_hash": _legacy_proof_hash(proof_steps),
    }
    if proof.strategy is not None:
        if isinstance(proof.strategy, CertificateStrategy):
            payload["strategy"] = proof.strategy.value
        else:
            payload["strategy"] = _require_str(proof.strategy, "proof.strategy")
    payload["proof_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _verify_payload(
    payload: Mapping[str, Any],
    *,
    require_proof: bool,
    current_drift_context: Optional[Union[CertificateDriftContext, Mapping[str, Any]]],
    require_drift_context: bool,
) -> SignedCertificateVerification:
    if payload.get("schema") != SCHEMA_VERSION:
        return SignedCertificateVerification(False, True, "unsupported schema")
    if payload.get("algorithm") != SIGNATURE_ALGORITHM:
        return SignedCertificateVerification(False, True, "unsupported signature algorithm")
    if payload.get("verdict") != "SAFE":
        return SignedCertificateVerification(False, True, "certificate verdict is not SAFE")

    model_name = payload.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        return SignedCertificateVerification(False, True, "missing model_name")
    properties = payload.get("properties")
    if not _is_nonempty_str_list(properties):
        return SignedCertificateVerification(
            False,
            True,
            "certificate proves no properties",
            model_name=model_name,
        )
    for field_name in ("k", "checked_steps"):
        if not _is_nonnegative_int(payload.get(field_name)):
            return SignedCertificateVerification(
                False,
                True,
                f"{field_name} must be a non-negative integer",
                model_name=model_name,
            )

    drift = payload.get("drift")
    if drift is None:
        if require_drift_context:
            return SignedCertificateVerification(
                False,
                True,
                "signed certificate missing drift context",
                model_name=model_name,
            )
    else:
        drift_check = _verify_drift_context(
            drift,
            current_drift_context=current_drift_context,
            require_drift_context=require_drift_context,
            model_name=model_name,
        )
        if not drift_check.ok:
            return drift_check

    proof = payload.get("proof")
    if proof is None:
        if require_proof:
            return SignedCertificateVerification(
                False,
                True,
                "signed certificate missing embedded proof",
                model_name=model_name,
            )
        return SignedCertificateVerification(
            True,
            True,
            "signed SafetyCertificate verified without embedded proof",
            model_name=model_name,
        )
    if not isinstance(proof, Mapping):
        return SignedCertificateVerification(False, True, "proof payload is not an object")

    proof_result = _verify_proof_payload(proof)
    if not proof_result.ok:
        return proof_result
    return SignedCertificateVerification(
        True,
        True,
        "signed SafetyCertificate and embedded proof replayed",
        proof_steps=proof_result.proof_steps,
        model_name=model_name,
    )


def _verify_drift_context(
    drift: Any,
    *,
    current_drift_context: Optional[Union[CertificateDriftContext, Mapping[str, Any]]],
    require_drift_context: bool,
    model_name: str,
) -> SignedCertificateVerification:
    if not isinstance(drift, Mapping):
        return SignedCertificateVerification(False, True, "drift context is not an object")
    try:
        signed = _normalise_drift_context(drift)
    except ValueError as exc:
        return SignedCertificateVerification(False, True, str(exc), model_name=model_name)
    if current_drift_context is None:
        if require_drift_context:
            return SignedCertificateVerification(
                False,
                True,
                "current drift context required for CI verification",
                model_name=model_name,
            )
        return SignedCertificateVerification(True, True, model_name=model_name)
    try:
        current = _normalise_drift_context(current_drift_context)
    except ValueError as exc:
        return SignedCertificateVerification(False, True, str(exc), model_name=model_name)
    for field_name in _DRIFT_FIELDS:
        if signed[field_name] != current[field_name]:
            drift_name = field_name.removesuffix("_sha256").replace("_", "-")
            return SignedCertificateVerification(
                False,
                True,
                f"{drift_name} drift detected",
                model_name=model_name,
            )
    return SignedCertificateVerification(True, True, model_name=model_name)


def _verify_proof_payload(proof: Mapping[str, Any]) -> SignedCertificateVerification:
    claimed_hash = proof.get("proof_sha256")
    proof_without_hash = dict(proof)
    proof_without_hash.pop("proof_sha256", None)
    actual_hash = hashlib.sha256(_canonical_json(proof_without_hash)).hexdigest()
    if claimed_hash != actual_hash:
        return SignedCertificateVerification(False, True, "proof SHA-256 mismatch")

    steps_obj = proof.get("steps")
    if not isinstance(steps_obj, Sequence) or isinstance(steps_obj, (str, bytes)):
        return SignedCertificateVerification(False, True, "proof steps must be a list")

    steps: List[ProofStep] = []
    for raw_step in steps_obj:
        if not isinstance(raw_step, Mapping):
            return SignedCertificateVerification(False, True, "proof step is not an object")
        rule = raw_step.get("rule")
        conclusion = raw_step.get("conclusion")
        premises = raw_step.get("premises")
        theory = raw_step.get("theory")
        if not isinstance(rule, str) or not isinstance(conclusion, str):
            return SignedCertificateVerification(False, True, "proof step missing rule/conclusion")
        if not isinstance(premises, Sequence) or isinstance(premises, (str, bytes)):
            return SignedCertificateVerification(False, True, "proof step premises must be a list")
        if any(not _is_nonnegative_int(p) for p in premises):
            return SignedCertificateVerification(False, True, "proof premise must be non-negative")
        if theory is not None and not isinstance(theory, str):
            return SignedCertificateVerification(False, True, "proof theory must be a string")
        steps.append(
            ProofStep(
                rule=rule,
                conclusion=conclusion,
                premises=[int(p) for p in premises],
                theory=theory,
            )
        )

    root_step = proof.get("root_step")
    if not _is_nonnegative_int(root_step):
        return SignedCertificateVerification(False, True, "proof root_step is invalid")
    properties = proof.get("properties")
    if not _is_nonempty_str_list(properties):
        return SignedCertificateVerification(False, True, "proof has no properties")
    model_name = proof.get("model_name")
    if not isinstance(model_name, str):
        return SignedCertificateVerification(False, True, "proof model_name is invalid")

    certificate_hash = _legacy_proof_hash([step.to_dict() for step in steps])
    if proof.get("certificate_hash") != certificate_hash:
        return SignedCertificateVerification(False, True, "proof certificate_hash mismatch")

    cert = ProofCertificate(
        model_name=model_name,
        properties=list(properties),
        steps=steps,
        root_step=int(root_step),
        theories_used=list(proof.get("theories_used", []) or []),
        verification_conditions=[],
        proof_source=str(proof.get("proof_source", "")),
        strategy=_strategy_from_payload(proof.get("strategy")),
    )
    if not cert.verify_locally():
        return SignedCertificateVerification(False, True, "embedded proof failed local replay")
    return SignedCertificateVerification(
        True,
        True,
        "embedded proof replayed",
        proof_steps=len(steps),
        model_name=model_name,
    )


def _strategy_from_payload(value: Any) -> Optional[CertificateStrategy]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    for strategy in CertificateStrategy:
        if strategy.value == value:
            return strategy
    return None


def _parse_artifact(artifact: JsonArtifact) -> Mapping[str, Any]:
    if isinstance(artifact, bytes):
        parsed = json.loads(artifact.decode("utf-8"))
    elif isinstance(artifact, str):
        parsed = json.loads(artifact)
    elif isinstance(artifact, Mapping):
        parsed = artifact
    else:
        raise TypeError("signed certificate artifact must be JSON text or a mapping")
    if not isinstance(parsed, Mapping):
        raise ValueError("signed certificate artifact must be a JSON object")
    return parsed


def _normalise_drift_context(
    drift_context: Union[CertificateDriftContext, Mapping[str, Any]],
) -> Dict[str, str]:
    if isinstance(drift_context, CertificateDriftContext):
        raw = drift_context.to_dict()
    elif isinstance(drift_context, Mapping):
        raw = drift_context
    else:
        raise ValueError("drift context must be a CertificateDriftContext or mapping")

    normalised: Dict[str, str] = {}
    for field_name in _DRIFT_FIELDS:
        value = raw.get(field_name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{field_name} must be a SHA-256 hex digest")
        int(value, 16)
        normalised[field_name] = value.lower()
    return normalised


def _hash_blob(value: Blob) -> str:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8")
    elif isinstance(value, Mapping):
        data = _canonical_json(value)
    elif isinstance(value, Sequence):
        data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    else:
        raise TypeError("drift inputs must be text, bytes, mappings, or sequences")
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign_bytes(data: bytes, secret: bytes) -> str:
    digest = hmac.new(secret, data, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _normalise_secret(secret: Secret) -> bytes:
    if isinstance(secret, str):
        secret_bytes = secret.encode("utf-8")
    elif isinstance(secret, bytes):
        secret_bytes = secret
    else:
        raise TypeError("certificate signing secret must be str or bytes")
    if not secret_bytes:
        raise ValueError("certificate signing secret must not be empty")
    return secret_bytes


def _legacy_proof_hash(steps: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for step in steps:
        digest.update(str(step["rule"]).encode("utf-8"))
        digest.update(str(step["conclusion"]).encode("utf-8"))
        for premise in step.get("premises", []):
            digest.update(int(premise).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_str_list(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of strings")
    result = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only strings")
        result.append(item)
    if not result and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _string_dict(value: Any, field_name: str) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    result: Dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{field_name} must map strings to strings")
        result[key] = item
    return result


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if not _is_nonnegative_int(value):
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_nonempty_str_list(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) > 0
        and all(isinstance(item, str) and item for item in value)
    )
