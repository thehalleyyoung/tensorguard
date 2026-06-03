"""Step 247 -- independent trusted checker for signed certificates."""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import inspect
import json
import textwrap

from src.api import verify_architecture
from src.certificate_checker import check_signed_certificate_artifact
from src.signed_certificate import (
    build_certificate_drift_context,
    sign_safety_certificate,
    verify_signed_certificate,
)


_SAFE_MODULE = textwrap.dedent(
    """
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)
    """
)


def _artifact():
    result = verify_architecture(
        _SAFE_MODULE,
        input_shapes={"x": ("batch", 10)},
        max_cegar_iterations=0,
        produce_certificates=True,
    )
    assert result.verdict == "SAFE"
    drift = build_certificate_drift_context(
        source=_SAFE_MODULE,
        config={"input_shapes": {"x": ["batch", 10]}, "max_cegar_iterations": 0},
        dependencies={"pyproject.toml": "z3-solver>=4.12,<5"},
        soundness_contract="sound mode abstains outside the fragment",
    )
    artifact = sign_safety_certificate(
        result.safety_certificate,
        "ci-secret",
        issued_at="2026-06-03T00:00:00+00:00",
        drift_context=drift,
    )
    return artifact, drift.to_dict()


def _canonical_payload(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _legacy_proof_hash(steps):
    digest = hashlib.sha256()
    for step in steps:
        digest.update(step["rule"].encode("utf-8"))
        digest.update(step["conclusion"].encode("utf-8"))
        for premise in step["premises"]:
            digest.update(int(premise).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


def _resign(artifact):
    payload_bytes = _canonical_payload(artifact["payload"])
    artifact["payload_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    artifact["signature"] = base64.urlsafe_b64encode(
        hmac.new(b"ci-secret", payload_bytes, hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")
    return artifact


def test_independent_checker_accepts_artifact_accepted_by_public_replay_path():
    artifact, drift = _artifact()

    public = verify_signed_certificate(
        artifact,
        "ci-secret",
        current_drift_context=drift,
        require_drift_context=True,
    )
    trusted = check_signed_certificate_artifact(
        artifact,
        "ci-secret",
        current_drift_context=drift,
        require_drift_context=True,
    )

    assert public.ok
    assert trusted.ok
    assert trusted.model_name == public.model_name == "M"
    assert trusted.proof_steps == public.proof_steps
    assert trusted.payload_sha256 == public.payload_sha256


def test_independent_checker_rejects_same_tampering_as_public_path():
    artifact, drift = _artifact()
    tampered = copy.deepcopy(artifact)
    tampered["payload"]["checked_steps"] = -1
    tampered = _resign(tampered)

    public = verify_signed_certificate(
        tampered,
        "ci-secret",
        current_drift_context=drift,
        require_drift_context=True,
    )
    trusted = check_signed_certificate_artifact(
        tampered,
        "ci-secret",
        current_drift_context=drift,
        require_drift_context=True,
    )

    assert not public.ok
    assert not trusted.ok
    assert "checked_steps" in public.reason
    assert "checked_steps" in trusted.reason


def test_independent_checker_catches_forward_premise_even_with_valid_mac():
    artifact, drift = _artifact()
    tampered = copy.deepcopy(artifact)
    steps = tampered["payload"]["proof"]["steps"]
    steps[0]["rule"] = "mp"
    steps[0]["premises"] = [1]
    tampered["payload"]["proof"]["certificate_hash"] = _legacy_proof_hash(steps)
    proof_without_hash = dict(tampered["payload"]["proof"])
    proof_without_hash.pop("proof_sha256")
    tampered["payload"]["proof"]["proof_sha256"] = hashlib.sha256(
        _canonical_payload(proof_without_hash)
    ).hexdigest()
    tampered = _resign(tampered)

    public = verify_signed_certificate(
        tampered,
        "ci-secret",
        current_drift_context=drift,
        require_drift_context=True,
    )
    trusted = check_signed_certificate_artifact(
        tampered,
        "ci-secret",
        current_drift_context=drift,
        require_drift_context=True,
    )

    assert not public.ok
    assert not trusted.ok
    assert "replay" in public.reason or "premise" in public.reason
    assert "premise is not earlier" in trusted.reason


def test_independent_checker_has_no_tensorguard_or_solver_imports():
    import src.certificate_checker as checker

    source = inspect.getsource(checker)
    assert "import z3" not in source
    assert "from z3" not in source
    assert "from src." not in source
    assert "import src." not in source
