"""Step 246 -- compact signed certificates for CI replay without Z3."""

from __future__ import annotations

import copy
import base64
import hashlib
import hmac
import json
import textwrap

import pytest

from src.api import verify_architecture
from src.signed_certificate import (
    build_certificate_drift_context,
    dumps_signed_certificate,
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


def _real_certificate():
    result = verify_architecture(
        _SAFE_MODULE,
        input_shapes={"x": ("batch", 10)},
        max_cegar_iterations=0,
        produce_certificates=True,
    )
    assert result.verdict == "SAFE"
    assert result.safety_certificate is not None
    assert result.proof_certificate is not None
    return result.safety_certificate


def _drift_context(
    *,
    source=_SAFE_MODULE,
    config=None,
    dependencies=None,
    soundness_contract="sound mode returns UNKNOWN outside the fragment",
):
    return build_certificate_drift_context(
        source=source,
        config=config or {
            "input_shapes": {"x": ["batch", 10]},
            "max_cegar_iterations": 0,
            "soundness_mode": "balanced",
        },
        dependencies=dependencies or {
            "pyproject.toml": "z3-solver>=4.12,<5",
            "src/proof_certificate.py": "local structural replay",
        },
        soundness_contract=soundness_contract,
    )


def _resign_artifact(artifact, secret):
    payload_bytes = json.dumps(
        artifact["payload"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact["payload_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    artifact["signature"] = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")
    return artifact


def test_sign_and_verify_real_safety_certificate_without_solver_replay():
    drift = _drift_context()
    artifact = sign_safety_certificate(
        _real_certificate(),
        "ci-secret",
        issued_at="2026-06-03T00:00:00+00:00",
        key_id="ci-key",
        drift_context=drift,
    )

    verification = verify_signed_certificate(
        artifact,
        "ci-secret",
        current_drift_context=drift,
        require_drift_context=True,
    )

    assert verification.ok
    assert verification.checked
    assert verification.model_name == "M"
    assert verification.proof_steps > 0
    assert verification.payload_sha256 == artifact["payload_sha256"]


def test_signed_certificate_round_trips_through_deterministic_json():
    drift = _drift_context()
    artifact = sign_safety_certificate(
        _real_certificate(),
        b"ci-secret",
        issued_at="2026-06-03T00:00:00+00:00",
        drift_context=drift,
    )
    encoded = dumps_signed_certificate(artifact)

    assert json.loads(encoded) == artifact
    assert verify_signed_certificate(
        encoded,
        b"ci-secret",
        current_drift_context=drift,
        require_drift_context=True,
    ).ok


def test_signed_certificate_is_deterministic_when_issued_at_is_fixed():
    cert = _real_certificate()
    drift = _drift_context()
    first = sign_safety_certificate(
        cert,
        "ci-secret",
        issued_at="2026-06-03T00:00:00+00:00",
        drift_context=drift,
    )
    second = sign_safety_certificate(
        cert,
        "ci-secret",
        issued_at="2026-06-03T00:00:00+00:00",
        drift_context=drift,
    )

    assert first == second
    assert "verification_time_ms" not in dumps_signed_certificate(first)
    assert "z3_total_time_ms" not in dumps_signed_certificate(first)


def test_wrong_secret_and_signed_payload_tampering_are_rejected():
    drift = _drift_context()
    artifact = sign_safety_certificate(
        _real_certificate(),
        "ci-secret",
        issued_at="2026-06-03T00:00:00+00:00",
        drift_context=drift,
    )
    assert not verify_signed_certificate(artifact, "wrong-secret").ok

    tampered = copy.deepcopy(artifact)
    tampered["payload"]["properties"] = ["shape_compatible", "fake_property"]
    assert not verify_signed_certificate(tampered, "ci-secret").ok


def test_payload_hash_and_validly_signed_bad_proof_hash_are_rejected():
    drift = _drift_context()
    artifact = sign_safety_certificate(
        _real_certificate(),
        "ci-secret",
        issued_at="2026-06-03T00:00:00+00:00",
        drift_context=drift,
    )

    bad_payload_hash = copy.deepcopy(artifact)
    bad_payload_hash["payload_sha256"] = "0" * 64
    assert not verify_signed_certificate(bad_payload_hash, "ci-secret").ok

    bad_proof_hash = copy.deepcopy(artifact)
    bad_proof_hash["payload"]["proof"]["proof_sha256"] = "0" * 64
    valid_outer_mac = _resign_artifact(bad_proof_hash, "ci-secret")
    verification = verify_signed_certificate(valid_outer_mac, "ci-secret")
    assert not verification.ok
    assert "proof SHA-256 mismatch" in verification.reason


@pytest.mark.parametrize(
    ("changed_context", "expected_reason"),
    [
        (_drift_context(source=_SAFE_MODULE + "\n# edit"), "source drift detected"),
        (
            _drift_context(config={"input_shapes": {"x": ["B", 10]}}),
            "config drift detected",
        ),
        (
            _drift_context(dependencies={"pyproject.toml": "z3-solver>=4.13,<5"}),
            "dependency drift detected",
        ),
        (
            _drift_context(soundness_contract="sound mode changed"),
            "soundness-contract drift detected",
        ),
    ],
)
def test_ci_verification_invalidates_on_source_config_dependency_or_contract_drift(
    changed_context,
    expected_reason,
):
    artifact = sign_safety_certificate(
        _real_certificate(),
        "ci-secret",
        issued_at="2026-06-03T00:00:00+00:00",
        drift_context=_drift_context(),
    )

    verification = verify_signed_certificate(
        artifact,
        "ci-secret",
        current_drift_context=changed_context,
        require_drift_context=True,
    )

    assert not verification.ok
    assert expected_reason in verification.reason


def test_signing_requires_a_real_embedded_proof_by_default():
    result = verify_architecture(
        _SAFE_MODULE,
        input_shapes={"x": ("batch", 10)},
        max_cegar_iterations=0,
        produce_certificates=False,
    )
    assert result.verdict == "SAFE"
    assert result.safety_certificate is not None
    assert result.proof_certificate is None

    with pytest.raises(ValueError, match="no embedded ProofCertificate"):
        sign_safety_certificate(
            result.safety_certificate,
            "ci-secret",
            issued_at="2026-06-03T00:00:00+00:00",
        )


def test_empty_secret_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        sign_safety_certificate(
            _real_certificate(),
            "",
            issued_at="2026-06-03T00:00:00+00:00",
        )
