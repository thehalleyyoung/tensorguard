"""Step 245 -- top-level SafetyCertificate replay on verify_architecture."""

import textwrap

from src.api import verify_architecture


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


_BUGGY_MODULE = textwrap.dedent(
    """
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 7)
            self.fc2 = nn.Linear(9, 3)

        def forward(self, x):
            return self.fc2(self.fc1(x))
    """
)


_OUT_OF_FRAGMENT_MODULE = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def forward(self, x):
            if x.sum().item() > 0:
                return x
            return x
    """
)


def test_verify_architecture_replays_top_level_safety_certificate():
    result = verify_architecture(
        _SAFE_MODULE,
        input_shapes={"x": ("batch", 10)},
        max_cegar_iterations=0,
        produce_certificates=True,
    )

    assert result.verdict == "SAFE"
    assert result.safety_certificate is not None
    assert result.proof_certificate is not None
    assert result.proof_certificate is result.safety_certificate.proof_certificate
    assert result.proof_certificate.verify_locally()
    assert result.certificate_replay is not None
    assert result.certificate_replay.checked
    assert result.certificate_replay.ok
    assert result.certificate_replay.proof_steps > 0
    assert result.certificate_replay.verification_conditions > 0
    assert result.certificate_replay.certificate_hash


def test_verify_architecture_replays_structural_certificate_without_proof_mode():
    result = verify_architecture(
        _SAFE_MODULE,
        input_shapes={"x": ("batch", 10)},
        max_cegar_iterations=0,
    )

    assert result.verdict == "SAFE"
    assert result.safety_certificate is not None
    assert result.proof_certificate is None
    assert result.certificate_replay is not None
    assert result.certificate_replay.checked
    assert result.certificate_replay.ok
    assert result.certificate_replay.proof_steps == 0
    assert "no embedded proof certificate" in result.certificate_replay.reason


def test_verify_architecture_reports_no_certificate_for_unsafe_model():
    result = verify_architecture(
        _BUGGY_MODULE,
        input_shapes={"x": ("batch", 10)},
        max_cegar_iterations=0,
        produce_certificates=True,
    )

    assert result.verdict == "UNSAFE"
    assert result.safety_certificate is None
    assert result.proof_certificate is None
    assert result.certificate_replay is not None
    assert not result.certificate_replay.ok
    assert not result.certificate_replay.checked
    assert "no SafetyCertificate" in result.certificate_replay.reason


def test_verify_architecture_does_not_certify_unknown_verdict():
    result = verify_architecture(
        _OUT_OF_FRAGMENT_MODULE,
        input_shapes={"x": ("batch", 10)},
        max_cegar_iterations=0,
        soundness_mode="sound",
        produce_certificates=True,
    )

    assert result.verdict == "UNKNOWN"
    assert result.safety_certificate is None
    assert result.proof_certificate is None
    assert result.certificate_replay is not None
    assert not result.certificate_replay.ok
    assert not result.certificate_replay.checked
    assert "top-level verdict UNKNOWN" in result.certificate_replay.reason
