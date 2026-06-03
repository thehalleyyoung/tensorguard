"""Step 275 -- TensorGuard-verified model-hub badge bundles."""

from __future__ import annotations

import json
import textwrap

import pytest

from src.certificate_checker import check_signed_certificate_artifact
from src.cli.main import ReftypeCliApp
from src.model_hub_badge import (
    BUNDLE_SCHEMA,
    CERTIFICATE_FILENAME,
    MANIFEST_FILENAME,
    MODEL_CARD_SNIPPET_FILENAME,
    write_model_hub_badge_bundle,
)


SAFE_MODEL = textwrap.dedent(
    """
    import torch.nn as nn

    class HubNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            return self.fc2(self.fc1(x))
    """
)


BUGGY_MODEL = SAFE_MODEL.replace("nn.Linear(20, 5)", "nn.Linear(30, 5)")
SHAPES = {"x": ("batch", 10)}
ISSUED_AT = "2026-06-03T00:00:00+00:00"
SECRET = "model-hub-test-secret"


def _write_bundle(tmp_path, source=SAFE_MODEL, output_name="bundle"):
    return write_model_hub_badge_bundle(
        source,
        input_shapes=SHAPES,
        output_dir=tmp_path / output_name,
        model_id="tensorguard/hubnet",
        secret=SECRET,
        filename="hubnet.py",
        issued_at=ISSUED_AT,
    )


def test_model_hub_bundle_contains_badge_manifest_certificate_and_snippet(tmp_path):
    bundle = _write_bundle(tmp_path)

    assert bundle.badge_svg_path.exists()
    assert bundle.certificate_path.name == CERTIFICATE_FILENAME
    assert bundle.manifest_path.name == MANIFEST_FILENAME
    assert bundle.model_card_snippet_path.name == MODEL_CARD_SNIPPET_FILENAME

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == BUNDLE_SCHEMA
    assert manifest["model_id"] == "tensorguard/hubnet"
    assert manifest["verdict"] == "SAFE"
    assert manifest["soundness_mode"] == "sound"
    assert manifest["badge"]["status"] == "verified"
    assert manifest["certificate"]["payload_sha256"] == bundle.payload_sha256
    assert manifest["certificate"]["embedded_proof"] is False
    assert manifest["certificate"]["proof_steps"] == 0
    assert manifest["replay"]["ok"] is True

    snippet = bundle.model_card_snippet_path.read_text(encoding="utf-8")
    assert "TensorGuard verification" in snippet
    assert "tensorguard model-hub-badge hubnet.py" in snippet
    assert bundle.payload_sha256 in snippet
    assert bundle.source_sha256 in snippet


def test_signed_certificate_in_bundle_replays_with_trusted_checker(tmp_path):
    bundle = _write_bundle(tmp_path)
    artifact = bundle.certificate_path.read_text(encoding="utf-8")

    public = check_signed_certificate_artifact(
        artifact,
        SECRET,
        require_proof=False,
        require_drift_context=False,
    )

    assert public.ok
    assert public.model_name == "HubNet"
    assert public.proof_steps == 0
    assert public.payload_sha256 == bundle.payload_sha256


def test_bundle_is_deterministic_for_fixed_timestamp_and_inputs(tmp_path):
    first = _write_bundle(tmp_path, output_name="a")
    second = _write_bundle(tmp_path, output_name="b")

    assert first.certificate_sha256 == second.certificate_sha256
    assert first.payload_sha256 == second.payload_sha256
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    first_manifest["model_card_snippet"]["sha256"] = "<output-path-dependent>"
    second_manifest["model_card_snippet"]["sha256"] = "<output-path-dependent>"
    assert first_manifest == second_manifest


def test_unsafe_model_is_not_badge_eligible_and_writes_no_bundle(tmp_path):
    out = tmp_path / "unsafe"

    with pytest.raises(ValueError, match="verdict=UNSAFE"):
        write_model_hub_badge_bundle(
            BUGGY_MODEL,
            input_shapes=SHAPES,
            output_dir=out,
            model_id="tensorguard/buggy",
            secret=SECRET,
            filename="buggy.py",
            issued_at=ISSUED_AT,
        )

    assert not out.exists()


def test_model_hub_badge_cli_writes_complete_bundle(tmp_path):
    model = tmp_path / "hubnet.py"
    model.write_text(SAFE_MODEL, encoding="utf-8")
    out = tmp_path / "bundle"

    rc = ReftypeCliApp().run(
        [
            "model-hub-badge",
            str(model),
            "-s",
            "x=batch,10",
            "--model-id",
            "tensorguard/hubnet",
            "--output",
            str(out),
            "--secret",
            SECRET,
            "--issued-at",
            ISSUED_AT,
            "--json",
        ]
    )

    assert rc == 0
    assert (out / "tensorguard-verified.svg").exists()
    assert (out / CERTIFICATE_FILENAME).exists()
    assert (out / MANIFEST_FILENAME).exists()
    assert (out / MODEL_CARD_SNIPPET_FILENAME).exists()


def test_model_hub_badge_cli_rejects_missing_secret(tmp_path, monkeypatch):
    model = tmp_path / "hubnet.py"
    model.write_text(SAFE_MODEL, encoding="utf-8")
    monkeypatch.delenv("TENSORGUARD_CERT_SECRET", raising=False)

    rc = ReftypeCliApp().run(
        [
            "model-hub-badge",
            str(model),
            "-s",
            "x=batch,10",
            "--model-id",
            "tensorguard/hubnet",
            "--output",
            str(tmp_path / "bundle"),
        ]
    )

    assert rc == 2
