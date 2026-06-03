"""Step 227 -- tests for the real-model deployment gallery."""

from __future__ import annotations

import pytest

from evaluation import deployment_gallery as dg


def _walk(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk(value)


def test_manifest_is_deterministic_and_measurement_free():
    m1 = dg.manifest()
    m2 = dg.manifest()
    assert m1 == m2
    assert len(m1["models"]) == len(dg.MODEL_SPECS)
    assert len(m1["gate_rows"]) == len(dg.MODEL_SPECS) * len(dg.GATE_SPECS)
    forbidden = {"latency", "memory", "wall_clock", "timestamp"}
    for key, value in _walk(m1):
        assert not any(token in str(key).lower() for token in forbidden)
        assert value is not None


def test_gallery_covers_required_deployment_families():
    families = {row["family"] for row in dg.manifest()["models"]}
    assert families == {
        "ResNet",
        "ViT",
        "Llama-style block",
        "Diffusion U-Net",
        "Recommender",
        "Speech",
    }
    assert len(families) == len(dg.MODEL_SPECS)


def test_every_model_has_fx_and_export_gates():
    by_model = {}
    for row in dg.gate_rows():
        by_model.setdefault(row["model"], set()).add((row["phase"], row["backend"]))
    assert set(by_model) == {spec.name for spec in dg.MODEL_SPECS}
    for gates in by_model.values():
        assert ("before_export", "fx") in gates
        assert ("after_export", "torch.export") in gates


def test_operator_surface_is_nontrivial_and_deployment_specific():
    rows = {row["model"]: row for row in dg.manifest()["models"]}
    assert "ResidualAdd" in rows["resnet_residual_stage"]["operator_surface"]
    assert "Embedding" in rows["llama_style_mlp_block"]["operator_surface"]
    assert "Embedding" in rows["recommender_two_tower"]["operator_surface"]
    assert "ConvTranspose2d" in rows["diffusion_unet_skip"]["operator_surface"]
    assert "GRU" in rows["speech_conv_gru_encoder"]["operator_surface"]
    assert all(len(row["operator_surface"]) >= 4 for row in rows.values())


def test_committed_manifest_is_up_to_date():
    assert dg.run(check=True) == 0


def test_gallery_models_execute_with_expected_shapes():
    import torch

    for spec in dg.MODEL_SPECS:
        model, examples = dg._build_model(spec.name)
        with torch.no_grad():
            output = model(*examples)
        assert tuple(output.shape) == spec.expected_output_shape


@pytest.mark.slow
def test_live_deployment_gallery_gate_passes_supported_backends():
    rows = dg.measure()
    assert len(rows) == len(dg.MODEL_SPECS) * len(dg.GATE_SPECS)
    failed = [row for row in rows if row["status"] == "failed"]
    passed = [row for row in rows if row["status"] == "passed"]
    assert not failed
    assert {row["backend"] for row in passed} >= {"fx"}
    if any(row["backend"] == "torch.export" for row in passed):
        assert all(
            row["status"] == "passed"
            for row in rows
            if row["backend"] == "torch.export"
        )
    assert dg.gate() == 0
