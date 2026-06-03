"""Step 214 - ONNX shape-inference round-trip validation.

The guarded exporter now compares TensorGuard's graph-level output-shape
prediction from ``torch.export`` metadata against ``onnx.shape_inference`` on
the just-written artifact.  The check is intentionally conservative: concrete
dimension disagreements fail, while symbolic or uninferred axes are skipped.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn

onnx = pytest.importorskip("onnx")

from src.torch_integration import (  # noqa: E402
    TensorGuardONNXShapeInferenceError,
    _post_export_check,
    guarded_onnx_export,
    verify_onnx_export_contract,
)


class CleanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 12)
        self.b = nn.Linear(12, 5)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


def _export_clean() -> tuple[io.BytesIO, tuple[tuple[int, ...], ...]]:
    model = CleanNet().eval()
    x = torch.randn(2, 10)
    gate = verify_onnx_export_contract(model, (x,))
    assert gate.predicted_output_shapes == ((2, 5),)
    buf = io.BytesIO()
    guarded_onnx_export(model, (x,), buf)
    return buf, gate.predicted_output_shapes


def test_real_onnx_shape_inference_matches_tensorguard_prediction():
    buf, expected = _export_clean()

    checks = _post_export_check(buf, expected_output_shapes=expected)

    assert len(checks) == 1
    assert checks[0].tensorguard_shape == (2, 5)
    assert checks[0].onnx_shape == (2, 5)
    assert checks[0].compared_axes == (0, 1)
    assert checks[0].matched is True


def test_concrete_onnx_shape_inference_mismatch_rejects(monkeypatch):
    real_infer_shapes = onnx.shape_inference.infer_shapes

    def _corrupt_feature_dim(model_proto, *args, **kwargs):
        inferred = real_infer_shapes(model_proto, *args, **kwargs)
        inferred.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 7
        return inferred

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", _corrupt_feature_dim)

    with pytest.raises(TensorGuardONNXShapeInferenceError) as exc:
        guarded_onnx_export(CleanNet().eval(), (torch.randn(2, 10),), io.BytesIO())

    issue = exc.value.issues[0]
    assert issue.category == "shape_inference_roundtrip"
    assert issue.output_name is not None
    assert "TensorGuard predicted 5" in issue.message
    assert exc.value.checks[0].matched is False


def test_symbolic_onnx_axes_are_skipped_but_concrete_axes_still_checked(monkeypatch):
    real_infer_shapes = onnx.shape_inference.infer_shapes

    def _symbolic_batch(model_proto, *args, **kwargs):
        inferred = real_infer_shapes(model_proto, *args, **kwargs)
        dim = inferred.graph.output[0].type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"
        return inferred

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", _symbolic_batch)

    buf = io.BytesIO()
    guarded_onnx_export(CleanNet().eval(), (torch.randn(2, 10),), buf)
    checks = _post_export_check(buf, expected_output_shapes=((2, 5),))

    assert checks[0].onnx_shape == (None, 5)
    assert checks[0].compared_axes == (1,)
    assert checks[0].matched is True


def test_shape_roundtrip_can_be_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        onnx.shape_inference,
        "infer_shapes",
        lambda model_proto, *args, **kwargs: calls.append(model_proto) or model_proto,
    )

    guarded_onnx_export(
        CleanNet().eval(),
        (torch.randn(2, 10),),
        io.BytesIO(),
        check_shape_roundtrip=False,
    )

    assert calls == []


def test_public_tensorguard_torch_exports_shape_roundtrip_types():
    from tensorguard.torch import (
        ONNXShapeRoundTripCheck,
        TensorGuardONNXShapeInferenceError as PublicError,
    )

    assert PublicError is TensorGuardONNXShapeInferenceError
    assert ONNXShapeRoundTripCheck.__name__ == "ONNXShapeRoundTripCheck"
