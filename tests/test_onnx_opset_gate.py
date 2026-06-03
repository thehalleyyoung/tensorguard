"""Step 213 - ONNX opset availability gates.

These tests avoid requiring the optional ``onnx`` package: they prove the new
pre-export gate with real ``torch.export`` lowered graphs and monkeypatch the
final ``torch.onnx.export`` call only to assert whether the exporter is reached.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn

from src.torch_integration import (
    TensorGuardONNXExportError,
    guarded_onnx_export,
    verify_exported_program,
    verify_onnx_export_contract,
)


class TriuNet(nn.Module):
    def forward(self, x):
        return torch.triu(x)


class EinsumNet(nn.Module):
    def forward(self, x, y):
        return torch.einsum("bd,df->bf", x, y)


class Identity2D(nn.Module):
    def forward(self, x):
        return x.relu()


class EchoNet(nn.Module):
    def forward(self, x):
        return x


def test_triu_opset_gate_rejects_before_exporter(monkeypatch):
    reached = []

    def _fake_export(*args, **kwargs):
        reached.append(kwargs)

    monkeypatch.setattr(torch.onnx, "export", _fake_export)
    with pytest.raises(TensorGuardONNXExportError) as exc:
        guarded_onnx_export(
            TriuNet(),
            (torch.randn(3, 3),),
            io.BytesIO(),
            opset_version=13,
            check_model=False,
        )

    assert reached == []
    issue = exc.value.issues[0]
    assert issue.category == "opset_version"
    assert issue.onnx_op == "Trilu"
    assert issue.min_opset == 14
    assert "triu" in (issue.op_name or "")


def test_triu_at_minimum_opset_reaches_exporter(monkeypatch):
    reached = []

    def _fake_export(*args, **kwargs):
        reached.append(kwargs)
        return "exported"

    monkeypatch.setattr(torch.onnx, "export", _fake_export)
    result = guarded_onnx_export(
        TriuNet(),
        (torch.randn(3, 3),),
        io.BytesIO(),
        opset_version=14,
        check_model=False,
    )

    assert result == "exported"
    assert len(reached) == 1
    assert reached[0]["opset_version"] == 14
    assert reached[0]["dynamo"] is False


def test_direct_gate_maps_real_lowered_einsum_op():
    gate = verify_onnx_export_contract(
        EinsumNet(),
        (torch.randn(2, 3), torch.randn(3, 4)),
        opset_version=11,
    )

    assert not gate.ok
    assert any(op.onnx_op == "Einsum" and op.min_opset == 12 for op in gate.checked_ops)
    issue = next(issue for issue in gate.issues if issue.category == "opset_version")
    assert issue.onnx_op == "Einsum"
    assert issue.min_opset == 12
    assert issue.requested_opset == 11


def test_dynamic_shapes_require_dynamo_exporter_before_onnx_export(monkeypatch):
    reached = []
    monkeypatch.setattr(torch.onnx, "export", lambda *a, **k: reached.append(k))
    b = torch.export.Dim("b", min=2, max=8)

    with pytest.raises(TensorGuardONNXExportError) as exc:
        guarded_onnx_export(
            Identity2D(),
            (torch.randn(3, 10),),
            io.BytesIO(),
            input_shapes={"x": ("b", 10)},
            dynamic_shapes={"x": {0: b}},
            check_model=False,
        )

    assert reached == []
    assert exc.value.issues[0].category == "dynamic_shape_export"
    assert "dynamo=True" in exc.value.issues[0].message


def test_onnx_rejects_derived_dim_relation_that_torch_export_accepts():
    b = torch.export.Dim("b", min=2, max=8)
    dynamic_shapes = {"x": {0: b, 1: 2 * b}}
    input_shapes = {"x": ("b", "2*b")}
    x = torch.randn(3, 6)

    ep = verify_exported_program(
        Identity2D(),
        (x,),
        input_shapes=input_shapes,
        dynamic_shapes=dynamic_shapes,
    )
    assert isinstance(ep, torch.export.ExportedProgram)

    gate = verify_onnx_export_contract(
        Identity2D(),
        (x,),
        input_shapes=input_shapes,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        opset_version=18,
    )
    assert not gate.ok
    assert any(
        issue.category == "dynamic_shape_export" and "derived" in issue.message
        for issue in gate.issues
    )


def test_graph_capture_failure_degrades_and_exporter_still_runs(monkeypatch):
    def _capture_fails(*args, **kwargs):
        raise RuntimeError("capture unavailable")

    reached = []
    monkeypatch.setattr(torch.export, "export", _capture_fails)
    monkeypatch.setattr(torch.onnx, "export", lambda *a, **k: reached.append(k) or "ok")

    gate = verify_onnx_export_contract(EchoNet(), (torch.randn(2, 3),))
    assert gate.ok
    assert gate.graph_capture_error == "RuntimeError: capture unavailable"
    assert gate.checked_ops == ()

    result = guarded_onnx_export(
        EchoNet(),
        (torch.randn(2, 3),),
        io.BytesIO(),
        check_model=False,
    )
    assert result == "ok"
    assert len(reached) == 1


def test_public_tensorguard_torch_exports_onnx_gate():
    from tensorguard.torch import (
        TensorGuardONNXExportError as PublicError,
        guarded_onnx_export as public_export,
        verify_onnx_export_contract as public_gate,
    )

    assert public_export is guarded_onnx_export
    assert public_gate is verify_onnx_export_contract
    assert PublicError is TensorGuardONNXExportError
