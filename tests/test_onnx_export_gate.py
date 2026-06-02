"""Step 167 — ONNX export gate, proven against the real ``torch.onnx`` exporter.

``guarded_onnx_export`` runs TensorGuard's static verification as the **first**
side effect, then delegates to ``torch.onnx.export``.  A real shape bug becomes
one ``TensorGuardViolation`` *before* anything is written to the export sink,
instead of a deep tracer error or a silently malformed graph.  We prove:

* a clean model exports to a ``BytesIO`` and the resulting bytes are a *valid*
  ONNX proto (``onnx.checker.check_model`` passes on the parsed graph);
* a buggy model raises ``TensorGuardViolation`` and **nothing is written** to
  the sink (the buffer stays empty);
* when ``input_shapes`` is omitted it is inferred from the example ``args`` so
  the shape that is verified is the shape that is exported.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn

onnx = pytest.importorskip("onnx")

from src.torch_integration import (  # noqa: E402
    TensorGuardViolation,
    _infer_shapes_from_args,
    guarded_onnx_export,
)


class CleanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 20)
        self.b = nn.Linear(20, 5)

    def forward(self, x):
        return self.b(self.a(x)).relu()


class BuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 20)
        self.b = nn.Linear(30, 5)  # expects 30, gets 20 -> real shape bug

    def forward(self, x):
        return self.b(self.a(x))


def test_infer_shapes_symbolises_batch():
    shapes = _infer_shapes_from_args(CleanNet(), (torch.randn(4, 10),))
    assert shapes == {"x": ("b", 10)}


def test_clean_model_exports_valid_onnx():
    buf = io.BytesIO()
    model = CleanNet().eval()
    guarded_onnx_export(model, (torch.randn(2, 10),), buf, input_shapes={"x": ("b", 10)})
    data = buf.getvalue()
    assert len(data) > 0
    proto = onnx.load_from_string(data)
    onnx.checker.check_model(proto)  # raises if the graph is malformed


def test_buggy_model_raises_before_writing_anything():
    buf = io.BytesIO()
    with pytest.raises(TensorGuardViolation):
        guarded_onnx_export(
            BuggyNet(), (torch.randn(2, 10),), buf, input_shapes={"x": ("b", 10)}
        )
    # Verification is the first side effect: the sink must be untouched.
    assert buf.getbuffer().nbytes == 0


def test_shapes_inferred_from_args_blocks_bug():
    buf = io.BytesIO()
    # No input_shapes passed: must be inferred from the example tensor.
    with pytest.raises(TensorGuardViolation):
        guarded_onnx_export(BuggyNet(), (torch.randn(3, 10),), buf)
    assert buf.getbuffer().nbytes == 0


def test_clean_export_to_path(tmp_path):
    model = CleanNet().eval()
    out = tmp_path / "clean.onnx"
    guarded_onnx_export(model, (torch.randn(2, 10),), str(out), input_shapes={"x": ("b", 10)})
    assert out.exists() and out.stat().st_size > 0
    onnx.checker.check_model(onnx.load(str(out)))
