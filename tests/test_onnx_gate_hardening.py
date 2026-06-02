"""Step 171 — ONNX export gate *hardening*, proven against the real exporter.

Step 167 shipped the gate (verify-before-export).  This step hardens it with a
**post-export ``onnx.checker.check_model`` assertion mode** and an opt-in
Dynamo/``onnxscript`` exporter path.  We prove:

* with ``check_model=True`` (default) the gate parses the just-written proto and
  runs ``onnx.checker.check_model`` (observed via a spy) — and on a *real* clean
  export the genuine checker passes;
* ``check_model=False`` skips the post-export assertion (spy never called);
* if ``check_model`` rejects the graph, the gate surfaces that error (a malformed
  graph fails loudly at export time, not at downstream load time);
* the exporter is invoked with ``dynamo=False`` by default and forwards an
  explicit ``dynamo=True`` opt-in unchanged.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn

onnx = pytest.importorskip("onnx")

from src.torch_integration import guarded_onnx_export  # noqa: E402


class CleanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 20)
        self.b = nn.Linear(20, 5)

    def forward(self, x):
        return self.b(self.a(x)).relu()


def test_check_model_runs_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(onnx.checker, "check_model", lambda m: calls.append(m))
    guarded_onnx_export(CleanNet(), (torch.randn(2, 10),), io.BytesIO())
    assert len(calls) == 1  # post-export assertion fired exactly once


def test_check_model_can_be_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(onnx.checker, "check_model", lambda m: calls.append(m))
    guarded_onnx_export(
        CleanNet(), (torch.randn(2, 10),), io.BytesIO(), check_model=False
    )
    assert calls == []  # assertion skipped


def test_real_checker_passes_on_clean_export():
    buf = io.BytesIO()
    guarded_onnx_export(CleanNet(), (torch.randn(2, 10),), buf)  # genuine checker, no spy
    assert buf.getbuffer().nbytes > 0
    onnx.checker.check_model(onnx.load_from_string(buf.getvalue()))  # independently valid


def test_invalid_graph_surfaces_at_export_time(monkeypatch):
    class _Boom(Exception):
        pass

    def _reject(_m):
        raise _Boom("structurally invalid graph")

    monkeypatch.setattr(onnx.checker, "check_model", _reject)
    with pytest.raises(_Boom):
        guarded_onnx_export(CleanNet(), (torch.randn(2, 10),), io.BytesIO())


def test_dynamo_flag_default_false_and_opt_in_forwarded(monkeypatch):
    seen = {}

    def _fake_export(model, args, f, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr("torch.onnx.export", _fake_export)
    guarded_onnx_export(CleanNet(), (torch.randn(2, 10),), io.BytesIO(), check_model=False)
    assert seen.get("dynamo") is False  # default

    seen.clear()
    guarded_onnx_export(
        CleanNet(), (torch.randn(2, 10),), io.BytesIO(), check_model=False, dynamo=True
    )
    assert seen.get("dynamo") is True  # explicit opt-in forwarded
