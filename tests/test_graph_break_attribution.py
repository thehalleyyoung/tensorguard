"""Step 225 — graph-break attribution for Dynamo/export failures."""

from __future__ import annotations

import pytest

from src.graph_break_attribution import classify_graph_break_failure
from src.model_checker import VerificationResult


_BRANCH_SOURCE = """
import torch
import torch.nn as nn

class BranchNet(nn.Module):
    def forward(self, x):
        if x.mean() > 0:
            return x + 1
        return x - 1
"""


def test_data_dependent_branch_maps_to_fragment_category_and_fix():
    report = classify_graph_break_failure(
        _BRANCH_SOURCE,
        "Could not guard on data-dependent expression Eq(u0, 1)",
        backend="export",
    )

    assert report.source_available is True
    assert report.attributions
    first = report.attributions[0]
    assert first.category == "DATA_DEPENDENT_CONTROL_FLOW"
    assert first.line == 7
    assert first.snippet == "if x.mean() > 0:"
    assert "torch.cond" in first.minimal_change or "torch.where" in first.minimal_change


def test_tensor_scalar_extraction_gets_minimal_change():
    source = """
import torch.nn as nn

class ScalarNet(nn.Module):
    def forward(self, x):
        n = int(x.sum().item())
        return x.reshape(n, -1)
"""
    report = classify_graph_break_failure(
        source,
        "trying to get a value out of a symbolic int",
        backend="dynamo",
    )

    first = report.attributions[0]
    assert first.category == "TENSOR_TO_SCALAR"
    assert first.snippet == "n = int(x.sum().item())"
    assert ".item()" in first.minimal_change


def test_dynamic_assertion_is_source_mapped():
    source = """
import torch.nn as nn

class AssertNet(nn.Module):
    def forward(self, x):
        assert x.shape[0] > 0
        return x
"""
    report = classify_graph_break_failure(
        source,
        "torch.export assertion failed",
        backend="export",
    )

    first = report.attributions[0]
    assert first.category == "DYNAMIC_ASSERTION"
    assert first.line == 6
    assert first.snippet == "assert x.shape[0] > 0"
    assert "input-shape contract" in first.minimal_change


def test_source_unavailable_falls_back_to_message_only():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    Dynamic = type("Dynamic", (nn.Module,), {"forward": lambda self, x: x})
    report = classify_graph_break_failure(
        Dynamic(),
        "Could not guard on data-dependent expression",
        backend="dynamo",
    )

    assert report.source_available is False
    assert report.attributions
    assert report.attributions[0].category == "DATA_DEPENDENT_CONTROL_FLOW"
    assert report.attributions[0].confidence == "low"


def test_export_failure_result_carries_attribution(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import src.export_extractor as export_extractor

    class ExportBranch(nn.Module):
        def forward(self, x):
            if x.mean() > 0:
                return x + 1
            return x - 1

    def fail_export(*args, **kwargs):
        raise RuntimeError("Could not guard on data-dependent expression")

    monkeypatch.setattr(export_extractor, "HAS_EXPORT", True)
    monkeypatch.setattr(export_extractor, "_torch_export", fail_export)

    result = export_extractor.verify_module_export(
        ExportBranch(),
        input_shapes={"x": (2, 3)},
        example_inputs=(torch.randn(2, 3),),
    )

    report = result.dynamic_features["graph_break_attribution"]
    assert result.safe is False
    assert report["backend"] == "export"
    assert report["attributions"][0]["category"] == "DATA_DEPENDENT_CONTROL_FLOW"
    assert result.dynamic_feature_warnings


def test_dynamo_fallback_preserves_attribution(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import src.dynamo_extractor as dynamo_extractor

    class DynamoBranch(nn.Module):
        def forward(self, x):
            if x.mean() > 0:
                return x + 1
            return x - 1

    def fail_dynamo(*args, **kwargs):
        raise RuntimeError("Could not guard on data-dependent expression")

    def fake_fx_verify(*args, **kwargs):
        return VerificationResult(safe=True, errors=[])

    monkeypatch.setattr(dynamo_extractor, "HAS_DYNAMO", True)
    monkeypatch.setattr(dynamo_extractor, "dynamo_trace_to_graph", fail_dynamo)
    monkeypatch.setattr(dynamo_extractor, "fx_verify_module", fake_fx_verify)

    result = dynamo_extractor.verify_module_dynamo(
        DynamoBranch(),
        input_shapes={"x": (2, 3)},
        example_inputs=(torch.randn(2, 3),),
        fallback_to_fx=True,
    )

    report = result.dynamic_features["graph_break_attribution"]
    assert result.safe is True
    assert report["backend"] == "dynamo"
    assert report["fallback_used"] == "fx"
    assert report["attributions"][0]["category"] == "DATA_DEPENDENT_CONTROL_FLOW"


def test_public_exports_available():
    import src
    import tensorguard
    import tensorguard.torch as tg_torch

    assert src.classify_graph_break_failure is classify_graph_break_failure
    assert tensorguard.classify_graph_break_failure is classify_graph_break_failure
    assert tg_torch.classify_graph_break_failure is classify_graph_break_failure
