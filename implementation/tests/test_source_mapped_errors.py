"""Tests for source_mapped_errors module."""

from __future__ import annotations

import json
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from src.source_mapped_errors import (
    RelatedLocation,
    SourceMappedDiagnostic,
    map_violations_to_diagnostics,
    format_plain,
    format_ansi,
    format_json,
    _get_snippet,
    _find_layer_def_line,
    _shape_str,
    _device_str,
    _severity_for,
)

# ---------------------------------------------------------------------------
# Attempt to import real model_checker types; fall back to lightweight stubs
# ---------------------------------------------------------------------------
try:
    from src.model_checker import (
        SafetyViolation,
        ComputationGraph,
        ComputationStep,
        LayerKind,
        OpKind,
        LayerDef,
        Device,
        Confidence,
    )
    _HAS_MC = True
except Exception:
    _HAS_MC = False

# Lightweight stubs used when the real imports are unavailable or when we
# want precise control over field values.

class _StubConfidence:
    def __init__(self, value: str):
        self.value = value

class _StubDevice:
    def __init__(self, value: str):
        self.value = value

class _StubOpKind:
    def __init__(self, name: str):
        self.name = name

class _StubLayerKind:
    def __init__(self, name: str):
        self.name = name

@dataclass
class _StubLayerDef:
    attr_name: str = ""
    kind: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    line: int = 0
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None

@dataclass
class _StubStep:
    op: Any = None
    inputs: List[str] = field(default_factory=list)
    output: str = ""
    layer_ref: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    line: int = 0
    col: int = 0

@dataclass
class _StubViolation:
    kind: str = ""
    step_index: int = 0
    step: Any = None
    message: str = ""
    tensor_a: Optional[str] = None
    tensor_b: Optional[str] = None
    shape_a: Any = None
    shape_b: Any = None
    device_a: Any = None
    device_b: Any = None
    confidence: Any = None
    fp_category: Optional[str] = None

@dataclass
class _StubGraph:
    class_name: str = "TestModel"
    layers: Dict[str, Any] = field(default_factory=dict)
    steps: List[Any] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

# If real model_checker types are present, also create factory helpers that
# produce real instances.
if _HAS_MC:
    def _real_step(**kw) -> ComputationStep:
        op = kw.pop("op", OpKind.LAYER_CALL)
        return ComputationStep(op=op, inputs=kw.pop("inputs", []), output=kw.pop("output", ""), **kw)

    def _real_violation(**kw) -> SafetyViolation:
        if "step" not in kw:
            kw["step"] = _real_step()
        if "step_index" not in kw:
            kw["step_index"] = 0
        if "confidence" not in kw:
            kw["confidence"] = Confidence.HIGH
        return SafetyViolation(**kw)


# ---------------------------------------------------------------------------
# Sample source snippets used across tests
# ---------------------------------------------------------------------------

SIMPLE_SOURCE = """\
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

CONV_SOURCE = """\
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
"""

DEVICE_SOURCE = """\
import torch
import torch.nn as nn

class DeviceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x)
        return x
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════════════════════

def _make_step(line=0, col=0, layer_ref=None, op_name="LAYER_CALL", inputs=None, output=""):
    return _StubStep(
        op=_StubOpKind(op_name),
        inputs=inputs or [],
        output=output,
        layer_ref=layer_ref,
        line=line,
        col=col,
    )


def _make_violation(kind="shape_incompatible", line=11, col=8, layer_ref="fc1",
                    message="", tensor_a=None, tensor_b=None, shape_a=None,
                    shape_b=None, device_a=None, device_b=None,
                    confidence=None, fp_category=None, op_name="LAYER_CALL"):
    step = _make_step(line=line, col=col, layer_ref=layer_ref, op_name=op_name)
    return _StubViolation(
        kind=kind,
        step_index=0,
        step=step,
        message=message,
        tensor_a=tensor_a,
        tensor_b=tensor_b,
        shape_a=shape_a,
        shape_b=shape_b,
        device_a=device_a,
        device_b=device_b,
        confidence=confidence or _StubConfidence("high"),
        fp_category=fp_category,
    )


def _make_graph(layers=None, class_name="TestModel"):
    return _StubGraph(class_name=class_name, layers=layers or {})


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — snippet extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestSnippetExtraction:
    def test_valid_line(self):
        assert "self.fc1" in _get_snippet(SIMPLE_SOURCE, 7)

    def test_first_line(self):
        assert "import torch" in _get_snippet(SIMPLE_SOURCE, 1)

    def test_last_line(self):
        # last non-empty line
        snippet = _get_snippet(SIMPLE_SOURCE, 13)
        assert "return" in snippet

    def test_out_of_range_zero(self):
        assert _get_snippet(SIMPLE_SOURCE, 0) == ""

    def test_out_of_range_large(self):
        assert _get_snippet(SIMPLE_SOURCE, 9999) == ""

    def test_empty_source(self):
        assert _get_snippet("", 1) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — layer def line finding
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindLayerDefLine:
    def test_find_fc1(self):
        assert _find_layer_def_line(SIMPLE_SOURCE, "fc1") == 7

    def test_find_fc2(self):
        assert _find_layer_def_line(SIMPLE_SOURCE, "fc2") == 8

    def test_not_found(self):
        assert _find_layer_def_line(SIMPLE_SOURCE, "nonexistent") == 0

    def test_conv_layer(self):
        assert _find_layer_def_line(CONV_SOURCE, "conv1") == 7


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — shape / device string formatting
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapeStr:
    def test_none(self):
        assert _shape_str(None) == "unknown"

    def test_tuple(self):
        assert _shape_str((3, 224, 224)) == "(3, 224, 224)"

    def test_list(self):
        assert _shape_str([1, 2, 3]) == "(1, 2, 3)"

    def test_object_with_dims(self):
        class FakeDim:
            def __init__(self, v):
                self.symbol = None
                self.value = v
        class FakeShape:
            def __init__(self, ds):
                self.dims = [FakeDim(d) for d in ds]
        assert _shape_str(FakeShape([3, 64])) == "(3, 64)"


class TestDeviceStr:
    def test_none(self):
        assert _device_str(None) == "unknown"

    def test_enum_like(self):
        assert _device_str(_StubDevice("cuda:0")) == "cuda:0"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — severity mapping
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeverity:
    def test_high_confidence_error(self):
        v = _make_violation(confidence=_StubConfidence("high"))
        assert _severity_for(v) == "error"

    def test_medium_confidence_warning(self):
        v = _make_violation(confidence=_StubConfidence("medium"))
        assert _severity_for(v) == "warning"

    def test_low_confidence_info(self):
        v = _make_violation(confidence=_StubConfidence("low"))
        assert _severity_for(v) == "info"

    def test_fp_category_is_warning(self):
        v = _make_violation(fp_category="missing_stdlib")
        assert _severity_for(v) == "warning"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — shape mismatch diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapeMismatch:
    def test_layer_call_with_in_features(self):
        layer_def = _StubLayerDef(attr_name="fc1", in_features=256, out_features=128)
        graph = _make_graph(layers={"fc1": layer_def})
        v = _make_violation(
            kind="shape_incompatible",
            line=11, col=8, layer_ref="fc1",
            tensor_a="x", shape_a=(512,),
        )
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [v], graph)
        assert len(diags) == 1
        d = diags[0]
        assert d.source_line == 11
        assert "expects input dimension 256" in d.message
        assert "(512,)" in d.message or "512" in d.message
        assert d.severity == "error"

    def test_layer_call_no_graph(self):
        v = _make_violation(
            kind="shape_incompatible",
            line=11, col=8, layer_ref="fc1",
            tensor_a="x", shape_a=(512,),
        )
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [v], None)
        assert len(diags) == 1
        assert "fc1" in diags[0].message

    def test_non_layer_op(self):
        v = _make_violation(
            kind="shape_incompatible",
            line=14, col=8, layer_ref=None,
            tensor_a="a", tensor_b="b",
            shape_a=(4, 64), shape_b=(4, 128),
            op_name="ADD",
        )
        diags = map_violations_to_diagnostics(CONV_SOURCE, [v])
        assert len(diags) == 1
        assert "ADD" in diags[0].message
        assert "a" in diags[0].message

    def test_related_location_populated(self):
        layer_def = _StubLayerDef(attr_name="fc1", in_features=256)
        graph = _make_graph(layers={"fc1": layer_def})
        v = _make_violation(
            kind="shape_incompatible", line=11, layer_ref="fc1",
            tensor_a="x", shape_a=(128,),
        )
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [v], graph)
        assert len(diags[0].related_locations) >= 1
        assert diags[0].related_locations[0].line == 7

    def test_fix_suggestion_for_shape(self):
        layer_def = _StubLayerDef(attr_name="fc1", in_features=256)
        graph = _make_graph(layers={"fc1": layer_def})
        v = _make_violation(
            kind="shape_incompatible", line=11, layer_ref="fc1",
            tensor_a="x", shape_a=(128,),
        )
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [v], graph)
        assert diags[0].fix_suggestion is not None
        assert "256" in diags[0].fix_suggestion


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — broadcast diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

class TestBroadcastErrors:
    def test_broadcast_message(self):
        v = _make_violation(
            kind="broadcast_incompatible",
            line=12, col=8, layer_ref=None,
            tensor_a="a", tensor_b="b",
            shape_a=(4, 64), shape_b=(4, 128),
            op_name="ADD",
        )
        diags = map_violations_to_diagnostics(CONV_SOURCE, [v])
        assert len(diags) == 1
        assert "Broadcast incompatibility" in diags[0].message
        assert "64" in diags[0].message
        assert "128" in diags[0].message

    def test_broadcast_fix_suggestion(self):
        v = _make_violation(
            kind="broadcast_incompatible",
            line=12, col=8, layer_ref=None,
            tensor_a="a", tensor_b="b",
            shape_a=(4, 64), shape_b=(4, 128),
            op_name="ADD",
        )
        diags = map_violations_to_diagnostics(CONV_SOURCE, [v])
        assert diags[0].fix_suggestion is not None
        assert "Reshape" in diags[0].fix_suggestion or "reshape" in diags[0].fix_suggestion.lower()

    def test_broadcast_dims_detail(self):
        v = _make_violation(
            kind="broadcast_incompatible",
            line=12, layer_ref=None,
            tensor_a="a", tensor_b="b",
            shape_a=(4, 64), shape_b=(4, 128),
        )
        diags = map_violations_to_diagnostics(CONV_SOURCE, [v])
        assert "neither equal nor 1" in diags[0].message


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — device mismatch diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeviceErrors:
    def test_device_mismatch_layer(self):
        v = _make_violation(
            kind="device_mismatch",
            line=11, col=8, layer_ref="conv1",
            tensor_a="conv1", tensor_b="x",
            device_a=_StubDevice("cuda:0"),
            device_b=_StubDevice("cpu"),
        )
        diags = map_violations_to_diagnostics(DEVICE_SOURCE, [v])
        assert len(diags) == 1
        assert "Device mismatch" in diags[0].message
        assert "cuda:0" in diags[0].message
        assert "cpu" in diags[0].message

    def test_device_fix_suggestion(self):
        v = _make_violation(
            kind="device_mismatch",
            line=11, col=8, layer_ref="conv1",
            device_a=_StubDevice("cuda:0"),
            device_b=_StubDevice("cpu"),
        )
        diags = map_violations_to_diagnostics(DEVICE_SOURCE, [v])
        assert diags[0].fix_suggestion is not None
        assert ".to(" in diags[0].fix_suggestion

    def test_device_no_layer(self):
        v = _make_violation(
            kind="device_mismatch",
            line=11, col=8, layer_ref=None,
            tensor_a="a", tensor_b="b",
            device_a=_StubDevice("cuda:0"),
            device_b=_StubDevice("cpu"),
            op_name="ADD",
        )
        diags = map_violations_to_diagnostics(DEVICE_SOURCE, [v])
        assert "Device mismatch" in diags[0].message


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — gradient diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradientErrors:
    def test_gradient_message(self):
        v = _make_violation(
            kind="gradient_invalid",
            line=11, col=8, layer_ref=None,
            message="parameter requires_grad=True but detach() was called",
        )
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [v])
        assert len(diags) == 1
        assert "Gradient error" in diags[0].message
        assert "detach" in diags[0].message


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — generic / unknown violation kind
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenericViolation:
    def test_unknown_kind(self):
        v = _make_violation(
            kind="phase_error",
            line=11, col=8, layer_ref=None,
            message="dropout used in eval mode",
        )
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [v])
        assert len(diags) == 1
        assert "Verification error" in diags[0].message
        assert "dropout" in diags[0].message


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — plain-text formatting
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlainFormat:
    def test_plain_output_contains_severity(self):
        d = SourceMappedDiagnostic(
            message="test message",
            severity="error",
            source_line=5,
            source_col=0,
            source_snippet="x = 1",
        )
        out = format_plain([d])
        assert "ERROR" in out
        assert "test message" in out

    def test_plain_output_line_col(self):
        d = SourceMappedDiagnostic(
            message="msg", severity="warning",
            source_line=10, source_col=4,
            source_snippet="y = z",
        )
        out = format_plain([d])
        assert "line 10" in out
        assert "col 4" in out

    def test_plain_related_locations(self):
        d = SourceMappedDiagnostic(
            message="msg", severity="error",
            source_line=10, source_col=0,
            source_snippet="code",
            related_locations=[
                RelatedLocation(message="defined here", line=3, col=0, snippet="def x"),
            ],
        )
        out = format_plain([d])
        assert "note:" in out
        assert "defined here" in out

    def test_plain_fix(self):
        d = SourceMappedDiagnostic(
            message="msg", severity="error",
            source_line=1, source_col=0,
            source_snippet="",
            fix_suggestion="try this",
        )
        out = format_plain([d])
        assert "fix:" in out
        assert "try this" in out


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — ANSI formatting
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnsiFormat:
    def test_ansi_contains_escape_codes(self):
        d = SourceMappedDiagnostic(
            message="test", severity="error",
            source_line=1, source_col=0,
            source_snippet="code",
        )
        out = format_ansi([d])
        assert "\033[" in out

    def test_ansi_error_red(self):
        d = SourceMappedDiagnostic(
            message="test", severity="error",
            source_line=1, source_col=0,
            source_snippet="code",
        )
        out = format_ansi([d])
        assert "\033[31m" in out  # red

    def test_ansi_warning_yellow(self):
        d = SourceMappedDiagnostic(
            message="test", severity="warning",
            source_line=1, source_col=0,
            source_snippet="code",
        )
        out = format_ansi([d])
        assert "\033[33m" in out  # yellow

    def test_ansi_info_cyan(self):
        d = SourceMappedDiagnostic(
            message="test", severity="info",
            source_line=1, source_col=0,
            source_snippet="code",
        )
        out = format_ansi([d])
        assert "\033[36m" in out  # cyan

    def test_ansi_related_and_fix(self):
        d = SourceMappedDiagnostic(
            message="msg", severity="error",
            source_line=1, source_col=0,
            source_snippet="code",
            related_locations=[
                RelatedLocation(message="see here", line=5, col=0, snippet="other"),
            ],
            fix_suggestion="do something",
        )
        out = format_ansi([d])
        assert "see here" in out
        assert "do something" in out


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — JSON formatting
# ═══════════════════════════════════════════════════════════════════════════════

class TestJsonFormat:
    def test_json_parses(self):
        d = SourceMappedDiagnostic(
            message="msg", severity="error",
            source_line=1, source_col=0,
            source_snippet="code",
        )
        out = format_json([d])
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["message"] == "msg"

    def test_json_related_locations(self):
        d = SourceMappedDiagnostic(
            message="msg", severity="error",
            source_line=1, source_col=0,
            source_snippet="code",
            related_locations=[
                RelatedLocation(message="note", line=5, col=2, snippet="s"),
            ],
            fix_suggestion="fix it",
        )
        out = format_json([d])
        parsed = json.loads(out)
        assert len(parsed[0]["related_locations"]) == 1
        assert parsed[0]["related_locations"][0]["line"] == 5
        assert parsed[0]["fix_suggestion"] == "fix it"

    def test_json_multiple_diagnostics(self):
        diags = [
            SourceMappedDiagnostic(
                message=f"msg{i}", severity="error",
                source_line=i, source_col=0, source_snippet="",
            )
            for i in range(3)
        ]
        out = format_json(diags)
        parsed = json.loads(out)
        assert len(parsed) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — integration with real nn.Module source
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealSourceIntegration:
    """End-to-end tests with realistic source and multiple violations."""

    def test_multiple_violations(self):
        violations = [
            _make_violation(
                kind="shape_incompatible", line=11, col=8,
                layer_ref="fc1", tensor_a="x", shape_a=(512,),
            ),
            _make_violation(
                kind="device_mismatch", line=12, col=8,
                layer_ref="fc2",
                device_a=_StubDevice("cuda:0"),
                device_b=_StubDevice("cpu"),
            ),
        ]
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, violations)
        assert len(diags) == 2
        assert diags[0].source_line == 11
        assert diags[1].source_line == 12

    def test_snippet_matches_source_line(self):
        v = _make_violation(kind="shape_incompatible", line=11, layer_ref="fc1",
                            tensor_a="x", shape_a=(512,))
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [v])
        assert "self.fc1" in diags[0].source_snippet

    def test_conv_source_snippet(self):
        v = _make_violation(kind="shape_incompatible", line=12, layer_ref="conv1",
                            tensor_a="x", shape_a=(3, 224, 224))
        diags = map_violations_to_diagnostics(CONV_SOURCE, [v])
        assert "self.conv1" in diags[0].source_snippet

    def test_empty_violations_list(self):
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [])
        assert diags == []

    def test_diagnostic_dataclass_fields(self):
        v = _make_violation(kind="shape_incompatible", line=11, col=8,
                            layer_ref="fc1", tensor_a="x", shape_a=(512,))
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [v])
        d = diags[0]
        assert isinstance(d.message, str)
        assert isinstance(d.severity, str)
        assert isinstance(d.source_line, int)
        assert isinstance(d.source_col, int)
        assert isinstance(d.source_snippet, str)
        assert isinstance(d.related_locations, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_violation_with_zero_line(self):
        v = _make_violation(kind="shape_incompatible", line=0, col=0,
                            layer_ref="fc1", tensor_a="x", shape_a=(10,))
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [v])
        assert diags[0].source_line == 0
        assert diags[0].source_snippet == ""

    def test_none_graph(self):
        v = _make_violation(kind="shape_incompatible", line=11, layer_ref="fc1",
                            tensor_a="x", shape_a=(10,))
        diags = map_violations_to_diagnostics(SIMPLE_SOURCE, [v], None)
        assert len(diags) == 1

    def test_format_plain_empty(self):
        assert format_plain([]) == ""

    def test_format_ansi_empty(self):
        assert format_ansi([]) == ""

    def test_format_json_empty(self):
        assert format_json([]) == "[]"
