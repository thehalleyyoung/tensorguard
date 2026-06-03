"""Step 209 — eager/FX quantization observer and placement gates."""

from __future__ import annotations

import pytest

from src.quantization_verify import (
    QuantizationVerdict,
    verify_quantization,
    verify_quantization_eager,
    verify_quantization_fx,
)

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

tq = pytest.importorskip("torch.ao.quantization")
from torch.fx import symbolic_trace  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:torch.ao.quantization is deprecated:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:Please use quant_min and quant_max:UserWarning"),
]


def _quant_engine() -> str:
    engines = [engine for engine in torch.backends.quantized.supported_engines if engine != "none"]
    if not engines:
        pytest.skip("no quantized engine available")
    engine = "qnnpack" if "qnnpack" in engines else engines[0]
    torch.backends.quantized.engine = engine
    return engine


def _default_qconfig():
    return tq.get_default_qconfig(_quant_engine())


class _WithStubsLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = tq.QuantStub()
        self.fc = nn.Linear(4, 3)
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.fc(self.quant(x)))


class _NoStubsLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class _NoDeQuantLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = tq.QuantStub()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(self.quant(x))


def _prepare(model: nn.Module, *, qconfig=None, calibrate: bool = True):
    model.eval()
    model.qconfig = qconfig or _default_qconfig()
    prepared = tq.prepare(model, inplace=False)
    if calibrate:
        with torch.no_grad():
            prepared(torch.randn(2, 4))
    return prepared


def _convert(model: nn.Module, *, qconfig=None):
    return tq.convert(_prepare(model, qconfig=qconfig, calibrate=True), inplace=False)


def _issue_kinds(verdict: QuantizationVerdict) -> set[str]:
    return {issue.kind for issue in verdict.issues}


def test_eager_prepared_model_requires_calibrated_observers_then_passes_after_real_data():
    prepared = _prepare(_WithStubsLinear(), calibrate=False)
    uncalibrated = verify_quantization_eager(prepared)
    assert not uncalibrated.ok
    assert "calibration_state" in _issue_kinds(uncalibrated)

    with torch.no_grad():
        prepared(torch.randn(2, 4))

    calibrated = verify_quantization_eager(prepared)
    assert calibrated.ok

    converted = tq.convert(prepared, inplace=False)
    converted_verdict = verify_quantization_eager(converted)
    assert converted_verdict.ok
    out = converted(torch.randn(2, 4))
    assert tuple(out.shape) == (2, 3)
    assert not out.is_quantized


def test_activation_per_channel_qscheme_is_rejected_before_real_torch_failure():
    bad_qconfig = tq.QConfig(
        activation=tq.PerChannelMinMaxObserver.with_args(
            ch_axis=0,
            dtype=torch.quint8,
            qscheme=torch.per_channel_affine,
        ),
        weight=tq.default_weight_observer,
    )
    prepared = _prepare(_WithStubsLinear(), qconfig=bad_qconfig, calibrate=False)
    verdict = verify_quantization_eager(prepared, require_calibrated=False)
    assert not verdict.ok
    assert "qscheme_placement" in _issue_kinds(verdict)

    with torch.no_grad():
        prepared(torch.randn(2, 4))
    with pytest.raises(RuntimeError):
        converted = tq.convert(prepared, inplace=False)
        converted(torch.randn(2, 4))


def test_eager_converted_model_without_stubs_is_rejected_and_really_fails():
    converted = _convert(_NoStubsLinear())
    verdict = verify_quantization_eager(converted)
    assert not verdict.ok
    assert {"missing_quantstub", "missing_dequantstub"} <= _issue_kinds(verdict)

    with pytest.raises((RuntimeError, NotImplementedError)):
        converted(torch.randn(2, 4))


def test_eager_missing_dequantstub_leaks_quantized_output():
    converted = _convert(_NoDeQuantLinear())
    verdict = verify_quantization_eager(converted)
    assert not verdict.ok
    assert "missing_dequantstub" in _issue_kinds(verdict)

    out = converted(torch.randn(2, 4))
    assert out.is_quantized
    assert verify_quantization_eager(converted, require_float_output=False).ok


def test_fx_graph_mode_rejects_quantized_linear_without_quantize_node():
    converted = _convert(_NoStubsLinear())
    gm = symbolic_trace(converted)
    verdict = verify_quantization_fx(gm)
    assert not verdict.ok
    assert "missing_quantstub" in _issue_kinds(verdict)

    with pytest.raises((RuntimeError, NotImplementedError)):
        gm(torch.randn(2, 4))


def test_fx_graph_mode_accepts_real_quantize_quantized_dequantize_chain():
    converted = _convert(_WithStubsLinear())
    gm = symbolic_trace(converted)
    verdict = verify_quantization_fx(gm)
    assert verdict.ok
    assert verify_quantization(gm).mode == "fx"
    out = gm(torch.randn(2, 4))
    assert tuple(out.shape) == (2, 3)
    assert not out.is_quantized


def test_fx_graph_mode_rejects_quantized_public_output():
    converted = _convert(_NoDeQuantLinear())
    gm = symbolic_trace(converted)
    verdict = verify_quantization_fx(gm)
    assert not verdict.ok
    assert "missing_dequantstub" in _issue_kinds(verdict)

    out = gm(torch.randn(2, 4))
    assert out.is_quantized
    assert verify_quantization_fx(gm, require_float_output=False).ok


class _BadQuantizeScale(nn.Module):
    def forward(self, x):
        return torch.quantize_per_tensor(x, 0.0, 0, torch.quint8)


def test_fx_quantize_per_tensor_scale_gate_rejects_nonpositive_scale():
    gm = symbolic_trace(_BadQuantizeScale())
    verdict = verify_quantization_fx(gm, require_float_output=False)
    assert not verdict.ok
    assert "qparams" in _issue_kinds(verdict)
    out = gm(torch.randn(2, 4))
    assert out.is_quantized
    assert out.q_scale() == 0.0


def test_public_exports_quantization_contract():
    import src
    import tensorguard

    assert src.verify_quantization_eager is verify_quantization_eager
    assert tensorguard.verify_quantization_fx is verify_quantization_fx
    assert isinstance(verify_quantization(_WithStubsLinear()), QuantizationVerdict)
