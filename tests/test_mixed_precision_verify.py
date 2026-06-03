"""Step 210 — mixed-precision/autocast dtype-divergence gates."""

from __future__ import annotations

import pytest

from src.mixed_precision_verify import (
    MixedPrecisionVerdict,
    verify_mixed_precision,
    verify_mixed_precision_fx,
)

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.fx import symbolic_trace  # noqa: E402


class _LinearOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def _issue_kinds(verdict: MixedPrecisionVerdict) -> set[str]:
    return {issue.kind for issue in verdict.issues}


def _entry(verdict: MixedPrecisionVerdict, op: str):
    for item in verdict.trace:
        if item.op == op:
            return item
    raise AssertionError(f"missing trace op {op!r}: {verdict.trace!r}")


def test_cpu_bfloat16_autocast_linear_policy_matches_real_torch():
    model = _LinearOnly().eval()
    verdict = verify_mixed_precision(model, backend="cpu", autocast_dtype=torch.bfloat16)

    assert verdict.ok
    assert _entry(verdict, "Linear").policy == "lower_precision"
    assert _entry(verdict, "Linear").predicted_dtype == "bfloat16"

    seen = {}

    def hook(_module, _inputs, output):
        seen["fc"] = output.dtype

    handle = model.fc.register_forward_hook(hook)
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(torch.randn(2, 4))
    finally:
        handle.remove()

    assert seen["fc"] is torch.bfloat16
    assert out.dtype is torch.bfloat16


def test_cpu_unsupported_autocast_dtype_is_flagged_and_pytorch_disables_it():
    model = _LinearOnly().eval()
    verdict = verify_mixed_precision(model, backend="cpu", autocast_dtype=torch.float32)

    assert not verdict.ok
    assert "unsupported_autocast_dtype" in _issue_kinds(verdict)

    with pytest.warns(UserWarning, match="CPU Autocast only supports"):
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.float32):
            out = model(torch.randn(2, 4))
    assert out.dtype is torch.float32


def test_half_trainable_parameters_are_rejected_before_real_dtype_failure():
    model = _LinearOnly().half().train()
    verdict = verify_mixed_precision(
        model,
        backend="cpu",
        autocast_dtype=torch.bfloat16,
        training=True,
        uses_grad_scaler=True,
    )

    assert not verdict.ok
    assert "parameter_not_fp32" in _issue_kinds(verdict)

    with pytest.raises(RuntimeError, match="same dtype|mat1 and mat2"):
        model(torch.randn(2, 4))


class _MixedParameterDtypes(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2).half()

    def forward(self, x):
        return self.b(self.a(x))


def test_heterogeneous_parameter_dtypes_are_flagged_and_really_fail():
    model = _MixedParameterDtypes().eval()
    verdict = verify_mixed_precision(model, backend="cpu", autocast_dtype=torch.bfloat16)

    assert not verdict.ok
    assert "parameter_dtype_mismatch" in _issue_kinds(verdict)

    with pytest.raises(RuntimeError, match="same dtype|mat1 and mat2"):
        model(torch.randn(2, 4))


class _ExplicitHalf(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x).half()


def test_explicit_reduced_precision_cast_is_visible_in_fx_and_real_output():
    model = _ExplicitHalf().eval()
    verdict = verify_mixed_precision(model, backend="cpu", autocast_dtype=torch.bfloat16)

    assert not verdict.ok
    assert "explicit_reduced_precision_cast" in _issue_kinds(verdict)
    assert _entry(verdict, "half").predicted_dtype == "float16"

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(torch.randn(2, 4))
    assert out.dtype is torch.float16


def test_cuda_fp16_training_requires_grad_scaler_but_bfloat16_does_not():
    model = _LinearOnly()
    missing = verify_mixed_precision(
        model,
        backend="cuda",
        autocast_dtype=torch.float16,
        training=True,
        uses_grad_scaler=False,
    )
    assert "amp_missing_grad_scaler" in _issue_kinds(missing)

    scaled = verify_mixed_precision(
        model,
        backend="cuda",
        autocast_dtype=torch.float16,
        training=True,
        uses_grad_scaler=True,
    )
    assert "amp_missing_grad_scaler" not in _issue_kinds(scaled)

    bf16 = verify_mixed_precision(
        model,
        backend="cuda",
        autocast_dtype=torch.bfloat16,
        training=True,
        uses_grad_scaler=False,
    )
    assert "amp_missing_grad_scaler" not in _issue_kinds(bf16)


def test_mps_backend_policy_is_checked_statically_without_hardware():
    model = _LinearOnly()
    supported = verify_mixed_precision(model, backend="mps", autocast_dtype=torch.float16)
    unsupported = verify_mixed_precision(model, backend="mps", autocast_dtype=torch.float64)

    assert "unsupported_autocast_dtype" not in _issue_kinds(supported)
    assert "unsupported_autocast_dtype" in _issue_kinds(unsupported)


def test_reduced_precision_public_output_boundary_can_be_required():
    verdict = verify_mixed_precision(
        _LinearOnly().eval(),
        backend="cpu",
        autocast_dtype=torch.bfloat16,
        require_float_output=True,
    )

    assert not verdict.ok
    assert "reduced_precision_output" in _issue_kinds(verdict)


def test_fx_verifier_uses_explicit_autocast_context():
    graph = symbolic_trace(_LinearOnly().eval())
    verdict = verify_mixed_precision_fx(
        graph,
        backend="cpu",
        autocast_dtype=torch.bfloat16,
        input_dtypes={"x": torch.float32},
    )

    assert verdict.ok
    assert verdict.mode == "fx"
    assert verdict.backend == "cpu"
    assert verdict.autocast_dtype == "bfloat16"
    assert _entry(verdict, "Linear").predicted_dtype == "bfloat16"


def test_public_exports_mixed_precision_contract():
    import src
    import tensorguard

    assert src.verify_mixed_precision is verify_mixed_precision
    assert tensorguard.verify_mixed_precision_fx is verify_mixed_precision_fx
    assert isinstance(verify_mixed_precision(_LinearOnly()), MixedPrecisionVerdict)
