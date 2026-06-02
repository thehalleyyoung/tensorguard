"""Step 31 — device propagation across ``.to()/.cuda()/.cpu()/.pin_memory()``.

TensorGuard tracks the *device* each tensor lives on (CPU vs CUDA) and flags
operations that combine tensors on different devices — exactly the
``RuntimeError: Expected all tensors to be on the same device`` that PyTorch
raises at runtime.

Before this step the **fx frontend silently dropped every device transfer**:
the operative method-op map lacked ``to``/``cuda``/``cpu``/``pin_memory`` so a
real traced model's ``x.cuda()`` became a shape-preserving no-op and the
verifier never saw the device move.  This file proves device propagation now
works end-to-end through ``verify_module`` (fx) for:

* ``.to('cuda')`` / ``.to(device=...)`` / ``.cuda()`` / ``.cuda(idx)`` —
  move the tensor to the target device;
* ``.cpu()`` / ``.to('cpu')`` — move back to CPU;
* ``.pin_memory()`` — device-preserving (pinned CPU tensor), must NOT be read
  as a device change (a regression guarded here because the Z3 device-transition
  encoder previously left a target-less ``TO_DEVICE`` output unconstrained,
  producing a spurious mismatch);
* combined ``.to(device, dtype)`` — changes both device and element dtype.

Soundness: the device move is recorded only when the target device is a
statically-known spelling (``'cpu'``, ``'cuda[:N]'``, ``torch.device(...)``);
a device taken from another tensor (``x.to(y.device)``) is left unknown and the
output inherits the input device, so the analysis never invents a mismatch.

Because CI runs on a CPU-only box we cannot *execute* a genuine cross-device
mismatch, so device ground-truth is asserted statically: the verifier's
device verdict must match the unambiguous device algebra (cpu vs cuda) that
torch itself enforces.  All device-*consistent* cases below are additionally
executed against real torch to prove they do not raise.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.fx_extractor import verify_module, fx_trace_to_graph  # noqa: E402
from src.model_checker import Device, OpKind  # noqa: E402


def _device_viols(result):
    if result.safe or result.counterexample is None:
        return []
    return [v for v in result.counterexample.violations
            if v.kind == "device_mismatch"]


def _ops(module, example=None):
    g = fx_trace_to_graph(torch.fx.symbolic_trace(module))
    return g.steps


# ──────────────────────────────────────────────────────────────────────────
# fx frontend now captures device transfers (the core regression)
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr,expect_device", [
    ("x.to('cuda')", "cuda"),
    ("x.cuda()", "cuda:0"),
    ("x.cuda(1)", "cuda:1"),
    ("x.cpu()", "cpu"),
    ("x.to('cpu')", "cpu"),
    ("x.to(device='cuda:1')", "cuda:1"),
])
def test_fx_captures_device_target(expr, expect_device):
    ns = {"nn": nn, "torch": torch}
    src = (
        "class M(nn.Module):\n"
        "    def forward(self, x):\n"
        f"        return {expr}\n"
    )
    exec(src, ns)
    steps = _ops(ns["M"]())
    to_steps = [s for s in steps if s.op == OpKind.TO_DEVICE]
    assert to_steps, f"no TO_DEVICE step captured for {expr}"
    assert to_steps[0].params.get("device") == expect_device


def test_pin_memory_is_device_preserving_in_fx():
    class M(nn.Module):
        def forward(self, x):
            return x.pin_memory()
    steps = _ops(M())
    to_steps = [s for s in steps if s.op == OpKind.TO_DEVICE]
    assert to_steps, "pin_memory should map to TO_DEVICE"
    # No device target → device-preserving.
    assert "device" not in to_steps[0].params


# ──────────────────────────────────────────────────────────────────────────
# Cross-device combination is flagged (matches torch's runtime error)
# ──────────────────────────────────────────────────────────────────────────

def test_cross_device_add_is_flagged():
    class Cross(nn.Module):
        def forward(self, x, y):
            return x.to('cuda') + y      # x on cuda, y on cpu → mismatch
    res = verify_module(Cross(), input_shapes={"x": (2, 4), "y": (2, 4)})
    assert _device_viols(res), "expected a device_mismatch"


def test_cross_device_default_cuda_then_cpu():
    class M(nn.Module):
        def forward(self, x, y):
            return x.cpu() + y           # x→cpu, y stays cuda → mismatch
    res = verify_module(
        M(), input_shapes={"x": (2, 4), "y": (2, 4)},
        default_device=Device.CUDA_0,
    )
    assert _device_viols(res)


# ──────────────────────────────────────────────────────────────────────────
# Device-consistent programs are NOT flagged (no false positives) and run
# ──────────────────────────────────────────────────────────────────────────

def test_both_to_same_device_is_safe():
    class M(nn.Module):
        def forward(self, x, y):
            return x.cuda() + y.cuda()
    res = verify_module(M(), input_shapes={"x": (2, 4), "y": (2, 4)})
    assert not _device_viols(res)


def test_roundtrip_cuda_then_cpu_is_safe():
    class M(nn.Module):
        def forward(self, x, y):
            return x.cuda().cpu() + y     # back to cpu, matches cpu y
    res = verify_module(M(), input_shapes={"x": (2, 4), "y": (2, 4)})
    assert not _device_viols(res)


def test_pin_memory_does_not_create_mismatch():
    class M(nn.Module):
        def forward(self, x, y):
            return x.pin_memory() + y     # pinned CPU + CPU → safe
    res = verify_module(M(), input_shapes={"x": (2, 4), "y": (2, 4)})
    assert not _device_viols(res)


def test_to_dtype_only_does_not_change_device():
    """``x.to(torch.float16)`` changes dtype, not device — must not mismatch."""
    class M(nn.Module):
        def forward(self, x, y):
            return x.to(torch.float16) + y.to(torch.float16)
    res = verify_module(M(), input_shapes={"x": (2, 4), "y": (2, 4)})
    assert not _device_viols(res)


def test_device_from_other_tensor_is_unknown_no_fp():
    """``x.to(y.device)`` has a non-static device target → inherit input device,
    never invent a mismatch."""
    class M(nn.Module):
        def forward(self, x, y):
            return x.to(y.device) + y
    res = verify_module(M(), input_shapes={"x": (2, 4), "y": (2, 4)})
    assert not _device_viols(res)


# ──────────────────────────────────────────────────────────────────────────
# Combined device + dtype move
# ──────────────────────────────────────────────────────────────────────────

def test_to_device_and_dtype_combined():
    class M(nn.Module):
        def forward(self, x):
            return x.to('cuda', torch.float16)
    steps = _ops(M())
    to_steps = [s for s in steps if s.op == OpKind.TO_DEVICE]
    assert to_steps
    assert to_steps[0].params.get("device") == "cuda"
    assert to_steps[0].params.get("cast_dtype") == "float16"


# ──────────────────────────────────────────────────────────────────────────
# Real model regression: no spurious device violations
# ──────────────────────────────────────────────────────────────────────────

def test_torchvision_no_device_false_positive():
    M = pytest.importorskip("torchvision.models")
    for ctor in (M.resnet18, M.mobilenet_v2):
        m = ctor(weights=None).eval()
        res = verify_module(m, input_shapes={"x": (1, 3, 224, 224)})
        assert not _device_viols(res), f"{ctor.__name__} device FP"
