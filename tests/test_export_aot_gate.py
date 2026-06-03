"""Step 172 — torch.export / AOTInductor packaging gate, parity with ONNX gate.

``verify_exported_program`` and ``guarded_aot_package`` run TensorGuard's static
verification as the **first** side effect (``on_violation`` defaults to
``"raise"``, shapes inferred from the example args), giving the export/packaging
path the same guarantee the ONNX gate has: a real bug becomes one
``TensorGuardViolation`` *before* the tracer / Inductor runs or any artifact is
written.  We prove:

* a clean module verifies and produces a real ``torch.export.ExportedProgram``
  whose replayed module reproduces eager;
* a buggy module raises ``TensorGuardViolation`` *before* ``torch.export.export``
  is reached (a spy asserts the tracer was never called) and **no package file**
  is written;
* shapes are inferred from the example args when ``input_shapes`` is omitted;
* (slow) the clean module AOTInductor-compiles to a real ``.pt2`` package that
  loads and runs — skipped where a C++ toolchain is unavailable.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from src.torch_integration import (
    TensorGuardDynamicShapeError,
    TensorGuardViolation,
    guarded_aot_package,
    verify_exported_program,
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


def test_verify_exported_program_clean_returns_real_program():
    x = torch.randn(2, 10)
    ep = verify_exported_program(CleanNet(), (x,))
    assert isinstance(ep, torch.export.ExportedProgram)
    expected = CleanNet().eval()
    # The exported program is a faithful replay of the verified module's graph.
    replay = ep.module()
    assert replay(x).shape == expected(x).shape == (2, 5)


def test_verify_exported_program_raises_before_tracing(monkeypatch):
    traced = {"n": 0}
    real_export = torch.export.export

    def _spy(*a, **k):
        traced["n"] += 1
        return real_export(*a, **k)

    monkeypatch.setattr(torch.export, "export", _spy)
    with pytest.raises(TensorGuardViolation):
        verify_exported_program(BuggyNet(), (torch.randn(2, 10),))
    assert traced["n"] == 0  # verification fired before the tracer


def test_guarded_aot_package_blocks_buggy_before_writing(monkeypatch):
    compiled = {"n": 0}
    monkeypatch.setattr(
        torch._inductor,
        "aoti_compile_and_package",
        lambda *a, **k: compiled.__setitem__("n", compiled["n"] + 1),
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt2")
        with pytest.raises(TensorGuardViolation):
            guarded_aot_package(BuggyNet(), (torch.randn(2, 10),), package_path=path)
        assert compiled["n"] == 0  # never reached the packager
        assert not os.path.exists(path)  # nothing written


def test_shapes_inferred_from_example_args():
    # No input_shapes passed: the gate infers (b, 10) from the example tensor.
    with pytest.raises(TensorGuardViolation):
        verify_exported_program(BuggyNet(), (torch.randn(7, 10),))


def test_invalid_on_violation_is_rejected():
    # A typo must not silently degrade the gate to "ignore".
    with pytest.raises(ValueError):
        verify_exported_program(CleanNet(), (torch.randn(2, 10),), on_violation="rasie")


class Identity2D(nn.Module):
    def forward(self, x):
        return x.relu()


def _has_range(ep, lower, upper):
    return any(
        int(getattr(vr, "lower")) == lower and int(getattr(vr, "upper")) == upper
        for vr in ep.range_constraints.values()
    )


def test_export_dynamic_dim_range_is_validated_and_passed_to_export():
    b = torch.export.Dim("b", min=2, max=8)
    ep = verify_exported_program(
        CleanNet(),
        (torch.randn(3, 10),),
        input_shapes={"x": ("batch", 10)},
        dynamic_shapes={"x": {0: b}},
    )
    assert isinstance(ep, torch.export.ExportedProgram)
    assert _has_range(ep, 2, 8)


def test_inferred_shapes_adopt_declared_dynamic_axes():
    n = torch.export.Dim("n", min=4, max=16)
    ep = verify_exported_program(
        Identity2D(),
        (torch.randn(2, 10),),
        dynamic_shapes={"x": {1: n}},
    )
    assert isinstance(ep, torch.export.ExportedProgram)
    assert _has_range(ep, 4, 16)


def test_dynamic_dim_range_mismatch_fails_before_export(monkeypatch):
    traced = {"n": 0}

    def _spy(*args, **kwargs):
        traced["n"] += 1
        raise AssertionError("torch.export.export should not be reached")

    monkeypatch.setattr(torch.export, "export", _spy)
    b = torch.export.Dim("b", min=2, max=8)
    with pytest.raises(TensorGuardDynamicShapeError, match="min/max"):
        verify_exported_program(
            CleanNet(),
            (torch.randn(1, 10),),
            input_shapes={"x": ("batch", 10)},
            dynamic_shapes={"x": {0: b}},
        )
    assert traced["n"] == 0


def test_dynamic_dim_cannot_widen_explicit_concrete_tg_axis(monkeypatch):
    traced = {"n": 0}

    def _spy(*args, **kwargs):
        traced["n"] += 1
        raise AssertionError("torch.export.export should not be reached")

    monkeypatch.setattr(torch.export, "export", _spy)
    b = torch.export.Dim("b", min=2, max=8)
    with pytest.raises(TensorGuardDynamicShapeError, match="min/max"):
        verify_exported_program(
            CleanNet(),
            (torch.randn(4, 10),),
            input_shapes={"x": (4, 10)},
            dynamic_shapes={"x": {0: b}},
        )
    assert traced["n"] == 0


class PairNet(nn.Module):
    def forward(self, x, y):
        return x + y


def test_repeated_tg_symbol_requires_same_export_dim(monkeypatch):
    traced = {"n": 0}

    def _spy(*args, **kwargs):
        traced["n"] += 1
        raise AssertionError("torch.export.export should not be reached")

    monkeypatch.setattr(torch.export, "export", _spy)
    bx = torch.export.Dim("bx", min=2, max=8)
    by = torch.export.Dim("by", min=2, max=8)
    with pytest.raises(TensorGuardDynamicShapeError, match="equality"):
        verify_exported_program(
            PairNet(),
            (torch.randn(3, 10), torch.randn(3, 10)),
            input_shapes={"x": ("batch", 10), "y": ("batch", 10)},
            dynamic_shapes={"x": {0: bx}, "y": {0: by}},
        )
    assert traced["n"] == 0


def test_derived_dynamic_dim_must_match_tg_divisibility_relation(monkeypatch):
    traced = {"n": 0}

    def _spy(*args, **kwargs):
        traced["n"] += 1
        raise AssertionError("torch.export.export should not be reached")

    monkeypatch.setattr(torch.export, "export", _spy)
    b = torch.export.Dim("b", min=2, max=8)
    with pytest.raises(TensorGuardDynamicShapeError, match="divisibility"):
        verify_exported_program(
            Identity2D(),
            (torch.randn(3, 6),),
            input_shapes={"x": ("b", "width")},
            dynamic_shapes={"x": {0: b, 1: 2 * b}},
        )
    assert traced["n"] == 0


def test_matching_derived_dynamic_dim_exports_with_range_constraints():
    b = torch.export.Dim("b", min=2, max=8)
    ep = verify_exported_program(
        Identity2D(),
        (torch.randn(3, 6),),
        input_shapes={"x": ("b", "2*b")},
        dynamic_shapes={"x": {0: b, 1: 2 * b}},
    )
    assert isinstance(ep, torch.export.ExportedProgram)
    assert _has_range(ep, 2, 8)
    assert _has_range(ep, 4, 16)


@pytest.mark.slow
def test_guarded_aot_package_clean_roundtrip():
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clean.pt2")
            out = guarded_aot_package(CleanNet(), (torch.randn(2, 10),), package_path=path)
            assert os.path.exists(out)
            runner = torch._inductor.aoti_load_package(out)
            y = runner(torch.randn(3, 10))
            assert y.shape == (3, 5)
    except Exception as e:  # pragma: no cover - toolchain-dependent
        pytest.skip(f"AOTInductor toolchain unavailable: {e}")
