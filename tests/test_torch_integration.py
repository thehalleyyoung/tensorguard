"""Step 74 — torch.compile/export integration, proven on real nn.Modules."""

import warnings

import pytest
import torch
import torch.nn as nn

from src.torch_integration import (
    TensorGuardViolation,
    guarded_compile,
    make_tensorguard_backend,
    module_source,
    verify_module,
)


class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)  # expects 30, gets 20 → shape bug

    def forward(self, x):
        return self.fc2(self.fc1(x))


class GoodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_module_source_recovered():
    src = module_source(GoodNet())
    assert src is not None
    assert "class GoodNet" in src
    assert "import torch.nn as nn" in src


def test_verify_module_flags_real_bug():
    res = verify_module(BadNet(), input_shapes={"x": ("batch", 10)})
    assert res is not None
    assert str(res.verdict).upper().endswith("UNSAFE")
    assert any("expects" in (b.message or "").lower() for b in res.bugs)


def test_verify_module_good_is_not_unsafe():
    res = verify_module(GoodNet(), input_shapes={"x": ("batch", 10)})
    assert res is not None
    assert not str(res.verdict).upper().endswith("UNSAFE")


def test_guarded_compile_raises_on_bug():
    with pytest.raises(TensorGuardViolation) as ei:
        guarded_compile(BadNet(), input_shapes={"x": ("batch", 10)})
    assert ei.value.bugs
    assert "verification issue" in str(ei.value)


def test_guarded_compile_warn_mode():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = guarded_compile(
            BadNet(), input_shapes={"x": ("batch", 10)}, on_violation="warn"
        )
    # the verification violation must have warned
    assert any("verification issue" in str(w.message) for w in caught)
    assert out is not None


def test_guarded_compile_good_model_runs_and_matches_eager():
    model = GoodNet()
    x = torch.randn(4, 10)
    expected = model(x)
    compiled = guarded_compile(model, input_shapes={"x": ("batch", 10)})
    got = compiled(x)
    assert torch.allclose(got, expected)


def test_guarded_compile_invalid_on_violation():
    with pytest.raises(ValueError):
        guarded_compile(GoodNet(), on_violation="explode")


def test_backend_raises_inside_pipeline_for_bug():
    backend = make_tensorguard_backend(
        BadNet(), input_shapes={"x": ("batch", 10)}
    )

    class _FakeGM:
        def forward(self, *a, **k):
            return None

    with pytest.raises(TensorGuardViolation):
        backend(_FakeGM(), [torch.randn(2, 10)])


def test_backend_delegates_for_good_model():
    sentinel = object()
    backend = make_tensorguard_backend(
        GoodNet(),
        input_shapes={"x": ("batch", 10)},
        inner=lambda gm, ex: sentinel,
    )

    class _FakeGM:
        forward = None

    assert backend(_FakeGM(), [torch.randn(2, 10)]) is sentinel


def test_backend_checks_only_once():
    calls = {"n": 0}

    class _CountingGM:
        def forward(self, *a, **k):
            return None

    def inner(gm, ex):
        calls["n"] += 1
        return gm.forward

    backend = make_tensorguard_backend(
        GoodNet(), input_shapes={"x": ("batch", 10)}, inner=inner
    )
    backend(_CountingGM(), [])
    backend(_CountingGM(), [])
    assert calls["n"] == 2  # inner runs each call; verification only gates once


def test_verify_module_abstains_without_source():
    # a dynamically built module class has no recoverable source file
    Dynamic = type(
        "Dynamic",
        (nn.Module,),
        {"forward": lambda self, x: x},
    )
    assert verify_module(Dynamic()) is None
