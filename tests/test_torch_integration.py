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


class SymbolicConvGood(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(16, 5)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.head(x)


class SymbolicConvBad(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(32, 5)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.head(x)


class LinearFirstAmbiguous(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)


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


def test_verify_module_infers_symbolic_conv_shape_without_input_shapes():
    res = verify_module(SymbolicConvGood())
    assert res is not None
    assert not str(res.verdict).upper().endswith("UNSAFE")
    assert res.inferred_input_shapes.get("x") == ("batch", 3, "height", "width")
    assert res.inferred_input_sources.get("x") == "layer:conv2d"


def test_verify_module_uses_symbolic_shape_to_catch_downstream_bug():
    res = verify_module(SymbolicConvBad())
    assert res is not None
    assert str(res.verdict).upper().endswith("UNSAFE")
    assert res.inferred_input_shapes.get("x") == ("batch", 3, "height", "width")
    assert any("expects" in (b.message or "").lower() for b in res.bugs)


def test_verify_module_abstains_on_rank_polymorphic_linear_first():
    res = verify_module(LinearFirstAmbiguous())
    assert res is not None
    assert res.inferred_input_shapes == {}
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


# Step 82 — coverage of the export pre-pass and the lenient bug-extraction
# branches that the py3.14 torch.compile fallback leaves untested.

from src.torch_integration import _real_bugs, verify_exported_program


def test_verify_exported_program_good_model_exports():
    ep = verify_exported_program(GoodNet(), (torch.randn(3, 10),))
    # torch.export returns an ExportedProgram we can run.
    out = ep.module()(torch.randn(3, 10))
    assert out.shape == (3, 5)


def test_verify_exported_program_raises_on_bug():
    with pytest.raises(TensorGuardViolation):
        verify_exported_program(
            BadNet(),
            (torch.randn(3, 10),),
            input_shapes={"x": ("batch", 10)},
            on_violation="raise",
        )


def test_real_bugs_lenient_verdict_tokens():
    class R:
        verdict = "BUG"
        bugs = [object(), object()]

    assert len(_real_bugs(R())) == 2

    class Fail:
        verdict = "FAIL"
        bugs = [object()]

    assert len(_real_bugs(Fail())) == 1


def test_real_bugs_none_and_safe():
    assert _real_bugs(None) == []

    class Safe:
        verdict = "SAFE"
        bugs = [object()]

    assert _real_bugs(Safe()) == []


def test_backend_delegates_to_inner_for_good_model():
    sentinel = object()

    def inner(gm, example_inputs):
        return sentinel

    backend = make_tensorguard_backend(
        GoodNet(), input_shapes={"x": ("batch", 10)}, inner=inner
    )

    class _GM:
        def forward(self, x):
            return x

    assert backend(_GM(), [torch.randn(2, 10)]) is sentinel


def test_module_source_abstains_for_typed_class():
    Dynamic = type("Dyn", (nn.Module,), {"forward": lambda self, x: x})
    assert module_source(Dynamic()) is None
