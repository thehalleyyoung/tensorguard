"""Step 165 — TensorGuard as a torch Dynamo backend, proven on real code.

``make_tensorguard_backend`` returns a ``torch.compile`` backend: Dynamo hands it
a captured ``GraphModule`` and example inputs, the backend verifies the module
and either raises on a real bug or delegates to an inner compiler. We prove this
two ways:

* against **real FX graphs** — ``torch.fx.symbolic_trace`` produces exactly the
  kind of ``GraphModule`` Dynamo passes; the backend runs the real verifier over
  the real module source and (a) delegates a clean model to an inner compiler
  whose output reproduces eager, (b) raises ``TensorGuardViolation`` on a buggy
  one. This is interpreter-independent;
* against the **genuine ``torch.compile`` pipeline** when the running interpreter
  supports Dynamo (skipped otherwise, e.g. Python 3.14+), asserting a clean model
  compiles and runs and a buggy one surfaces ``TensorGuardViolation`` in the
  raised exception's chain (Dynamo wraps backend errors).

The pre-pass ``guarded_compile`` is also proven to block a buggy model *before*
compilation regardless of Dynamo availability.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.torch_integration import (
    TensorGuardViolation,
    guarded_compile,
    make_tensorguard_backend,
)

_SHAPES = {"x": ("b", 10)}


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


def _compile_supported() -> bool:
    if not hasattr(torch, "compile"):
        return False
    try:
        m = nn.Linear(4, 4)
        calls = {"n": 0}

        def be(gm, ex):
            calls["n"] += 1
            return gm.forward

        torch.compile(m, backend=be)(torch.randn(2, 4))
        return calls["n"] > 0
    except Exception:
        return False


def _chain_has(exc, exc_type) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, exc_type):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def test_backend_delegates_clean_model_to_inner_compiler():
    model = CleanNet()
    gm = torch.fx.symbolic_trace(model)
    x = torch.randn(3, 10)
    calls = {"n": 0}

    def inner(g, example_inputs):
        calls["n"] += 1
        return g.forward

    backend = make_tensorguard_backend(model, input_shapes=_SHAPES, inner=inner)
    compiled_fwd = backend(gm, [x])           # what Dynamo would invoke
    assert calls["n"] == 1, "inner compiler was not called for a clean model"
    out = compiled_fwd(x)
    assert torch.allclose(out, model(x), atol=1e-5)


def test_backend_raises_on_buggy_model_real_graph():
    model = BuggyNet()
    gm = torch.fx.symbolic_trace(model)
    x = torch.randn(3, 10)
    backend = make_tensorguard_backend(
        model, input_shapes=_SHAPES, on_violation="raise"
    )
    with pytest.raises(TensorGuardViolation) as ei:
        backend(gm, [x])
    assert ei.value.bugs, "no structured bugs attached to the violation"


def test_guarded_compile_prepass_blocks_bug_without_dynamo():
    # The pre-pass verifies before ever calling torch.compile, so a bug is caught
    # even on interpreters where Dynamo is unavailable.
    with pytest.raises(TensorGuardViolation):
        guarded_compile(BuggyNet(), input_shapes=_SHAPES, on_violation="raise")


def test_guarded_compile_returns_runnable_clean_model():
    model = CleanNet()
    compiled = guarded_compile(model, input_shapes=_SHAPES, on_violation="raise")
    x = torch.randn(2, 10)
    assert torch.allclose(compiled(x), model(x), atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(not _compile_supported(), reason="torch.compile/Dynamo unsupported here")
def test_real_torch_compile_pipeline():
    torch._dynamo.reset()
    # Clean: the backend is invoked by Dynamo and the compiled model runs.
    model = CleanNet()
    calls = {"n": 0}

    def inner(g, ex):
        calls["n"] += 1
        return g.forward

    backend = make_tensorguard_backend(model, input_shapes=_SHAPES, inner=inner)
    compiled = torch.compile(model, backend=backend)
    x = torch.randn(4, 10)
    assert torch.allclose(compiled(x), model(x), atol=1e-5)
    assert calls["n"] >= 1, "Dynamo never invoked the TensorGuard backend"

    # Buggy: the violation surfaces through the compile pipeline.
    torch._dynamo.reset()
    bug = BuggyNet()
    bad_backend = make_tensorguard_backend(
        bug, input_shapes=_SHAPES, on_violation="raise"
    )
    compiled_bug = torch.compile(bug, backend=bad_backend)
    with pytest.raises(Exception) as ei:
        compiled_bug(torch.randn(4, 10))
    assert _chain_has(ei.value, TensorGuardViolation), ei.value
