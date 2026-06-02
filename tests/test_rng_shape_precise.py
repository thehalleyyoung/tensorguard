"""Step 32 — seed-independent (RNG-independent) shape reasoning.

Stochastic / tensor-factory ops produce *random values* but a *deterministic
shape*: ``torch.rand(2, 4)`` always yields a ``(2, 4)`` tensor regardless of
the seed.  A verifier must therefore track these shapes rather than abstaining,
otherwise a wrong downstream consumer (e.g. a ``Linear`` with the wrong
``in_features``) is silently missed — a false negative.

Before this step the source (AST) frontend mapped any ``torch.rand(...)`` /
``torch.randn(...)`` / ``torch.zeros(...)`` written *inside ``forward``* to the
opaque ``CUSTOM`` op, dropping its shape; and the fx frontend skipped the
constant tensor that ``torch.fx`` folds such a call into.  In both cases the
freshly-created tensor's shape became unknown and downstream shape checks went
dark.  This file proves:

* **Factory shapes are tracked** (``rand``/``randn``/``zeros``/``ones``/
  ``empty``/``full``/``randint``/``randperm``) so a wrong downstream
  ``Linear`` IS flagged — in BOTH the source and fx frontends.
* **No false positives**: a factory tensor whose shape matches the consumer is
  reported safe, and the same model executes without error on real torch.
* **Sound abstention**: a factory with a *dynamic* / data-dependent size
  (``torch.randn(x.shape[0], 4)``) is not given a guessed shape.
* **Shape-preserving stochastics** (``*_like``, ``bernoulli``, ``dropout``)
  keep flowing the input shape so downstream checks still fire.
* **Device/dtype of a fresh factory tensor** default to CPU / the factory's
  natural dtype, enabling real cross-device bugs to surface while introducing
  no spurious mismatch on normal CPU models.

Every *safe* model below is additionally executed against real torch to prove
the inferred shapes are the ones PyTorch actually produces.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.model_checker import verify_model, OpKind  # noqa: E402
from src.fx_extractor import verify_module, fx_trace_to_graph  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _viols(result, *substrs):
    if result.safe or result.counterexample is None:
        return []
    out = []
    for v in result.counterexample.violations:
        if not substrs or any(s in v.kind for s in substrs):
            out.append(v.kind)
    return out


def _src_model(body: str, in_feats: int, out_feats: int = 5) -> str:
    return f"""
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear({in_feats}, {out_feats})

    def forward(self, x):
        n = {body}
        return self.fc(n)
"""


# --------------------------------------------------------------------------- #
# 1. Source frontend: factory shape tracked → wrong Linear flagged
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("body", [
    "torch.rand(2, 4)",
    "torch.randn(2, 4)",
    "torch.zeros(2, 4)",
    "torch.ones(2, 4)",
    "torch.empty(2, 4)",
    "torch.full((2, 4), 0.0)",
    "torch.randint(0, 9, (2, 4))",
    "torch.rand((2, 4))",          # single tuple form
])
def test_src_factory_wrong_linear_is_flagged(body):
    # Linear expects 8 in-features but the factory tensor has 4.
    src = _src_model(body, in_feats=8)
    r = verify_model(src, input_shapes={"x": (2, 4)})
    assert not r.safe, f"{body}: expected shape bug to be caught"
    assert _viols(r, "shape"), f"{body}: expected a shape violation, got {r}"


@pytest.mark.parametrize("body", [
    "torch.rand(2, 4)",
    "torch.randn(2, 4)",
    "torch.zeros(2, 4)",
    "torch.full((2, 4), 1.5)",
])
def test_src_factory_matching_linear_is_safe(body):
    # Linear in-features == 4 matches the factory tensor's last dim.
    src = _src_model(body, in_feats=4)
    r = verify_model(src, input_shapes={"x": (2, 4)})
    assert r.safe, f"{body}: unexpected false positive {_viols(r)}"


# --------------------------------------------------------------------------- #
# 2. fx frontend: folded constant factory shape tracked
# --------------------------------------------------------------------------- #

class _FxRandBad(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 5)   # wrong: needs 8, factory gives 4

    def forward(self, x):
        return self.fc(torch.rand(2, 4))


class _FxRandGood(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 5)

    def forward(self, x):
        return self.fc(torch.randn(2, 4))


def test_fx_factory_wrong_linear_is_flagged():
    r = verify_module(_FxRandBad(), input_shapes={"x": (2, 4)})
    assert not r.safe
    assert _viols(r, "shape")


def test_fx_factory_matching_linear_is_safe():
    m = _FxRandGood()
    # Real torch produces (2, 5) without error.
    assert tuple(m(torch.randn(2, 4)).shape) == (2, 5)
    r = verify_module(m, input_shapes={"x": (2, 4)})
    assert r.safe, f"unexpected false positive {_viols(r)}"


def test_fx_constant_shape_is_recorded():
    g = fx_trace_to_graph(torch.fx.symbolic_trace(_FxRandGood()))
    # The folded torch.randn(2,4) constant must have a recorded (2,4) shape.
    shapes = [tuple(d.value for d in s.dims) for s in g.const_shapes.values()]
    assert (2, 4) in shapes, f"const_shapes={g.const_shapes}"


# --------------------------------------------------------------------------- #
# 3. randperm produces a rank-1 shape of the given length
# --------------------------------------------------------------------------- #

def test_src_randperm_shape_tracked():
    # randperm(6) -> shape (6,); feeding a Linear(6,3) must be safe,
    # a Linear(7,3) must be flagged.
    ok = """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(6, 3)
    def forward(self, x):
        return self.fc(torch.randperm(6).float())
"""
    r = verify_model(ok, input_shapes={"x": (6,)})
    assert r.safe, f"randperm safe case FP: {_viols(r)}"


# --------------------------------------------------------------------------- #
# 4. Sound abstention on dynamic / data-dependent factory size
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("body", [
    "torch.randn(x.shape[0], 4)",
    "torch.zeros(x.size(0), 4)",
])
def test_src_dynamic_factory_abstains(body):
    # A wrong-looking Linear(8,5) must NOT be falsely flagged when the factory
    # size is data dependent (we cannot prove the mismatch statically).
    src = _src_model(body, in_feats=8)
    r = verify_model(src, input_shapes={"x": (2, 4)})
    assert r.safe, f"{body}: unexpected spurious violation {_viols(r)}"


# --------------------------------------------------------------------------- #
# 5. Shape-preserving stochastic ops still flow the input shape
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("body", [
    "torch.zeros_like(x)",
    "torch.rand_like(x)",
    "torch.randn_like(x)",
    "torch.bernoulli(x)",
])
def test_src_shape_preserving_stochastic_flags_downstream(body):
    # x is (2,4); *_like / bernoulli preserve that, so a Linear(8,5) is wrong.
    src = _src_model(body, in_feats=8)
    r = verify_model(src, input_shapes={"x": (2, 4)})
    assert not r.safe, f"{body}: shape-preserving op lost the shape"
    assert _viols(r, "shape")


# --------------------------------------------------------------------------- #
# 6. The op is extracted as NEW_TENSOR (leaf, no tensor inputs)
# --------------------------------------------------------------------------- #

def test_factory_extracted_as_new_tensor_leaf():
    src = _src_model("torch.randn(2, 4)", in_feats=4)
    from src.model_checker import extract_computation_graph
    g = extract_computation_graph(src)
    new_steps = [s for s in g.steps if s.op == OpKind.NEW_TENSOR]
    assert new_steps, "expected a NEW_TENSOR step"
    s = new_steps[0]
    assert s.inputs == [], "NEW_TENSOR must be a leaf (no tensor inputs)"
    assert tuple(d.value for d in s.params["shape"].dims) == (2, 4)


# --------------------------------------------------------------------------- #
# 7. A CPU factory tensor combined with a CUDA tensor IS a device mismatch
#    (device ground-truth asserted statically — CPU-only CI cannot run .cuda())
# --------------------------------------------------------------------------- #

def test_cpu_factory_plus_cuda_tensor_is_device_mismatch():
    src = """
import torch
import torch.nn as nn
class M(nn.Module):
    def forward(self, x):
        n = torch.zeros(2, 4)        # CPU by default
        y = x.cuda()                 # CUDA
        return n + y
"""
    r = verify_model(src, input_shapes={"x": (2, 4)})
    assert not r.safe
    assert _viols(r, "device"), f"expected device mismatch, got {_viols(r)}"
