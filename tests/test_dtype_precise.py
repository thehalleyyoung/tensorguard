"""Step 30 — dtype inference & promotion (a second algebra alongside shape).

TensorGuard tracks an *element dtype* for every tensor in addition to its
shape, device, phase and gradient status.  Several torch operations raise a
``RuntimeError`` purely because of a dtype mismatch — independently of shape:

* ``Linear`` / ``Conv*`` perform a matmul / convolution against a stored
  parameter and require the *input* dtype to **exactly** equal the parameter
  dtype, otherwise torch raises
  ``"mat1 and mat2 must have the same dtype, but got Half and Float"`` or
  ``"Input type (...) and bias type (...) should be the same"``.
* ``torch.matmul`` / ``mm`` / ``bmm`` require both operands to share a dtype.
* ``nn.Embedding`` (and ``index_select`` / ``gather`` / ``scatter``) require an
  **integer** index tensor; a floating index raises
  ``"Expected tensor for argument index to have scalar type Long"``.

Crucially, torch *type-promotes* element-wise ``add`` / ``cat`` (``f32 + f16``,
``cat([f32, f16])`` are fine), so those must **not** be flagged.

Soundness contract verified here: the dtype algebra only reasons about *known*
dtypes (explicit input annotations, a layer's real parameter dtype, or an
explicit ``.half()/.float()/.double()/.bfloat16()`` cast).  Any unknown dtype
makes the relevant check abstain, so the analysis never raises a false
positive.  Every emitted ``dtype_error`` corresponds to a real torch
``RuntimeError`` under the recorded dtypes — proven below by large differential
sweeps where the verifier's verdict must exactly equal whether torch raises.
"""

import itertools

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.fx_extractor import verify_module  # noqa: E402
from src.model_checker import verify_model  # noqa: E402


# Map a tensor cast-method name to the torch dtype it produces.
_CAST_METHODS = {
    "half": torch.float16,
    "float": torch.float32,
    "double": torch.float64,
    "bfloat16": torch.bfloat16,
}


def _dtype_viols(result):
    if result.safe or result.counterexample is None:
        return []
    return [v for v in result.counterexample.violations
            if v.kind in ("dtype_error", "dtype_mismatch")]


def _torch_dtype_error(fn):
    """Run *fn*; return True iff it raises a torch dtype-mismatch RuntimeError.

    Returns the string ``"other"`` when it raises for a non-dtype reason (e.g.
    an op unsupported on CPU for a given dtype) so the caller can skip that
    configuration."""
    try:
        fn()
        return False
    except RuntimeError as exc:
        msg = str(exc).lower()
        if ("same dtype" in msg or "should be the same" in msg
                or "scalar type" in msg or "must have the same" in msg):
            return True
        return "other"


# ──────────────────────────────────────────────────────────────────────────
# Differential sweep: Linear input-vs-parameter dtype
# ──────────────────────────────────────────────────────────────────────────

class _LinWrap(nn.Module):
    """``Linear`` whose parameter is cast to *param_dt* and whose input is cast
    via *input_method* inside forward (so the input dtype is statically known
    to the verifier)."""

    def __init__(self, param_dt, input_method):
        super().__init__()
        self.lin = nn.Linear(8, 4).to(param_dt)
        self._m = input_method

    def forward(self, x):
        return self.lin(getattr(x, self._m)())


def test_linear_dtype_differential_sweep():
    methods = list(_CAST_METHODS)
    checked = 0
    disagreements = []
    for param_m, input_m in itertools.product(methods, methods):
        param_dt = _CAST_METHODS[param_m]
        m = _LinWrap(param_dt, input_m).eval()

        def run():
            x = torch.randn(3, 8)
            return m(x)

        gt = _torch_dtype_error(run)
        if gt == "other":
            continue  # op unsupported on CPU for this dtype combo — skip

        res = verify_module(m, input_shapes={"x": (3, 8)})
        flagged = len(_dtype_viols(res)) > 0
        checked += 1
        if flagged != gt:
            disagreements.append((param_m, input_m, gt, flagged))

    assert checked >= 12, f"too few configs checked: {checked}"
    assert not disagreements, f"verifier disagreed with torch: {disagreements}"


# ──────────────────────────────────────────────────────────────────────────
# Differential sweep: Conv2d input-vs-parameter dtype
# ──────────────────────────────────────────────────────────────────────────

class _ConvWrap(nn.Module):
    def __init__(self, param_dt, input_method):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1).to(param_dt)
        self._m = input_method

    def forward(self, x):
        return self.conv(getattr(x, self._m)())


def test_conv2d_dtype_differential_sweep():
    # bfloat16 / float16 conv may be unsupported on CPU → handled via skip.
    methods = list(_CAST_METHODS)
    checked = 0
    disagreements = []
    for param_m, input_m in itertools.product(methods, methods):
        param_dt = _CAST_METHODS[param_m]
        m = _ConvWrap(param_dt, input_m).eval()

        def run():
            x = torch.randn(2, 3, 8, 8)
            return m(x)

        gt = _torch_dtype_error(run)
        if gt == "other":
            continue

        res = verify_module(m, input_shapes={"x": (2, 3, 8, 8)})
        flagged = len(_dtype_viols(res)) > 0
        checked += 1
        if flagged != gt:
            disagreements.append((param_m, input_m, gt, flagged))

    assert checked >= 6, f"too few conv configs checked: {checked}"
    assert not disagreements, f"verifier disagreed with torch: {disagreements}"


# ──────────────────────────────────────────────────────────────────────────
# Internal, input-independent mismatch
# ──────────────────────────────────────────────────────────────────────────

class _TwoLinearMixed(nn.Module):
    """First Linear cast to fp16, second left fp32: the fp16 activation feeds
    the fp32 layer → torch raises regardless of the input dtype."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 5)
        self.b = nn.Linear(5, 6)
        self.a.half()

    def forward(self, x):
        return self.b(self.a(x.half()))


def test_internal_mixed_precision_is_unsafe():
    m = _TwoLinearMixed().eval()
    res = verify_module(m, input_shapes={"x": (2, 4)})
    assert not res.safe
    assert _dtype_viols(res), "expected a dtype_error"
    # Ground-truth: torch really raises.
    with pytest.raises(RuntimeError):
        m(torch.randn(2, 4))


class _TwoLinearClean(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 5)
        self.b = nn.Linear(5, 6)

    def forward(self, x):
        return self.b(self.a(x))


def test_clean_model_is_safe_no_dtype_fp():
    m = _TwoLinearClean().eval()
    res = verify_module(m, input_shapes={"x": (2, 4)})
    assert res.safe
    m(torch.randn(2, 4))  # torch agrees


def test_whole_model_half_consistent_is_safe():
    """A model cast wholesale to fp16 and fed an fp16-annotated input is safe."""
    m = _TwoLinearClean().half().eval()
    res = verify_module(
        m, input_shapes={"x": (2, 4)}, input_dtypes={"x": "float16"}
    )
    assert res.safe
    m(torch.randn(2, 4).half())  # torch agrees


def test_whole_model_half_f32_input_is_unsafe():
    m = _TwoLinearClean().half().eval()
    res = verify_module(
        m, input_shapes={"x": (2, 4)}, input_dtypes={"x": "float32"}
    )
    assert not res.safe
    assert _dtype_viols(res)
    with pytest.raises(RuntimeError):
        m(torch.randn(2, 4))  # f32 input, fp16 weights


# ──────────────────────────────────────────────────────────────────────────
# Embedding / index dtype
# ──────────────────────────────────────────────────────────────────────────

class _EmbFloatIndex(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(10, 4)

    def forward(self, x):
        return self.e(x.float())


def test_embedding_float_index_is_unsafe():
    m = _EmbFloatIndex().eval()
    res = verify_module(m, input_shapes={"x": (2, 3)})
    assert not res.safe
    assert _dtype_viols(res)
    with pytest.raises((RuntimeError, IndexError)):
        m(torch.randint(0, 10, (2, 3)))  # .float() index → torch raises


class _EmbPlain(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(10, 4)

    def forward(self, x):
        return self.e(x)


def test_embedding_unannotated_index_no_fp():
    """Unannotated index dtype is unknown → the verifier abstains (no FP)."""
    m = _EmbPlain().eval()
    res = verify_module(m, input_shapes={"x": (2, 3)})
    assert res.safe
    m(torch.randint(0, 10, (2, 3)))  # torch agrees


# ──────────────────────────────────────────────────────────────────────────
# Promotion: add / cat must NOT be flagged (torch promotes)
# ──────────────────────────────────────────────────────────────────────────

def test_add_mixed_dtype_not_flagged():
    src = """
import torch
import torch.nn as nn
class M(nn.Module):
    def forward(self, x, y):
        return x + y
"""
    res = verify_model(
        src, input_shapes={"x": (2, 4), "y": (2, 4)},
        input_dtypes={"x": "float32", "y": "float16"},
    )
    assert not _dtype_viols(res)
    # torch really promotes:
    (torch.randn(2, 4) + torch.randn(2, 4).half())


# ──────────────────────────────────────────────────────────────────────────
# Abstention guarantees (soundness w/o false positives)
# ──────────────────────────────────────────────────────────────────────────

def test_unknown_input_dtype_abstains():
    """No input_dtypes → input dtype unknown → Linear check abstains (safe)."""
    src = """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 5)
    def forward(self, x):
        return self.fc(x)
"""
    res = verify_model(src, input_shapes={"x": (2, 4)})
    assert res.safe


def test_check_dtypes_flag_suppresses():
    m = _TwoLinearMixed().eval()
    on = verify_module(m, input_shapes={"x": (2, 4)})
    assert not on.safe and _dtype_viols(on)
    off = verify_module(m, input_shapes={"x": (2, 4)}, check_dtypes=False)
    assert off.safe or not _dtype_viols(off)


def test_symbolic_shape_dtype_still_works():
    """Dtype reasoning is orthogonal to shape: a symbolic batch dim does not
    suppress an otherwise-known dtype mismatch."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 5)
            self.b = nn.Linear(5, 6)
            self.a.half()

        def forward(self, x):
            return self.b(self.a(x.half()))

    res = verify_module(M().eval(), input_shapes={"x": ("batch", 4)})
    assert not res.safe
    assert _dtype_viols(res)
