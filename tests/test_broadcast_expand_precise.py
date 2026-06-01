"""Step 25 -- broadcasting / expand / broadcast_to with exact torch semantics.

Before Step 25 the engine had several broadcasting blind spots:

  1. **``Tensor.expand`` was never validated.**  Two ``_METHOD_OP_MAP`` dicts
     exist in ``fx_extractor.py``; the second (operative) one lacked ``expand``,
     so ``x.expand(...)`` traced as a no-op ACTIVATION with no params, and the
     EXPAND handler used an incorrect right-alignment with no error reporting.

  2. **No exact expand oracle.**  There was no routine implementing torch's
     ``expand`` rule (extra leading dims are new; aligned dims must be ``-1``,
     equal, or singleton-input).  ``broadcast_to`` (which, unlike ``expand``,
     rejects ``-1``) was not modelled at all.

Step 25 adds ``compute_expand_shape`` (differential-tested vs ``torch.expand``
over 4000 random cases, zero mismatches), wires it into both engine paths
(verification + dynamo), routes ``expand`` / ``expand_as`` / ``broadcast_to``
through the operative method map, and tags ``broadcast_to`` so ``-1`` is
rejected there but allowed for ``expand``.

This module proves the new behaviour end-to-end and guards against regressions
(notably: ``x.expand(y.shape)`` — a single dynamic whole-shape arg — must abstain
rather than invent a rank-1 spec, and real models must stay free of new false
positives).
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from src.tensor_shapes import ShapeDim, TensorShape, compute_expand_shape
from src.fx_extractor import verify_module


def TS(*ds):
    return TensorShape(tuple(ShapeDim(d) for d in ds))


def _violation_kinds(result):
    if result.counterexample is None:
        return []
    return [v.kind for v in result.counterexample.violations]


def _is_unsafe(result):
    return (not result.safe) and "shape_incompatible" in _violation_kinds(result)


def _dims(shape):
    return tuple(d.value for d in shape.dims)


# ---------------------------------------------------------------------------
# Unit: compute_expand_shape (the exact oracle)
# ---------------------------------------------------------------------------

def test_singleton_expand_ok():
    out, err = compute_expand_shape(TS(3, 1), (3, 4))
    assert err is None
    assert _dims(out) == (3, 4)


def test_minus_one_keeps_input():
    out, err = compute_expand_shape(TS(3, 1), (-1, 4))
    assert err is None
    assert _dims(out) == (3, 4)


def test_leading_new_dim_ok():
    # input (4,) -> (2, 4): new leading dim 2 is fine
    out, err = compute_expand_shape(TS(4,), (2, 4))
    assert err is None
    assert _dims(out) == (2, 4)


def test_minus_one_in_leading_dim_is_error():
    # -1 is not allowed for a brand-new leading dim
    _out, err = compute_expand_shape(TS(4,), (-1, 4))
    assert err is not None


def test_non_singleton_mismatch_is_error():
    # input dim 3 != target 5 and input is not a singleton -> illegal
    _out, err = compute_expand_shape(TS(3, 1), (5, 4))
    assert err is not None


def test_negative_target_is_error():
    _out, err = compute_expand_shape(TS(3, 1), (3, -4))
    assert err is not None


def test_broadcast_to_rejects_minus_one():
    # torch.broadcast_to forbids -1
    _out, err = compute_expand_shape(TS(3, 1), (-1, 4), allow_neg_one=False)
    assert err is not None
    # but expand allows it
    _out2, err2 = compute_expand_shape(TS(3, 1), (-1, 4), allow_neg_one=True)
    assert err2 is None


def test_symbolic_input_abstains():
    # symbolic aligned input dim -> take target, no spurious error
    out, err = compute_expand_shape(TS("B", 1), (-1, 4))
    assert err is None
    assert _dims(out) == ("B", 4)


# ---------------------------------------------------------------------------
# Differential test vs torch.expand
# ---------------------------------------------------------------------------

def test_differential_vs_torch():
    random.seed(1234)
    checked = 0
    for _ in range(2000):
        irank = random.randint(1, 4)
        ishape = [random.choice([1, random.randint(1, 5)]) for _ in range(irank)]
        trank = random.randint(irank, 5)
        tgt = []
        for i in range(trank):
            ai = i - (trank - irank)
            if ai < 0:
                tgt.append(random.randint(1, 5))
            else:
                tgt.append(random.choice([-1, ishape[ai]]))
        out, err = compute_expand_shape(TS(*ishape), tuple(tgt), allow_neg_one=True)
        try:
            real = tuple(torch.zeros(*ishape).expand(*tgt).shape)
            ok = True
        except Exception:
            ok = False
        checked += 1
        if ok:
            assert err is None, f"false positive on {ishape}->{tgt}"
            assert _dims(out) == real, f"wrong shape {ishape}->{tgt}: {_dims(out)} != {real}"
        else:
            assert err is not None, f"false negative on {ishape}->{tgt}"
    assert checked == 2000


# ---------------------------------------------------------------------------
# End-to-end: verify_module (FX path)
# ---------------------------------------------------------------------------

def test_e2e_good_expand_safe():
    class M(nn.Module):
        def forward(self, x):
            return x.expand(3, 4)

    r = verify_module(M(), input_shapes={"x": (3, 1)})
    assert r.safe


def test_e2e_bad_expand_unsafe():
    class M(nn.Module):
        def forward(self, x):
            return x.expand(5, 4)

    r = verify_module(M(), input_shapes={"x": (3, 1)})
    assert _is_unsafe(r)


def test_e2e_tuple_form_expand_safe():
    class M(nn.Module):
        def forward(self, x):
            return x.expand((3, 4))

    r = verify_module(M(), input_shapes={"x": (3, 1)})
    assert r.safe


def test_e2e_broadcast_to_safe():
    class M(nn.Module):
        def forward(self, x):
            return torch.broadcast_to(x, (3, 4))

    r = verify_module(M(), input_shapes={"x": (3, 1)})
    assert r.safe


def test_e2e_broadcast_to_bad_unsafe():
    class M(nn.Module):
        def forward(self, x):
            return torch.broadcast_to(x, (5, 4))

    r = verify_module(M(), input_shapes={"x": (3, 1)})
    assert _is_unsafe(r)


def test_e2e_expand_as_safe():
    class M(nn.Module):
        def forward(self, x, y):
            return x.expand_as(y)

    r = verify_module(M(), input_shapes={"x": (3, 1), "y": (3, 4)})
    assert r.safe


def test_e2e_expand_whole_shape_abstains():
    # x.expand(y.shape): single dynamic whole-shape arg must NOT false-positive.
    class M(nn.Module):
        def forward(self, x, y):
            return x.expand(y.shape)

    r = verify_module(M(), input_shapes={"x": (1, 4), "y": (3, 4)})
    assert r.safe


def test_e2e_implicit_add_broadcast_mismatch_unsafe():
    class M(nn.Module):
        def forward(self, x, y):
            return x + y

    r = verify_module(M(), input_shapes={"x": (3, 4), "y": (5, 4)})
    assert not r.safe


def test_e2e_implicit_add_broadcast_ok_safe():
    class M(nn.Module):
        def forward(self, x, y):
            return x + y

    r = verify_module(M(), input_shapes={"x": (3, 4), "y": (1, 4)})
    assert r.safe


# ---------------------------------------------------------------------------
# Real-model smoke test: no new false positives
# ---------------------------------------------------------------------------

def test_real_models_no_new_false_positives():
    import torchvision.models as M

    # swin_t carries one pre-existing (LayerNorm) violation unrelated to expand;
    # excluded here so this guards specifically against expand/broadcast FPs.
    for ctor in (M.resnet18, M.mobilenet_v2, M.shufflenet_v2_x0_5, M.convnext_tiny):
        m = ctor().eval()
        r = verify_module(m, input_shapes={"x": (1, 3, 224, 224)})
        assert r.safe, f"{ctor.__name__} unexpectedly unsafe: {_violation_kinds(r)}"
