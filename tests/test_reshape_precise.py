"""Step 24 -- reshape / view / flatten with Z3-backed element-count reasoning.

Before Step 24 the engine had three reshape blind spots:

  1. **FX extraction dropped the shape spec.**  The fx extractor emitted
     ``params["target_shape"]`` (int-only), while every model_checker reshape
     handler reads ``params["dims"]`` — so on the torch.fx path *reshape was
     never checked at all* (``x.reshape(5, 5)`` on a 24-element tensor was
     silently reported safe).  ``torch.reshape`` (the function form) wasn't even
     mapped to ``OpKind.RESHAPE``.

  2. **No symbolic product reasoning.**  Incompatibilities that only manifest
     symbolically (``(B, 5) -> (B, 3)``) were missed, while a *latent false
     positive* flagged divisible-but-symbolic reshapes (``(B, 10) -> (-1, 3)``,
     which is fine for ``B = 3``).

  3. **``flatten`` ignored ``end_dim``.**  ``x.flatten(1, 2)`` always collapsed
     to the end, producing the wrong rank/shape.

Step 24 adds ``src/smt/reshape_theory.py`` (sound Z3 element-count oracle:
flags a reshape only when ``prod(input) == prod(output)`` is UNSAT for all
dimension assignments ``>= 1``), wires it into ``_apply_reshape`` ahead of the
legacy syntactic abstain, captures the full dim spec (with placeholders for
dynamic args) in the fx extractor for both method and function forms, and makes
``_propagate_flatten`` honour ``end_dim``.

This module proves the new behaviour end-to-end and guards against regressions
(notably: no false positives on real models such as ShuffleNet's channel
shuffle, whose colliding ``_dynN`` placeholders must not be coupled).
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from src.tensor_shapes import ShapeDim, TensorShape
from src.smt.reshape_theory import check_reshape_compatible, HAS_Z3
from src.model_checker import _propagate_flatten
from src.fx_extractor import verify_module


def TS(*ds):
    return TensorShape(tuple(ShapeDim(d) for d in ds))


def _violation_kinds(result):
    if result.counterexample is None:
        return []
    return [v.kind for v in result.counterexample.violations]


def _is_unsafe(result):
    return (not result.safe) and "shape_incompatible" in _violation_kinds(result)


# ---------------------------------------------------------------------------
# Unit: the Z3 element-count oracle
# ---------------------------------------------------------------------------

def test_concrete_incompatible_flagged():
    assert check_reshape_compatible(TS(2, 3, 4), (5, 5)) is not None
    assert check_reshape_compatible(TS(2, 3, 4), (7, 7)) is not None


def test_concrete_compatible_not_flagged():
    assert check_reshape_compatible(TS(2, 3, 4), (2, 12)) is None
    assert check_reshape_compatible(TS(2, 3, 4), (6, 4)) is None
    assert check_reshape_compatible(TS(2, 3, 4), (24,)) is None


def test_minus_one_inference():
    # 24 / 4 = 6 -> ok
    assert check_reshape_compatible(TS(2, 3, 4), (-1, 4)) is None
    # 24 not divisible by 5 -> impossible
    assert check_reshape_compatible(TS(2, 3, 4), (-1, 5)) is not None


def test_symbolic_unsat_flagged():
    # B*5 == B*3 has no solution for B >= 1.
    assert check_reshape_compatible(TS("B", 5), ("B", 3)) is not None
    # B*12 == B*15 likewise.
    assert check_reshape_compatible(TS("B", 12), ("B", 3, 5)) is not None


def test_symbolic_divisible_not_flagged():
    # (B, 10) -> (-1, 3): B = 3 works, so the reshape is satisfiable.
    assert check_reshape_compatible(TS("B", 10), (-1, 3)) is None
    # (B, 12) -> (B, -1): infer = 12.
    assert check_reshape_compatible(TS("B", 12), ("B", -1)) is None
    # Pure symbolic identity.
    assert check_reshape_compatible(TS("B", "C"), ("B", "C")) is None


def test_double_minus_one_is_invalid():
    msg = check_reshape_compatible(TS("B", 6), (-1, -1))
    assert msg is not None and "infer" in msg.lower()


def test_sentinel_copy_resolution():
    # Sentinel 0 copies input dim 0 (=2): (2, 12) -> ok.
    assert check_reshape_compatible(TS(2, 3, 4), (0, 12)) is None
    # (2, 11) = 22 != 24 -> incompatible.
    assert check_reshape_compatible(TS(2, 3, 4), (0, 11)) is not None


def test_placeholder_dims_not_coupled():
    # ShuffleNet channel-shuffle: input and target both carry *independently*
    # numbered ``_dynN`` placeholders that happen to collide.  They must NOT be
    # coupled (that would yield a spurious UNSAT / false positive).
    inp = TS("_dyn0", 2, "_dyn1", "_dyn2", "_dyn3")
    assert check_reshape_compatible(inp, ("_dyn0", "_dyn1", "_dyn2", "_dyn3")) is None


def test_dynamic_arg_no_false_positive():
    # x.reshape(x.shape[0], -1) -> ("_dyn0", -1): always satisfiable.
    assert check_reshape_compatible(TS("B", 4, 5), ("_dyn0", -1)) is None


def test_oracle_sound_on_random_concrete():
    """Differential test vs. torch: never flag a reshape torch accepts."""
    rng = random.Random(20240611)
    checked = 0
    for _ in range(400):
        ndim = rng.randint(1, 4)
        dims = [rng.randint(1, 6) for _ in range(ndim)]
        total = 1
        for d in dims:
            total *= d
        # Build a valid target shape by factoring ``total``.
        target = []
        remaining = total
        parts = rng.randint(1, 3)
        for _ in range(parts - 1):
            divisors = [d for d in range(1, remaining + 1) if remaining % d == 0]
            f = rng.choice(divisors)
            target.append(f)
            remaining //= f
        target.append(remaining)
        # Confirm torch agrees this is a valid reshape.
        t = torch.empty(dims)
        try:
            t.reshape(target)
        except Exception:
            continue
        checked += 1
        assert check_reshape_compatible(TS(*dims), tuple(target)) is None, (
            f"false positive on valid reshape {dims} -> {target}"
        )
    assert checked > 100


def test_oracle_flags_random_incompatible():
    """When the element counts genuinely differ, the oracle must flag."""
    rng = random.Random(99)
    flagged = 0
    for _ in range(200):
        dims = [rng.randint(2, 6) for _ in range(rng.randint(1, 3))]
        total = 1
        for d in dims:
            total *= d
        # Pick a target whose product is total + 1 (coprime-ish, guaranteed !=).
        bad = total + 1
        # Validate torch rejects it.
        t = torch.empty(dims)
        try:
            t.reshape((bad,))
            continue  # accidentally valid; skip
        except Exception:
            pass
        msg = check_reshape_compatible(TS(*dims), (bad,))
        assert msg is not None
        flagged += 1
    assert flagged > 100


# ---------------------------------------------------------------------------
# End-to-end: verify_module on the fx path (method + function forms)
# ---------------------------------------------------------------------------

def test_fx_method_reshape_incompatible_flagged():
    class M(nn.Module):
        def forward(self, x):
            return x.reshape(5, 5)

    assert _is_unsafe(verify_module(M(), input_shapes={"x": (2, 3, 4)}))


def test_fx_function_reshape_incompatible_flagged():
    class M(nn.Module):
        def forward(self, x):
            return torch.reshape(x, (5, 5))

    assert _is_unsafe(verify_module(M(), input_shapes={"x": (2, 3, 4)}))


def test_fx_view_incompatible_flagged():
    class M(nn.Module):
        def forward(self, x):
            return x.view(7, 7)

    assert _is_unsafe(verify_module(M(), input_shapes={"x": (2, 3, 4)}))


def test_fx_tuple_form_reshape_valid_safe():
    class M(nn.Module):
        def forward(self, x):
            return x.reshape((2, 12))

    assert verify_module(M(), input_shapes={"x": (2, 3, 4)}).safe


def test_fx_valid_view_safe():
    class M(nn.Module):
        def forward(self, x):
            return x.view(6, 4)

    assert verify_module(M(), input_shapes={"x": (2, 3, 4)}).safe


def test_fx_dynamic_reshape_no_false_positive():
    class M(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)

    assert verify_module(M(), input_shapes={"x": (2, 3, 4)}).safe


def test_fx_function_dynamic_reshape_no_false_positive():
    class M(nn.Module):
        def forward(self, x):
            return torch.reshape(x, (x.shape[0], -1))

    assert verify_module(M(), input_shapes={"x": (2, 3, 4)}).safe


# ---------------------------------------------------------------------------
# flatten honours end_dim
# ---------------------------------------------------------------------------

def _dims(shape):
    return tuple(shape.dims[i].value for i in range(shape.ndim))


def test_flatten_end_dim_partial_span():
    inp = TS("B", 3, 4, 5)
    out, _ = _propagate_flatten(inp, 1, 2)
    # dims 1..2 (=3*4) collapse, dim 3 (=5) preserved.
    assert _dims(out) == ("B", 12, 5)


def test_flatten_default_end_to_end():
    inp = TS("B", 3, 4, 5)
    out, _ = _propagate_flatten(inp, 1)
    assert _dims(out) == ("B", 60)


def test_flatten_negative_end_dim():
    inp = TS("B", 3, 4, 5)
    out, _ = _propagate_flatten(inp, 1, -2)  # -2 -> index 2
    assert _dims(out) == ("B", 12, 5)


def test_flatten_symbolic_span():
    inp = TS("B", "C", 4, 5)
    out, _ = _propagate_flatten(inp, 1, 2)
    # symbolic dim in span -> symbolic flattened dim, suffix preserved.
    assert out.ndim == 3
    assert out.dims[0].value == "B"
    assert out.dims[0 + 2].value == 5
    assert out.dims[1].is_symbolic


def test_flatten_matches_torch_concrete():
    rng = random.Random(7)
    for _ in range(50):
        dims = [rng.randint(1, 5) for _ in range(rng.randint(2, 4))]
        nd = len(dims)
        start = rng.randint(0, nd - 1)
        end = rng.randint(start, nd - 1)
        expected = tuple(torch.empty(dims).flatten(start, end).shape)
        out, _ = _propagate_flatten(TS(*dims), start, end)
        assert _dims(out) == expected, (dims, start, end)
