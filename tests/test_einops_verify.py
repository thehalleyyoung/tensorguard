"""Differential test: ``src.einops_verify`` vs the real ``einops`` package.

For every (op, pattern, shape, axes) case below we run the *real* einops op on
a concrete tensor and compare its observable behaviour — raises vs. succeeds,
and the resulting shape on success — against the purely static verdict from
:func:`src.einops_verify.verify_einops`.  A random-pattern fuzzer
(``test_fuzz_matches_real_einops``) generates thousands of additional cases so
the static model cannot silently drift from real einops semantics.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

einops = pytest.importorskip("einops")
from einops import rearrange, reduce, repeat  # noqa: E402

from src.einops_verify import verify_einops  # noqa: E402

_OPS = {
    "rearrange": lambda x, p, **k: rearrange(x, p, **k),
    "reduce": lambda x, p, **k: reduce(x, p, reduction="sum", **k),
    "repeat": lambda x, p, **k: repeat(x, p, **k),
}


def _real(op, pattern, shape, axes):
    """Return ('ok', out_shape) or ('err', None) for real einops."""
    x = np.zeros(shape, dtype=np.float32)
    try:
        out = _OPS[op](x, pattern, **axes)
        return "ok", tuple(out.shape)
    except Exception:  # einops.EinopsError and friends
        return "err", None


def _check(op, pattern, shape, **axes):
    real_status, real_shape = _real(op, pattern, shape, axes)
    v = verify_einops(op, pattern, shape, **axes)
    static_status = "ok" if v.ok else "err"
    assert static_status == real_status, (
        f"{op} {pattern!r} shape={shape} axes={axes}: "
        f"real={real_status} static={static_status} ({v.error})"
    )
    if real_status == "ok":
        assert tuple(v.output_shape) == real_shape, (
            f"{op} {pattern!r} shape={shape} axes={axes}: "
            f"real_shape={real_shape} static_shape={v.output_shape}"
        )


# ── hand-written cases covering each failure mode ────────────────────────────

VALID_CASES = [
    ("rearrange", "a b -> b a", (2, 3), {}),
    ("rearrange", "(h w) c -> h w c", (12, 5), {"h": 4}),
    ("rearrange", "(h w) c -> h w c", (12, 5), {"w": 3}),
    ("rearrange", "b (h w) -> b h w", (2, 12), {"h": 3}),
    ("rearrange", "b c h w -> b (c h w)", (2, 3, 4, 5), {}),
    ("rearrange", "b c h w -> b h w c", (2, 3, 4, 5), {}),
    ("rearrange", "b ... c -> b c ...", (2, 3, 4, 5), {}),
    ("rearrange", "b h w c -> b (h w) c", (2, 4, 4, 8), {}),
    ("rearrange", "a 1 c -> a c", (3, 1, 5), {}),
    ("reduce", "a b c -> a c", (2, 3, 4), {}),
    ("reduce", "b c h w -> b c", (2, 3, 4, 5), {}),
    ("reduce", "(h h2) w -> h w", (12, 5), {"h2": 3}),
    ("reduce", "b ... -> b", (2, 3, 4), {}),
    ("repeat", "a b -> a b c", (2, 3), {"c": 4}),
    ("repeat", "a b -> a (b c)", (2, 3), {"c": 4}),
    ("repeat", "h w -> (h 2) w", (4, 5), {}),
]

INVALID_CASES = [
    ("rearrange", "(h w) c -> h w c", (10, 5), {"h": 3}),   # non-divisible
    ("rearrange", "(h w) c -> h w c", (12, 5), {}),         # underdetermined
    ("rearrange", "a b -> b a c", (2, 3), {}),              # new axis
    ("rearrange", "a b c -> a b", (2, 3, 4), {}),           # dropped axis
    ("rearrange", "a b -> a b", (2, 3, 4), {}),             # rank mismatch
    ("rearrange", "a a -> a", (2, 2), {}),                  # duplicate
    ("rearrange", "a 1 c -> a c", (3, 2, 5), {}),           # anon != 1
    ("reduce", "a b c -> a c d", (2, 3, 4), {}),            # new axis on rhs
    ("repeat", "a b c -> a b", (2, 3, 4), {}),              # repeat drops axis
    ("repeat", "a b -> a b c", (2, 3), {}),                 # missing length
    ("rearrange", "(h w) -> h w", (7, ), {"h": 2}),         # non-divisible 1D
]


@pytest.mark.parametrize("op,pattern,shape,axes", VALID_CASES)
def test_valid_cases_match(op, pattern, shape, axes):
    _check(op, pattern, shape, **axes)
    v = verify_einops(op, pattern, shape, **axes)
    assert v.ok


@pytest.mark.parametrize("op,pattern,shape,axes", INVALID_CASES)
def test_invalid_cases_match(op, pattern, shape, axes):
    _check(op, pattern, shape, **axes)
    v = verify_einops(op, pattern, shape, **axes)
    assert not v.ok
    assert v.error_kind is not None


def test_symbolic_dims_never_false_positive():
    # A symbolic axis length must not be refuted even when concrete would fail.
    v = verify_einops("rearrange", "(h w) c -> h w c", ("seq", 5), h=8)
    assert v.ok
    assert v.output_shape[0] == 8


def test_error_kinds_are_specific():
    assert verify_einops(
        "rearrange", "(h w) c -> h w c", (10, 5), h=3
    ).error_kind == "non_divisible"
    assert verify_einops(
        "rearrange", "a b -> b a c", (2, 3)
    ).error_kind == "axis_set_mismatch"
    assert verify_einops(
        "repeat", "a b -> a b c", (2, 3)
    ).error_kind == "missing_length"


# ── fuzzer: random patterns vs real einops ───────────────────────────────────


def _random_pattern(rng):
    """Generate a random (op, pattern, shape, axes) tuple."""
    names = ["a", "b", "c", "d"][: rng.randint(2, 4)]
    op = rng.choice(["rearrange", "reduce", "repeat"])

    # assign small sizes
    sizes = {n: rng.randint(2, 4) for n in names}
    lhs = list(names)

    # maybe fuse two adjacent lhs axes into a group
    grouped_pairs = []
    if len(lhs) >= 2 and rng.random() < 0.5:
        i = rng.randrange(len(lhs) - 1)
        grouped_pairs.append((lhs[i], lhs[i + 1]))

    def render_lhs():
        toks, i = [], 0
        while i < len(lhs):
            if i + 1 < len(lhs) and (lhs[i], lhs[i + 1]) in grouped_pairs:
                toks.append(f"({lhs[i]} {lhs[i + 1]})")
                i += 2
            else:
                toks.append(lhs[i])
                i += 1
        return toks

    lhs_toks = render_lhs()

    # build the concrete shape implied by lhs
    shape = []
    i = 0
    flat_lhs_order = []
    for tok in lhs_toks:
        if tok.startswith("("):
            inner = tok[1:-1].split()
            shape.append(sizes[inner[0]] * sizes[inner[1]])
            flat_lhs_order.extend(inner)
        else:
            shape.append(sizes[tok])
            flat_lhs_order.append(tok)

    # choose rhs axis set per op
    if op == "rearrange":
        rhs_names = list(flat_lhs_order)
        rng.shuffle(rhs_names)
    elif op == "reduce":
        keep = [n for n in flat_lhs_order if rng.random() < 0.7] or [flat_lhs_order[0]]
        rhs_names = keep
    else:  # repeat
        rhs_names = list(flat_lhs_order)
        if rng.random() < 0.6:
            rhs_names.append("r")
            sizes["r"] = rng.randint(2, 3)

    rhs_toks = list(rhs_names)
    pattern = " ".join(lhs_toks) + " -> " + " ".join(rhs_toks)

    # axes kwargs: supply lengths needed for group splits / repeat-new-axis
    axes = {}
    for tok in lhs_toks:
        if tok.startswith("("):
            inner = tok[1:-1].split()
            # supply one of the two so the split is determined
            chosen = rng.choice(inner)
            axes[chosen] = sizes[chosen]
    if op == "repeat" and "r" in rhs_names:
        axes["r"] = sizes["r"]

    # randomly corrupt to exercise the error paths
    if rng.random() < 0.35:
        kind = rng.choice(["dim", "drop_len", "new_axis"])
        if kind == "dim" and shape:
            shape[rng.randrange(len(shape))] += 1  # may break divisibility
        elif kind == "drop_len" and axes:
            axes.pop(rng.choice(list(axes)))
        elif kind == "new_axis":
            rhs_toks.append("z")
            pattern = " ".join(lhs_toks) + " -> " + " ".join(rhs_toks)

    return op, pattern, tuple(shape), axes


def test_fuzz_matches_real_einops():
    rng = random.Random(0xC0FFEE)
    checked = 0
    for _ in range(3000):
        op, pattern, shape, axes = _random_pattern(rng)
        if not shape:
            continue
        _check(op, pattern, shape, **axes)
        checked += 1
    assert checked > 1500
