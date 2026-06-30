"""Tests for SMT-synthesized fix proposals (src/symexec/fix_synth.py).

These cover the Phase 0 primitives (shape parsing, solver factor synthesis) and
the Phase 1 synthesizers (reshape target, matmul transpose) both in isolation and
end-to-end through the re-verification gate (`repair`). The contract under test:
the synthesizer proposes a *minimal, intent-preserving* edit when it is provably
unambiguous, and abstains (returns nothing / falls back) otherwise.
"""

from __future__ import annotations

from src.symexec import repair
from src.symexec.fix_synth import (
    parse_shape,
    parse_two_shapes,
    solve_inferred_factor,
    synth_reshape_target,
)


# --------------------------------------------------------------------------- #
# Phase 0 primitives.                                                          #
# --------------------------------------------------------------------------- #
def test_parse_shape_ints():
    assert parse_shape("foo (2, 3, 4) bar") == (2, 3, 4)
    assert parse_shape("()") == tuple()


def test_parse_shape_symbolic_returns_none():
    assert parse_shape("(2, b, 4)") is None


def test_parse_two_shapes():
    assert parse_two_shapes("(2, 3) @ (5, 3)") == ((2, 3), (5, 3))
    assert parse_two_shapes("(2, 3)") is None


def test_solve_inferred_factor_unique():
    # 6 * f == 24 -> f == 4, uniquely.
    assert solve_inferred_factor(6, 24) == 4


def test_solve_inferred_factor_non_divisible():
    # 5 does not divide 24 -> no positive integer factor.
    assert solve_inferred_factor(5, 24) is None


def test_solve_inferred_factor_guards():
    assert solve_inferred_factor(0, 24) is None
    assert solve_inferred_factor(6, -1) is None


# --------------------------------------------------------------------------- #
# R1 — reshape target synthesis.                                              #
# --------------------------------------------------------------------------- #
def test_synth_reshape_target_keeps_intended_factor():
    # numel 24, user wrote (6, 5): keep the 6, infer the wrong factor -> (6, -1).
    assert synth_reshape_target(24, (6, 5)) == (6, -1)


def test_synth_reshape_target_ambiguous_abstains():
    # numel 24, (5, 5): neither kept-product (5) divides 24 -> no single-factor
    # repair -> abstain (caller falls back to flatten).
    assert synth_reshape_target(24, (5, 5)) is None


def test_synth_reshape_target_already_valid():
    assert synth_reshape_target(24, (6, 4)) is None


def test_synth_reshape_target_with_existing_neg1_abstains():
    assert synth_reshape_target(24, (6, -1)) is None


# --------------------------------------------------------------------------- #
# End-to-end through the re-verification gate.                                #
# --------------------------------------------------------------------------- #
_RESHAPE_SMART = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    x = torch.randn(2, 3, 4)\n"
    "    y = x.reshape(6, 5)\n"
)
_RESHAPE_BLUNT = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    x = torch.randn(2, 3, 4)\n"
    "    y = x.reshape(5, 5)\n"
)
_MATMUL_T = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    a = torch.randn(2, 3); b = torch.randn(5, 3); c = a @ b\n"
)
_MATMUL_NOFIX = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    a = torch.randn(2, 3); b = torch.randn(4, 5); c = a @ b\n"
)


def test_repair_reshape_uses_synth_target():
    fixes = repair(_RESHAPE_SMART, filename="m.py")
    f = next(x for x in fixes if x.kind == "reshape_size_mismatch")
    assert f.verified
    assert f.strategy == "reshape-synth-target"
    assert "reshape(6, -1)" in f.patched_source


def test_repair_reshape_falls_back_to_flatten():
    fixes = repair(_RESHAPE_BLUNT, filename="m.py")
    f = next(x for x in fixes if x.kind == "reshape_size_mismatch")
    assert f.verified
    assert f.strategy == "reshape-flatten"
    assert "reshape(-1)" in f.patched_source


def test_repair_matmul_transpose_when_stored_transposed():
    fixes = repair(_MATMUL_T, filename="m.py")
    f = next(x for x in fixes if x.kind == "matmul_dim_mismatch")
    assert f.verified
    assert f.strategy == "matmul-transpose"
    assert "b.transpose(-1, -2)" in f.patched_source


def test_repair_matmul_no_transpose_fix_abstains():
    fixes = repair(_MATMUL_NOFIX, filename="m.py")
    assert all(f.kind != "matmul_dim_mismatch" for f in fixes)


def test_synth_fixes_are_deterministic():
    # Same input -> identical synthesized patch across repeated runs.
    out = {repair(_RESHAPE_SMART, filename="m.py")[0].patched_source for _ in range(5)}
    assert len(out) == 1
    out2 = {repair(_MATMUL_T, filename="m.py")[0].patched_source for _ in range(5)}
    assert len(out2) == 1


# --------------------------------------------------------------------------- #
# R5a — repeat: left-pad the repeat-dim list to the tensor rank.               #
# --------------------------------------------------------------------------- #
_REPEAT_INTS = (
    "import torch\n"
    "def f():\n"
    "    x = torch.zeros(2, 3, 4)\n"
    "    return x.repeat(2)\n"
)

_REPEAT_TUPLE = (
    "import torch\n"
    "def f():\n"
    "    x = torch.zeros(2, 3, 4)\n"
    "    return x.repeat((2,))\n"
)


def test_repair_repeat_left_pads_positional_ints():
    fixes = repair(_REPEAT_INTS, filename="m.py")
    f = next(x for x in fixes if x.kind == "repeat_dims_too_few")
    assert f.verified
    assert f.strategy == "repeat-left-pad"
    # rank 3, one dim given -> two leading 1s, user's 2 kept on the trailing axis.
    assert "x.repeat(1, 1, 2)" in f.patched_source


def test_repair_repeat_left_pads_tuple_literal():
    fixes = repair(_REPEAT_TUPLE, filename="m.py")
    f = next(x for x in fixes if x.kind == "repeat_dims_too_few")
    assert f.verified
    assert f.strategy == "repeat-left-pad"
    assert "1, 1, 2" in f.patched_source.splitlines()[-1]


def test_repair_repeat_ambiguous_two_calls_on_one_line_abstains():
    # Two `.repeat(` on the same line -> the receiver is not uniquely locatable.
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    return x.repeat(2), x.repeat(3)\n"
    )
    fixes = repair(src, filename="m.py")
    assert all(f.kind != "repeat_dims_too_few" for f in fixes)


def test_repeat_fix_is_deterministic():
    out = {repair(_REPEAT_INTS, filename="m.py")[0].patched_source for _ in range(5)}
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# R5b — expand: left-pad with `-1` (keep) to the tensor rank.                  #
# --------------------------------------------------------------------------- #
_EXPAND_FEW = (
    "import torch\n"
    "def f():\n"
    "    x = torch.zeros(3, 4)\n"
    "    return x.expand(4)\n"
)


def test_repair_expand_left_pads_with_keep():
    fixes = repair(_EXPAND_FEW, filename="m.py")
    f = next(x for x in fixes if x.kind == "expand_shape_mismatch")
    assert f.verified
    assert f.strategy == "expand-left-pad"
    # rank 2, one size given -> one leading -1 (keep existing dim 0).
    assert "x.expand(-1, 4)" in f.patched_source


def test_repair_expand_non_singleton_mismatch_abstains():
    # expand(5, 4) on (3, 4): dim 0 is a known non-singleton 3 != 5 -> no unique
    # intent-preserving fix, so the synthesizer abstains.
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(3, 4)\n"
        "    return x.expand(5, 4)\n"
    )
    fixes = repair(src, filename="m.py")
    assert all(f.kind != "expand_shape_mismatch" for f in fixes)


# --------------------------------------------------------------------------- #
# R7 — cat: choose the concat dim (sound via the re-verification gate).        #
# --------------------------------------------------------------------------- #
_CAT_KW_DIM = (
    "import torch\n"
    "def f():\n"
    "    a = torch.zeros(2, 3)\n"
    "    b = torch.zeros(2, 5)\n"
    "    return torch.cat([a, b], dim=0)\n"
)

_CAT_NO_DIM = (
    "import torch\n"
    "def f():\n"
    "    a = torch.zeros(2, 3)\n"
    "    b = torch.zeros(2, 5)\n"
    "    return torch.cat([a, b])\n"
)


def test_repair_cat_retargets_dim_kwarg():
    fixes = repair(_CAT_KW_DIM, filename="m.py")
    f = next(x for x in fixes if x.kind == "cat_shape_mismatch")
    assert f.verified
    assert f.strategy == "cat-concat-dim"
    assert "dim=1" in f.patched_source


def test_repair_cat_appends_dim_when_absent():
    fixes = repair(_CAT_NO_DIM, filename="m.py")
    f = next(x for x in fixes if x.kind == "cat_shape_mismatch")
    assert f.verified
    assert "torch.cat([a, b], dim=1)" in f.patched_source


def test_repair_cat_multi_axis_disagreement_abstains():
    # Inputs disagree on BOTH axes: re-pointing dim to one leaves the other
    # broken, so the re-verification gate rejects the edit (abstains).
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(4, 5)\n"
        "    return torch.cat([a, b], dim=0)\n"
    )
    fixes = repair(src, filename="m.py")
    assert all(f.kind != "cat_shape_mismatch" for f in fixes)
