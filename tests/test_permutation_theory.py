"""Tests for the permutation theory SMT plugin."""

import pytest

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src.smt.permutation_theory import (
    is_valid_permutation,
    apply_concrete_permutation,
    apply_concrete_transpose,
    compose_permutations,
    inverse_permutation,
    swap_permutation,
    HAS_Z3 as PERM_HAS_Z3,
)

if PERM_HAS_Z3:
    from src.smt.permutation_theory import (
        PermutationPropagator,
        PermutationTheoryPlugin,
        apply_permutation,
        apply_transpose,
    )

pytestmark = pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Pure helper tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPureHelpers:
    def test_valid_permutation_identity(self):
        assert is_valid_permutation((0, 1, 2), 3)

    def test_valid_permutation_swap(self):
        assert is_valid_permutation((1, 0, 2), 3)

    def test_invalid_permutation_wrong_length(self):
        assert not is_valid_permutation((0, 1), 3)

    def test_invalid_permutation_duplicate(self):
        assert not is_valid_permutation((0, 0, 2), 3)

    def test_invalid_permutation_out_of_range(self):
        assert not is_valid_permutation((0, 1, 5), 3)

    def test_apply_concrete_permutation(self):
        assert apply_concrete_permutation((2, 3, 4), (2, 0, 1)) == (4, 2, 3)

    def test_apply_concrete_transpose(self):
        assert apply_concrete_transpose((2, 3, 4), 0, 2) == (4, 3, 2)

    def test_compose_permutations(self):
        p1 = (1, 2, 0)
        p2 = (2, 0, 1)
        composed = compose_permutations(p1, p2)
        shape = (10, 20, 30)
        assert apply_concrete_permutation(
            apply_concrete_permutation(shape, p2), p1
        ) == apply_concrete_permutation(shape, composed)

    def test_inverse_permutation(self):
        p = (2, 0, 1)
        inv = inverse_permutation(p)
        composed = compose_permutations(p, inv)
        assert composed == (0, 1, 2)

    def test_swap_permutation(self):
        s = swap_permutation(4, 1, 3)
        assert s == (0, 3, 2, 1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Z3 transpose constraint tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTransposeEncoding:
    def test_transpose_basic_sat(self):
        """Transpose (2,3,4) swapping dim 0 and 2 should yield (4,3,2)."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1, d2 = z3.Ints("d0 d1 d2")
        o0, o1, o2 = z3.Ints("o0 o1 o2")
        s.add(plugin.apply_transpose([d0, d1, d2], 0, 2, [o0, o1, o2]))
        s.add(d0 == 2, d1 == 3, d2 == 4)
        assert s.check() == z3.sat
        m = s.model()
        assert m[o0].as_long() == 4
        assert m[o1].as_long() == 3
        assert m[o2].as_long() == 2

    def test_transpose_same_dim(self):
        """Transpose swapping a dim with itself is identity."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1 = z3.Ints("d0 d1")
        o0, o1 = z3.Ints("o0 o1")
        s.add(plugin.apply_transpose([d0, d1], 1, 1, [o0, o1]))
        s.add(d0 == 5, d1 == 7)
        assert s.check() == z3.sat
        m = s.model()
        assert m[o0].as_long() == 5
        assert m[o1].as_long() == 7

    def test_transpose_wrong_output_unsat(self):
        """Transpose with contradictory output should be UNSAT."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1, d2 = z3.Ints("d0 d1 d2")
        o0, o1, o2 = z3.Ints("o0 o1 o2")
        s.add(plugin.apply_transpose([d0, d1, d2], 0, 2, [o0, o1, o2]))
        s.add(d0 == 2, d1 == 3, d2 == 4)
        s.add(o0 == 2)  # Should be 4 (from d2), not 2
        assert s.check() == z3.unsat

    def test_transpose_preserves_middle_dim(self):
        """When swapping 0 and 2, dim 1 is preserved."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1, d2 = z3.Ints("d0 d1 d2")
        o0, o1, o2 = z3.Ints("o0 o1 o2")
        s.add(plugin.apply_transpose([d0, d1, d2], 0, 2, [o0, o1, o2]))
        s.add(d0 == 10, d1 == 20, d2 == 30)
        s.add(o1 == 20)  # Middle dim preserved
        assert s.check() == z3.sat

    def test_transpose_invalid_dim_returns_false(self):
        """Out-of-range dim should return BoolVal(False)."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1 = z3.Ints("d0 d1")
        o0, o1 = z3.Ints("o0 o1")
        result = plugin.apply_transpose([d0, d1], 0, 5, [o0, o1])
        assert z3.is_false(result)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Z3 permute constraint tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPermuteEncoding:
    def test_permute_basic_sat(self):
        """Permute (2,3,4) with perm (2,0,1) -> (4,2,3)."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1, d2 = z3.Ints("d0 d1 d2")
        o0, o1, o2 = z3.Ints("o0 o1 o2")
        s.add(plugin.apply_permutation([d0, d1, d2], (2, 0, 1), [o0, o1, o2]))
        s.add(d0 == 2, d1 == 3, d2 == 4)
        assert s.check() == z3.sat
        m = s.model()
        assert m[o0].as_long() == 4
        assert m[o1].as_long() == 2
        assert m[o2].as_long() == 3

    def test_permute_identity(self):
        """Identity permutation preserves all dims."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1, d2 = z3.Ints("d0 d1 d2")
        o0, o1, o2 = z3.Ints("o0 o1 o2")
        s.add(plugin.apply_permutation([d0, d1, d2], (0, 1, 2), [o0, o1, o2]))
        s.add(d0 == 5, d1 == 10, d2 == 15)
        assert s.check() == z3.sat
        m = s.model()
        assert m[o0].as_long() == 5
        assert m[o1].as_long() == 10
        assert m[o2].as_long() == 15

    def test_permute_wrong_output_unsat(self):
        """Permute with contradictory output dims should be UNSAT."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1, d2 = z3.Ints("d0 d1 d2")
        o0, o1, o2 = z3.Ints("o0 o1 o2")
        s.add(plugin.apply_permutation([d0, d1, d2], (2, 0, 1), [o0, o1, o2]))
        s.add(d0 == 2, d1 == 3, d2 == 4)
        s.add(o0 == 999)  # Should be 4
        assert s.check() == z3.unsat

    def test_permute_invalid_perm_returns_false(self):
        """Invalid permutation should return BoolVal(False)."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1 = z3.Ints("d0 d1")
        o0, o1 = z3.Ints("o0 o1")
        result = plugin.apply_permutation([d0, d1], (0, 0), [o0, o1])
        assert z3.is_false(result)

    def test_permute_length_mismatch_returns_false(self):
        """Mismatched input/output length should return BoolVal(False)."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1, d2 = z3.Ints("d0 d1 d2")
        o0, o1 = z3.Ints("o0 o1")
        result = plugin.apply_permutation([d0, d1, d2], (1, 0, 2), [o0, o1])
        assert z3.is_false(result)

    def test_permute_4d_tensor(self):
        """Permute a 4D tensor (NCHW -> NHWC)."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        n, c, h, w = z3.Ints("n c h w")
        on, oc, oh, ow = z3.Ints("on oc oh ow")
        # NCHW -> NHWC: perm = (0, 2, 3, 1)
        s.add(plugin.apply_permutation(
            [n, c, h, w], (0, 2, 3, 1), [on, oh, ow, oc]
        ))
        s.add(n == 1, c == 3, h == 224, w == 224)
        assert s.check() == z3.sat
        m = s.model()
        assert m[on].as_long() == 1
        assert m[oh].as_long() == 224
        assert m[ow].as_long() == 224
        assert m[oc].as_long() == 3


# ═══════════════════════════════════════════════════════════════════════════
# 4. Axis identity / composition tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAxisIdentity:
    def test_double_transpose_is_identity(self):
        """Applying the same transpose twice should return original shape."""
        shape = (2, 3, 4)
        t1 = apply_concrete_transpose(shape, 0, 2)
        t2 = apply_concrete_transpose(t1, 0, 2)
        assert t2 == shape

    def test_permute_then_inverse_is_identity(self):
        """Applying a permutation then its inverse yields identity."""
        shape = (5, 10, 15)
        p = (2, 0, 1)
        inv = inverse_permutation(p)
        result = apply_concrete_permutation(
            apply_concrete_permutation(shape, p), inv
        )
        assert result == shape

    def test_transpose_as_permute(self):
        """Transpose(dim0, dim1) == permute(swap(dim0, dim1))."""
        shape = (2, 3, 4, 5)
        d0, d1 = 1, 3
        t_result = apply_concrete_transpose(shape, d0, d1)
        p = swap_permutation(4, d0, d1)
        p_result = apply_concrete_permutation(shape, p)
        assert t_result == p_result

    def test_axis_identity_z3_double_transpose(self):
        """Z3: two transposes (same axes) should allow original shape."""
        s = z3.Solver()
        plugin = PermutationTheoryPlugin(s)
        d0, d1, d2 = z3.Ints("d0 d1 d2")
        m0, m1, m2 = z3.Ints("m0 m1 m2")
        o0, o1, o2 = z3.Ints("o0 o1 o2")
        # First transpose: swap 0,2
        s.add(plugin.apply_transpose([d0, d1, d2], 0, 2, [m0, m1, m2]))
        # Second transpose: swap 0,2 again
        s2 = z3.Solver()
        plugin2 = PermutationTheoryPlugin(s2)
        s2.add(plugin2.apply_transpose([m0, m1, m2], 0, 2, [o0, o1, o2]))
        # Combine
        combined = z3.Solver()
        combined.add(m0 == d2, m1 == d1, m2 == d0)  # first transpose
        combined.add(o0 == m2, o1 == m1, o2 == m0)  # second transpose
        combined.add(d0 == 10, d1 == 20, d2 == 30)
        assert combined.check() == z3.sat
        cm = combined.model()
        assert cm[o0].as_long() == 10
        assert cm[o1].as_long() == 20
        assert cm[o2].as_long() == 30

    def test_axis_permutation_composition(self):
        """Composing two permutations should match sequential application."""
        p1 = (1, 2, 0)
        p2 = (2, 0, 1)
        composed = compose_permutations(p1, p2)
        shape = (7, 8, 9)
        step1 = apply_concrete_permutation(shape, p2)
        step2 = apply_concrete_permutation(step1, p1)
        direct = apply_concrete_permutation(shape, composed)
        assert step2 == direct


# ═══════════════════════════════════════════════════════════════════════════
# 5. Integration with model_checker patterns
# ═══════════════════════════════════════════════════════════════════════════


class TestModelCheckerIntegration:
    def test_transpose_constraint_generation(self):
        """Simulate how model_checker generates transpose constraints."""
        s = z3.Solver()
        pre_d = [z3.Int(f"pre_{i}") for i in range(4)]
        post_d = [z3.Int(f"post_{i}") for i in range(4)]
        dim0, dim1 = 1, 2
        for i in range(4):
            if i == dim0:
                s.add(post_d[i] == pre_d[dim1])
            elif i == dim1:
                s.add(post_d[i] == pre_d[dim0])
            else:
                s.add(post_d[i] == pre_d[i])
        s.add(pre_d[0] == 1, pre_d[1] == 3, pre_d[2] == 224, pre_d[3] == 224)
        assert s.check() == z3.sat
        m = s.model()
        assert m[post_d[0]].as_long() == 1
        assert m[post_d[1]].as_long() == 224
        assert m[post_d[2]].as_long() == 3
        assert m[post_d[3]].as_long() == 224

    def test_missing_transpose_detected(self):
        """If transpose is missing, output should differ from correct."""
        s = z3.Solver()
        pre_d = [z3.Int(f"pre_{i}") for i in range(3)]
        post_correct = [z3.Int(f"pc_{i}") for i in range(3)]
        post_wrong = [z3.Int(f"pw_{i}") for i in range(3)]

        # Correct: transpose(0,2)
        s.add(post_correct[0] == pre_d[2])
        s.add(post_correct[1] == pre_d[1])
        s.add(post_correct[2] == pre_d[0])

        # Wrong: identity (missing transpose)
        s.add(post_wrong[0] == pre_d[0])
        s.add(post_wrong[1] == pre_d[1])
        s.add(post_wrong[2] == pre_d[2])

        s.add(pre_d[0] == 2, pre_d[1] == 3, pre_d[2] == 4)

        # They should differ
        s.add(z3.Or(
            post_correct[0] != post_wrong[0],
            post_correct[1] != post_wrong[1],
            post_correct[2] != post_wrong[2],
        ))
        assert s.check() == z3.sat

    def test_permute_constraint_generation(self):
        """Simulate model_checker PERMUTE constraint generation."""
        s = z3.Solver()
        pre_d = [z3.Int(f"pre_{i}") for i in range(3)]
        post_d = [z3.Int(f"post_{i}") for i in range(3)]
        perm = (2, 0, 1)
        for i, p in enumerate(perm):
            s.add(post_d[i] == pre_d[p])
        s.add(pre_d[0] == 10, pre_d[1] == 20, pre_d[2] == 30)
        assert s.check() == z3.sat
        m = s.model()
        assert m[post_d[0]].as_long() == 30
        assert m[post_d[1]].as_long() == 10
        assert m[post_d[2]].as_long() == 20
