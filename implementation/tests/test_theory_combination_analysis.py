"""
Tests for the theory combination precondition analysis.

Verifies that:
1. The precondition analysis runs and reports correctly.
2. Stable infiniteness is correctly classified for each theory.
3. Polite witnessability is verified for finite-domain theories.
4. Signature disjointness is checked for all theory pairs.
5. Convexity is correctly identified.
6. The perm-stride interface predicate works correctly.
7. Examples where all 5 theories interact correctly.
"""

from __future__ import annotations

import pytest

from src.theory_combination_analysis import (
    CombinationAnalysisResult,
    ConvexityResult,
    DomainType,
    PermStrideInterface,
    PoliteWitnessability,
    SignatureOverlap,
    StableInfiniteness,
    THEORY_SIGNATURES,
    TheoryCombinationAnalysis,
    TheorySignature,
    verify_combination_preconditions,
)
from src.smt.theory_interface import (
    stride_after_permute_concrete,
    shape_after_permute_concrete,
    is_contiguous_after_permute,
)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. Full analysis runs and reports correctly
# ═══════════════════════════════════════════════════════════════════════════


class TestVerifyCombinationPreconditions:
    """Test the top-level verify_combination_preconditions function."""

    def test_returns_structured_dict(self):
        result = verify_combination_preconditions()
        assert isinstance(result, dict)
        for key in [
            "theories", "stable_infiniteness", "polite_witnessability",
            "signature_disjointness", "convexity", "overall_sound", "caveats",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_all_theories_listed(self):
        result = verify_combination_preconditions()
        assert set(result["theories"]) == set(THEORY_SIGNATURES.keys())

    def test_overall_sound(self):
        result = verify_combination_preconditions()
        assert result["overall_sound"] is True

    def test_has_caveats(self):
        result = verify_combination_preconditions()
        assert len(result["caveats"]) > 0


class TestTheoryCombinationAnalysis:
    """Test the TheoryCombinationAnalysis class."""

    def setup_method(self):
        self.analysis = TheoryCombinationAnalysis()

    def test_run_returns_result(self):
        result = self.analysis.run()
        assert isinstance(result, CombinationAnalysisResult)

    def test_to_dict_roundtrip(self):
        result = self.analysis.run()
        d = result.to_dict()
        assert d["overall_sound"] == result.overall_sound
        assert len(d["theories"]) == len(result.theories)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Stable infiniteness
# ═══════════════════════════════════════════════════════════════════════════


class TestStableInfiniteness:
    """Verify stable infiniteness classification."""

    def setup_method(self):
        self.analysis = TheoryCombinationAnalysis()
        self.results = self.analysis.check_stable_infiniteness()

    def test_shape_is_stably_infinite(self):
        r = self.results["T_shape"]
        assert r.is_stably_infinite is True

    def test_broadcast_is_stably_infinite(self):
        r = self.results["T_broadcast"]
        assert r.is_stably_infinite is True

    def test_stride_is_stably_infinite(self):
        r = self.results["T_stride"]
        assert r.is_stably_infinite is True

    def test_perm_is_stably_infinite(self):
        r = self.results["T_perm"]
        assert r.is_stably_infinite is True

    def test_device_is_not_stably_infinite(self):
        r = self.results["T_device"]
        assert r.is_stably_infinite is False
        assert "Tinelli-Zarba" in r.alternative

    def test_phase_is_not_stably_infinite(self):
        r = self.results["T_phase"]
        assert r.is_stably_infinite is False
        assert "Tinelli-Zarba" in r.alternative

    def test_all_theories_have_justifications(self):
        for name, r in self.results.items():
            assert len(r.justification) > 0, f"{name} missing justification"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Polite witnessability
# ═══════════════════════════════════════════════════════════════════════════


class TestPoliteWitnessability:
    """Verify polite witnessability for finite theories."""

    def setup_method(self):
        self.analysis = TheoryCombinationAnalysis()
        self.results = self.analysis.check_polite_witnessability()

    def test_device_is_polite(self):
        r = self.results["T_device"]
        assert r.is_polite is True

    def test_phase_is_polite(self):
        r = self.results["T_phase"]
        assert r.is_polite is True

    def test_device_domain_elements(self):
        r = self.results["T_device"]
        assert len(r.domain_elements) == 5
        assert "CPU" in r.domain_elements

    def test_phase_domain_elements(self):
        r = self.results["T_phase"]
        assert len(r.domain_elements) == 2
        assert "TRAIN" in r.domain_elements
        assert "EVAL" in r.domain_elements

    def test_only_finite_theories_checked(self):
        assert "T_shape" not in self.results
        assert "T_stride" not in self.results
        assert "T_perm" not in self.results

    def test_witness_extension_proofs_exist(self):
        for name, r in self.results.items():
            assert len(r.witness_extension_proof) > 0, (
                f"{name} missing witness extension proof"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Signature disjointness
# ═══════════════════════════════════════════════════════════════════════════


class TestSignatureDisjointness:
    """Verify pairwise signature disjointness."""

    def setup_method(self):
        self.analysis = TheoryCombinationAnalysis()
        self.overlaps = self.analysis.check_signature_disjointness()

    def test_correct_number_of_pairs(self):
        n = len(THEORY_SIGNATURES)
        expected = n * (n - 1) // 2
        assert len(self.overlaps) == expected

    def test_all_function_symbols_disjoint(self):
        for o in self.overlaps:
            assert len(o.shared_functions) == 0, (
                f"Shared functions between {o.theory_a} and {o.theory_b}: "
                f"{o.shared_functions}"
            )

    def test_all_predicate_symbols_disjoint(self):
        for o in self.overlaps:
            assert len(o.shared_predicates) == 0, (
                f"Shared predicates between {o.theory_a} and {o.theory_b}: "
                f"{o.shared_predicates}"
            )

    def test_perm_stride_share_dim_sort(self):
        perm_stride = [
            o for o in self.overlaps
            if {o.theory_a, o.theory_b} == {"T_perm", "T_stride"}
        ]
        assert len(perm_stride) == 1
        o = perm_stride[0]
        assert "Dim" in o.shared_sorts
        assert o.is_disjoint is True

    def test_perm_stride_has_interface_fix(self):
        perm_stride = [
            o for o in self.overlaps
            if {o.theory_a, o.theory_b} == {"T_perm", "T_stride"}
        ]
        o = perm_stride[0]
        assert "stride_after_permute" in o.fix_applied

    def test_device_phase_disjoint(self):
        dp = [
            o for o in self.overlaps
            if {o.theory_a, o.theory_b} == {"T_device", "T_phase"}
        ]
        assert len(dp) == 1
        assert dp[0].is_disjoint is True


# ═══════════════════════════════════════════════════════════════════════════
# 5. Convexity
# ═══════════════════════════════════════════════════════════════════════════


class TestConvexity:
    """Verify convexity classification."""

    def setup_method(self):
        self.analysis = TheoryCombinationAnalysis()
        self.results = self.analysis.check_convexity()

    def test_shape_is_convex(self):
        assert self.results["T_shape"].is_convex is True

    def test_broadcast_is_convex(self):
        assert self.results["T_broadcast"].is_convex is True

    def test_stride_is_convex(self):
        assert self.results["T_stride"].is_convex is True

    def test_perm_is_convex(self):
        assert self.results["T_perm"].is_convex is True

    def test_device_is_not_convex(self):
        assert self.results["T_device"].is_convex is False

    def test_phase_is_not_convex(self):
        assert self.results["T_phase"].is_convex is False

    def test_non_convex_have_tinelli_zarba_implication(self):
        for name, r in self.results.items():
            if not r.is_convex:
                assert "Tinelli-Zarba" in r.implication or "arrangement" in r.implication


# ═══════════════════════════════════════════════════════════════════════════
# 6. Perm-stride interface predicate
# ═══════════════════════════════════════════════════════════════════════════


class TestPermStrideInterface:
    """Test the perm-stride interface predicate (concrete)."""

    def test_identity_permutation_preserves_strides(self):
        strides = (12, 4, 1)
        perm = (0, 1, 2)
        result = stride_after_permute_concrete(strides, perm)
        assert result == strides

    def test_transpose_reorders_strides(self):
        # shape (2, 3, 4) → strides (12, 4, 1)
        # transpose(0, 2) → shape (4, 3, 2) → strides (1, 4, 12)
        strides = (12, 4, 1)
        perm = (2, 1, 0)
        result = stride_after_permute_concrete(strides, perm)
        assert result == (1, 4, 12)

    def test_cyclic_permutation(self):
        strides = (12, 4, 1)
        perm = (2, 0, 1)
        result = stride_after_permute_concrete(strides, perm)
        assert result == (1, 12, 4)

    def test_shape_after_permute(self):
        shape = (2, 3, 4)
        perm = (2, 0, 1)
        result = shape_after_permute_concrete(shape, perm)
        assert result == (4, 2, 3)

    def test_identity_is_contiguous(self):
        assert is_contiguous_after_permute((2, 3, 4), (0, 1, 2)) is True

    def test_transpose_is_not_contiguous(self):
        assert is_contiguous_after_permute((2, 3, 4), (2, 0, 1)) is False

    def test_permstride_interface_class(self):
        strides = (12, 4, 1)
        perm = (2, 0, 1)
        result = PermStrideInterface.stride_after_permute(strides, perm)
        assert result == (1, 12, 4)

    def test_contiguous_after_identity(self):
        shape = (2, 3, 4)
        strides = (12, 4, 1)
        assert PermStrideInterface.is_contiguous_after_permute(
            shape, strides, (0, 1, 2)
        ) is True

    def test_not_contiguous_after_transpose(self):
        shape = (2, 3, 4)
        strides = (12, 4, 1)
        assert PermStrideInterface.is_contiguous_after_permute(
            shape, strides, (2, 1, 0)
        ) is False

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            stride_after_permute_concrete((12, 4, 1), (0, 1))


# ═══════════════════════════════════════════════════════════════════════════
# 7. Z3 symbolic interface predicate tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
class TestPermStrideSymbolic:
    """Test the Z3 symbolic perm-stride interface."""

    def test_stride_after_permute_sat(self):
        from src.smt.theory_interface import stride_after_permute_symbolic

        s = z3.Solver()
        old_s = z3.Ints("os0 os1 os2")
        new_s = z3.Ints("ns0 ns1 ns2")
        perm = (2, 0, 1)

        s.add(old_s[0] == 12, old_s[1] == 4, old_s[2] == 1)
        s.add(stride_after_permute_symbolic(list(old_s), perm, list(new_s)))

        assert s.check() == z3.sat
        m = s.model()
        assert m[new_s[0]].as_long() == 1
        assert m[new_s[1]].as_long() == 12
        assert m[new_s[2]].as_long() == 4

    def test_contiguous_after_permute_constraint(self):
        from src.smt.theory_interface import contiguous_after_permute_constraint

        s = z3.Solver()
        shape = z3.Ints("d0 d1 d2")
        new_strides = z3.Ints("ns0 ns1 ns2")
        perm = (2, 0, 1)  # shape becomes (d2, d0, d1)

        s.add(shape[0] == 2, shape[1] == 3, shape[2] == 4)
        s.add(contiguous_after_permute_constraint(
            list(shape), perm, list(new_strides)
        ))

        assert s.check() == z3.sat
        m = s.model()
        # permuted shape = (4, 2, 3), contiguous strides = (6, 3, 1)
        assert m[new_strides[0]].as_long() == 6
        assert m[new_strides[1]].as_long() == 3
        assert m[new_strides[2]].as_long() == 1


# ═══════════════════════════════════════════════════════════════════════════
# 8. Multi-theory interaction tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
class TestMultiTheoryInteraction:
    """Test examples where multiple theories interact."""

    def test_perm_then_stride_interaction(self):
        """Permute a tensor and verify stride constraints update."""
        from src.smt.theory_interface import stride_after_permute_symbolic
        from src.smt.stride_theory import compute_contiguous_strides

        s = z3.Solver()

        # Original shape (2, 3, 4) with contiguous strides
        shape = (2, 3, 4)
        old_strides = compute_contiguous_strides(shape)
        assert old_strides == (12, 4, 1)

        # Apply permutation (2, 0, 1) → shape becomes (4, 2, 3)
        perm = (2, 0, 1)
        new_strides_expected = stride_after_permute_concrete(old_strides, perm)
        assert new_strides_expected == (1, 12, 4)

        # Verify symbolically
        os = [z3.IntVal(v) for v in old_strides]
        ns = z3.Ints("ns0 ns1 ns2")
        s.add(stride_after_permute_symbolic(os, perm, list(ns)))

        assert s.check() == z3.sat
        m = s.model()
        for i, expected in enumerate(new_strides_expected):
            assert m[ns[i]].as_long() == expected

    def test_five_theory_combination_setup(self):
        """Verify that all 5 theories can be set up for combination."""
        from src.smt.theory_combination import (
            TensorTheoryCombination,
            DomainKind,
        )

        combo = TensorTheoryCombination()

        # Add broadcast (stably infinite)
        bs = z3.Solver()
        d1, d2 = z3.Ints("bc_d1 bc_d2")
        bs.add(d1 > 0, d2 > 0)
        combo.add_broadcast_theory(bs, [d1, d2])

        # Add stride (stably infinite)
        ss = z3.Solver()
        ss.add(d1 > 0, d2 > 0)
        combo.add_stride_theory(ss, [d1, d2])

        # Add device (finite, 5)
        ds = z3.Solver()
        dev = z3.Int("dev")
        ds.add(dev >= 0, dev < 5)
        combo.add_device_theory(ds, [dev])

        # Add phase (finite, 2)
        ps = z3.Solver()
        phase = z3.Int("phase")
        ps.add(phase >= 0, phase < 2)
        combo.add_phase_theory(ps, [phase])

        # Add permutation (stably infinite)
        perms = z3.Solver()
        perms.add(d1 > 0, d2 > 0)
        combo.add_permutation_theory(perms, [d1, d2])

        assert len(combo.theories) == 5

        # Verify domain kinds
        kinds = {t.name: t.domain_kind for t in combo.theories}
        assert kinds["broadcast"] == DomainKind.STABLY_INFINITE
        assert kinds["stride"] == DomainKind.STABLY_INFINITE
        assert kinds["device"] == DomainKind.FINITE
        assert kinds["phase"] == DomainKind.FINITE
        assert kinds["permutation"] == DomainKind.STABLY_INFINITE


# ═══════════════════════════════════════════════════════════════════════════
# 9. Custom signature analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestCustomSignatureAnalysis:
    """Test with custom/modified signatures to detect issues."""

    def test_overlapping_signatures_detected(self):
        """If two theories share a function symbol, it should be detected."""
        sigs = {
            "T_A": TheorySignature(
                name="T_A",
                sorts=frozenset({"S"}),
                function_symbols=frozenset({"f", "g"}),
                predicate_symbols=frozenset(),
                domain_type=DomainType.STABLY_INFINITE,
            ),
            "T_B": TheorySignature(
                name="T_B",
                sorts=frozenset({"S"}),
                function_symbols=frozenset({"g", "h"}),
                predicate_symbols=frozenset(),
                domain_type=DomainType.STABLY_INFINITE,
            ),
        }
        analysis = TheoryCombinationAnalysis(sigs)
        result = analysis.run()
        assert result.overall_sound is False
        assert any("overlap" in c.lower() or "shared" in c.lower() or "g" in c
                    for c in result.caveats)

    def test_finite_without_polite_detected(self):
        """A finite theory not checked for politeness should be flagged."""
        sigs = {
            "T_weird": TheorySignature(
                name="T_weird",
                sorts=frozenset({"W"}),
                function_symbols=frozenset(),
                predicate_symbols=frozenset(),
                domain_type=DomainType.FINITE,
                domain_size=7,
            ),
        }
        analysis = TheoryCombinationAnalysis(sigs)
        result = analysis.run()
        assert result.overall_sound is False
        assert any("polite" in c.lower() for c in result.caveats)
