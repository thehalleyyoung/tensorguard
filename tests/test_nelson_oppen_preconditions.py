"""
Tests for Nelson-Oppen/Tinelli-Zarba precondition verification.

Validates that the five-theory combination satisfies all required
preconditions for sound theory combination:
1. Stable infiniteness for infinite-domain theories
2. Polite witnessability for finite-domain theories  
3. Pairwise signature disjointness
"""

import json
import pytest

from src.smt.theory_combination import (
    THEORY_SIGNATURES,
    THEORY_DOMAINS,
    PreconditionReport,
    verify_combination_preconditions,
)


class TestTheorySignatures:
    """Tests for theory signature definitions."""

    def test_all_five_theories_have_signatures(self):
        expected = {"broadcast", "stride", "device", "phase", "permutation"}
        assert set(THEORY_SIGNATURES.keys()) == expected

    def test_all_five_theories_have_domains(self):
        expected = {"broadcast", "stride", "device", "phase", "permutation"}
        assert set(THEORY_DOMAINS.keys()) == expected

    def test_signatures_are_nonempty(self):
        for name, sig in THEORY_SIGNATURES.items():
            assert len(sig) > 0, f"Theory {name} has empty signature"

    def test_broadcast_stride_disjoint(self):
        """Critical check: broadcast and stride must be signature-disjoint."""
        shared = THEORY_SIGNATURES["broadcast"] & THEORY_SIGNATURES["stride"]
        assert len(shared) == 0, f"broadcast-stride share: {shared}"

    def test_permutation_stride_disjoint(self):
        """Critical check: permutation and stride must be signature-disjoint.
        
        This is the pair specifically flagged by Chang in the review:
        'T_perm and T_stride share reasoning about memory layout'.
        We verify they are disjoint at the function symbol level.
        """
        shared = THEORY_SIGNATURES["permutation"] & THEORY_SIGNATURES["stride"]
        assert len(shared) == 0, f"permutation-stride share: {shared}"

    def test_all_pairs_disjoint(self):
        theories = list(THEORY_SIGNATURES.keys())
        for i, t1 in enumerate(theories):
            for t2 in theories[i + 1:]:
                shared = THEORY_SIGNATURES[t1] & THEORY_SIGNATURES[t2]
                assert len(shared) == 0, (
                    f"{t1}-{t2} share symbols: {shared}"
                )


class TestPreconditionVerification:
    """Tests for the full precondition verification."""

    def test_all_preconditions_satisfied(self):
        report = verify_combination_preconditions()
        assert report.all_satisfied

    def test_stable_infiniteness_all_covered(self):
        report = verify_combination_preconditions()
        expected = {"broadcast", "stride", "device", "phase", "permutation"}
        assert set(report.stable_infiniteness.keys()) == expected
        for name, entry in report.stable_infiniteness.items():
            assert entry["satisfied"], f"{name} fails stable infiniteness"

    def test_polite_witnessability_finite_theories(self):
        report = verify_combination_preconditions()
        assert "device" in report.polite_witnessability
        assert "phase" in report.polite_witnessability
        assert report.polite_witnessability["device"]["satisfied"]
        assert report.polite_witnessability["phase"]["satisfied"]
        assert report.polite_witnessability["device"]["domain_size"] == 5
        assert report.polite_witnessability["phase"]["domain_size"] == 2

    def test_signature_disjointness_all_pairs(self):
        report = verify_combination_preconditions()
        for pair_key, entry in report.signature_disjointness.items():
            assert entry["disjoint"], (
                f"Pair {pair_key} not disjoint: {entry['shared_symbols']}"
            )

    def test_shared_sort_analysis(self):
        report = verify_combination_preconditions()
        # Dim sort is shared by broadcast, stride, permutation
        assert "Dim" in report.shared_sort_analysis
        dim_info = report.shared_sort_analysis["Dim"]
        assert "broadcast" in dim_info["theories"]
        assert "stride" in dim_info["theories"]
        assert "permutation" in dim_info["theories"]

    def test_subset_verification(self):
        """Verify preconditions for a subset of theories."""
        report = verify_combination_preconditions(["broadcast", "device"])
        assert report.all_satisfied
        assert len(report.stable_infiniteness) == 2
        assert len(report.polite_witnessability) == 1  # only device

    def test_report_serialization(self):
        report = verify_combination_preconditions()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "all_satisfied" in d
        assert d["all_satisfied"] is True

    def test_device_elements(self):
        report = verify_combination_preconditions()
        device = report.polite_witnessability["device"]
        assert len(device["elements"]) == 5
        assert "CPU" in device["elements"]

    def test_phase_elements(self):
        report = verify_combination_preconditions()
        phase = report.polite_witnessability["phase"]
        assert len(phase["elements"]) == 2
        assert "TRAIN" in phase["elements"]
        assert "EVAL" in phase["elements"]


class TestPermutationStrideInteraction:
    """Specific tests for the T_perm-T_stride interaction.
    
    Addresses Chang's concern that 'transposing changes stride pattern'
    could violate signature disjointness.
    """

    def test_permutation_does_not_use_stride_symbols(self):
        perm_sig = THEORY_SIGNATURES["permutation"]
        stride_symbols = {"stride_val", "contiguous", "stride_eq",
                         "stride_compat", "stride_product", "dim_div"}
        for sym in stride_symbols:
            assert sym not in perm_sig, (
                f"Permutation theory uses stride symbol: {sym}"
            )

    def test_stride_does_not_use_permutation_symbols(self):
        stride_sig = THEORY_SIGNATURES["stride"]
        perm_symbols = {"apply_perm", "perm_compose", "perm_inv",
                       "perm_id", "axis_at", "perm_eq"}
        for sym in perm_symbols:
            assert sym not in stride_sig, (
                f"Stride theory uses permutation symbol: {sym}"
            )

    def test_shared_sort_documented(self):
        """Both share Dim sort but communicate via equality propagation."""
        perm_domain = THEORY_DOMAINS["permutation"]
        stride_domain = THEORY_DOMAINS["stride"]
        assert perm_domain["sort"] == "Dim"
        assert stride_domain["sort"] == "Dim"
        # Both stably infinite on the shared sort
        assert perm_domain["kind"] == "stably_infinite"
        assert stride_domain["kind"] == "stably_infinite"
