"""Tests for proof_certificate module — genuine proof certificates with
inference chains for TensorGuard.
"""
from __future__ import annotations

import json
import textwrap
from typing import List

import pytest

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src.proof_certificate import (
    ProofStep,
    ProofCertificate,
    ProofExtractor,
    CertificateStrategy,
    extract_proof_certificate,
    _extract_unsat_core_certificate,
    _extract_replay_certificate,
    _extract_dual_solver_certificate,
    _guess_theory,
    _is_proof_node,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ProofStep basic tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestProofStep:
    def test_create_asserted(self):
        step = ProofStep(rule="asserted", conclusion="(> x 0)")
        assert step.rule == "asserted"
        assert step.conclusion == "(> x 0)"
        assert step.premises == []
        assert step.theory is None

    def test_create_with_premises(self):
        step = ProofStep(rule="mp", conclusion="false", premises=[0, 1])
        assert step.premises == [0, 1]

    def test_create_theory_lemma(self):
        step = ProofStep(
            rule="th-lemma",
            conclusion="(>= (+ x y) 0)",
            premises=[2],
            theory="arith",
        )
        assert step.theory == "arith"

    def test_to_dict(self):
        step = ProofStep(rule="asserted", conclusion="true")
        d = step.to_dict()
        assert d["rule"] == "asserted"
        assert d["conclusion"] == "true"
        assert d["premises"] == []
        assert "theory" not in d  # None omitted

    def test_to_dict_with_theory(self):
        step = ProofStep(rule="th-lemma", conclusion="c", theory="arith")
        d = step.to_dict()
        assert d["theory"] == "arith"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ProofCertificate tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_small_cert() -> ProofCertificate:
    """Build a minimal valid proof certificate by hand.

    Proof sketch:
      @0  asserted  (> x 0)
      @1  asserted  (<= x 0)
      @2  th-lemma  false   :premises (0 1)  :theory arith
    """
    steps = [
        ProofStep(rule="asserted", conclusion="(> x 0)"),
        ProofStep(rule="asserted", conclusion="(<= x 0)"),
        ProofStep(
            rule="th-lemma",
            conclusion="false",
            premises=[0, 1],
            theory="arith",
        ),
    ]
    return ProofCertificate(
        model_name="TestModel",
        properties=["shape_compatible"],
        steps=steps,
        root_step=2,
        theories_used=["arith"],
        verification_conditions=["(> x 0)", "(<= x 0)"],
    )


class TestProofCertificate:
    def test_verify_locally_valid(self):
        cert = _make_small_cert()
        assert cert.verify_locally() is True

    def test_verify_locally_empty_steps(self):
        cert = ProofCertificate(
            model_name="M",
            properties=[],
            steps=[],
            root_step=0,
        )
        assert cert.verify_locally() is False

    def test_verify_locally_bad_root(self):
        cert = _make_small_cert()
        cert.root_step = 99
        assert cert.verify_locally() is False

    def test_verify_locally_bad_premise_index(self):
        steps = [
            ProofStep(rule="asserted", conclusion="a"),
            ProofStep(rule="mp", conclusion="b", premises=[5]),  # bad
        ]
        cert = ProofCertificate(
            model_name="M", properties=[], steps=steps, root_step=1,
        )
        assert cert.verify_locally() is False

    def test_verify_locally_forward_premise_rejected(self):
        """Premises must refer to earlier (lower-index) steps."""
        steps = [
            ProofStep(rule="mp", conclusion="a", premises=[1]),
            ProofStep(rule="asserted", conclusion="b"),
        ]
        cert = ProofCertificate(
            model_name="M", properties=[], steps=steps, root_step=0,
        )
        assert cert.verify_locally() is False

    def test_verify_locally_asserted_with_premises_rejected(self):
        steps = [
            ProofStep(rule="asserted", conclusion="a", premises=[0]),
        ]
        cert = ProofCertificate(
            model_name="M", properties=[], steps=steps, root_step=0,
        )
        assert cert.verify_locally() is False

    def test_to_alethe_output_format(self):
        cert = _make_small_cert()
        alethe = cert.to_alethe()
        assert "(unsat" in alethe
        assert ":rule assert" in alethe
        assert ":rule th-lemma" in alethe
        assert ":theory arith" in alethe
        assert "@0" in alethe
        assert "@2" in alethe

    def test_to_dict_roundtrip(self):
        cert = _make_small_cert()
        d = cert.to_dict()
        assert d["model_name"] == "TestModel"
        assert len(d["steps"]) == 3
        assert d["root_step"] == 2
        assert isinstance(d["certificate_hash"], str)
        # JSON serializable
        json.dumps(d)

    def test_pretty_output(self):
        cert = _make_small_cert()
        p = cert.pretty()
        assert "ProofCertificate(TestModel)" in p
        assert "Proof steps:" in p
        assert "Theory lemmas:" in p
        assert "Max depth:" in p

    def test_summary_stats(self):
        cert = _make_small_cert()
        stats = cert.summary_stats()
        assert stats["step_count"] == 3
        assert stats["theory_lemma_count"] == 1
        assert stats["max_depth"] == 1
        assert "asserted" in stats["rule_histogram"]
        assert stats["rule_histogram"]["asserted"] == 2

    def test_certificate_hash_deterministic(self):
        c1 = _make_small_cert()
        c2 = _make_small_cert()
        assert c1.certificate_hash == c2.certificate_hash

    def test_certificate_hash_changes_with_content(self):
        c1 = _make_small_cert()
        c2 = _make_small_cert()
        c2.steps[0] = ProofStep(rule="asserted", conclusion="(> y 0)")
        c2.certificate_hash = c2._compute_hash()
        assert c1.certificate_hash != c2.certificate_hash


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ProofExtractor tests (require Z3)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
class TestProofExtractor:
    def test_simple_unsat_qf_lia(self):
        """Extract a proof from a simple contradictory QF_LIA system."""
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            x = z3.Int("x")
            s.add(x > 0)
            s.add(x < 0)
            assert s.check() == z3.unsat

            extractor = ProofExtractor(s, list(s.assertions()))
            cert = extractor.extract(model_name="SimpleTest", properties=["p1"])

            assert cert is not None
            assert len(cert.steps) > 0
            assert cert.verify_locally()
            assert cert.model_name == "SimpleTest"
            assert cert.properties == ["p1"]
        finally:
            z3.set_param("proof", False)

    def test_returns_none_on_sat(self):
        """Proof extraction should return None when the formula is SAT."""
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            x = z3.Int("x")
            s.add(x > 0)
            s.add(x < 10)
            assert s.check() == z3.sat

            extractor = ProofExtractor(s, list(s.assertions()))
            cert = extractor.extract()
            assert cert is None
        finally:
            z3.set_param("proof", False)

    def test_multi_constraint_unsat(self):
        """Extract proof from a more complex UNSAT system."""
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            x, y = z3.Ints("x y")
            s.add(x + y == 5)
            s.add(x > 3)
            s.add(y > 3)
            assert s.check() == z3.unsat

            extractor = ProofExtractor(s, list(s.assertions()))
            cert = extractor.extract(model_name="MultiConstraint")
            assert cert is not None
            assert cert.verify_locally()
            assert len(cert.steps) >= 3
        finally:
            z3.set_param("proof", False)

    def test_extracted_proof_has_asserted_steps(self):
        """The proof should contain at least the asserted leaf steps."""
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            x = z3.Int("x")
            s.add(x > 10)
            s.add(x < 5)
            assert s.check() == z3.unsat

            extractor = ProofExtractor(s, list(s.assertions()))
            cert = extractor.extract()
            assert cert is not None
            asserted_steps = [s for s in cert.steps if s.rule == "asserted"]
            assert len(asserted_steps) >= 1
        finally:
            z3.set_param("proof", False)

    def test_alethe_from_extracted_proof(self):
        """to_alethe() on an extracted proof should produce valid output."""
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            x = z3.Int("x")
            s.add(x >= 0)
            s.add(x <= -1)
            assert s.check() == z3.unsat

            extractor = ProofExtractor(s, list(s.assertions()))
            cert = extractor.extract()
            assert cert is not None
            alethe = cert.to_alethe()
            assert "(unsat" in alethe
        finally:
            z3.set_param("proof", False)

    def test_summary_stats_from_extracted(self):
        """summary_stats() should report sensible numbers."""
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            a, b = z3.Ints("a b")
            s.add(a > b)
            s.add(b > a)
            assert s.check() == z3.unsat

            extractor = ProofExtractor(s, list(s.assertions()))
            cert = extractor.extract()
            assert cert is not None
            stats = cert.summary_stats()
            assert stats["step_count"] > 0
            assert stats["max_depth"] >= 0
        finally:
            z3.set_param("proof", False)

    def test_verification_conditions_recorded(self):
        """The VC strings should be present in the certificate."""
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            x = z3.Int("x")
            s.add(x > 0)
            s.add(x < 0)
            assert s.check() == z3.unsat

            extractor = ProofExtractor(s, list(s.assertions()))
            cert = extractor.extract()
            assert cert is not None
            assert len(cert.verification_conditions) == 2
        finally:
            z3.set_param("proof", False)

    def test_extraction_time_recorded(self):
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            x = z3.Int("x")
            s.add(x > 0)
            s.add(x < 0)
            assert s.check() == z3.unsat

            extractor = ProofExtractor(s, list(s.assertions()))
            cert = extractor.extract()
            assert cert is not None
            assert cert.extraction_time_ms >= 0.0
        finally:
            z3.set_param("proof", False)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Convenience function tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
class TestExtractProofCertificateConvenience:
    def test_convenience_unsat(self):
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            x = z3.Int("x")
            s.add(x > 0)
            s.add(x < 0)
            assert s.check() == z3.unsat

            cert = extract_proof_certificate(
                model_name="Conv",
                properties=["p1"],
                solver=s,
                assertions=list(s.assertions()),
            )
            assert cert is not None
            assert cert.model_name == "Conv"
        finally:
            z3.set_param("proof", False)

    def test_convenience_sat_returns_none(self):
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            x = z3.Int("x")
            s.add(x > 0)
            assert s.check() == z3.sat
            cert = extract_proof_certificate("M", ["p"], s, list(s.assertions()))
            assert cert is None
        finally:
            z3.set_param("proof", False)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Helper function tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_guess_theory_arith(self):
        assert _guess_theory("(+ x 1)") == "arith"
        assert _guess_theory("(<= a b)") == "arith"

    def test_guess_theory_eq(self):
        assert _guess_theory("(= a b)") == "eq"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Integration with model_checker (structural check only)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegrationStructural:
    """Test that SafetyCertificate and VerificationResult have the new field."""

    def test_safety_certificate_has_proof_certificate_field(self):
        from src.model_checker import SafetyCertificate

        cert = SafetyCertificate(
            model_name="M",
            properties=["shape_compatible"],
            k=1,
        )
        assert cert.proof_certificate is None

    def test_verification_result_has_proof_certificate_field(self):
        from src.model_checker import VerificationResult

        vr = VerificationResult(safe=True)
        assert vr.proof_certificate is None

    def test_safety_certificate_with_proof_cert(self):
        from src.model_checker import SafetyCertificate

        pc = _make_small_cert()
        cert = SafetyCertificate(
            model_name="M",
            properties=["p"],
            k=1,
            proof_certificate=pc,
        )
        assert cert.proof_certificate is not None
        assert cert.proof_certificate.verify_locally()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Edge-case & robustness tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_single_asserted_step_valid(self):
        steps = [ProofStep(rule="asserted", conclusion="true")]
        cert = ProofCertificate(
            model_name="M", properties=[], steps=steps, root_step=0,
        )
        assert cert.verify_locally() is True

    def test_rewrite_no_premises_valid(self):
        """rewrite may appear without premises in Z3 proofs."""
        steps = [
            ProofStep(rule="rewrite", conclusion="(= (+ 0 x) x)"),
        ]
        cert = ProofCertificate(
            model_name="M", properties=[], steps=steps, root_step=0,
        )
        assert cert.verify_locally() is True

    def test_negative_root_rejected(self):
        steps = [ProofStep(rule="asserted", conclusion="a")]
        cert = ProofCertificate(
            model_name="M", properties=[], steps=steps, root_step=-1,
        )
        assert cert.verify_locally() is False

    def test_large_proof_verify(self):
        """A chain of 100 mp steps should verify."""
        steps = [ProofStep(rule="asserted", conclusion="a0")]
        for i in range(1, 100):
            steps.append(
                ProofStep(rule="mp", conclusion=f"a{i}", premises=[i - 1])
            )
        cert = ProofCertificate(
            model_name="M", properties=[], steps=steps, root_step=99,
        )
        assert cert.verify_locally() is True

    def test_to_dict_json_serializable(self):
        cert = _make_small_cert()
        s = json.dumps(cert.to_dict())
        assert isinstance(s, str)

    def test_pretty_includes_hash_prefix(self):
        cert = _make_small_cert()
        p = cert.pretty()
        assert cert.certificate_hash[:16] in p


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CertificateStrategy enum tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCertificateStrategy:
    def test_enum_values(self):
        assert CertificateStrategy.Z3_NATIVE_PROOF.value == "z3_native_proof"
        assert CertificateStrategy.UNSAT_CORE.value == "unsat_core"
        assert CertificateStrategy.REPLAY.value == "replay"
        assert CertificateStrategy.DUAL_SOLVER.value == "dual_solver"

    def test_strategy_in_certificate(self):
        cert = _make_small_cert()
        cert.strategy = CertificateStrategy.UNSAT_CORE
        assert cert.strategy == CertificateStrategy.UNSAT_CORE
        d = cert.to_dict()
        assert d["strategy"] == "unsat_core"

    def test_strategy_none_not_in_dict(self):
        cert = _make_small_cert()
        cert.strategy = None
        d = cert.to_dict()
        assert "strategy" not in d

    def test_pretty_shows_strategy(self):
        cert = _make_small_cert()
        cert.strategy = CertificateStrategy.REPLAY
        p = cert.pretty()
        assert "replay" in p


# ═══════════════════════════════════════════════════════════════════════════════
# 9. UNSAT-core certificate strategy tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
class TestUnsatCoreCertificate:
    def test_simple_contradiction(self):
        """UNSAT core strategy should work on a simple contradiction."""
        x = z3.Int("x")
        assertions = [x > 0, x < 0]
        cert = _extract_unsat_core_certificate("M", ["p1"], assertions)
        assert cert is not None
        assert cert.strategy == CertificateStrategy.UNSAT_CORE
        assert cert.verify_locally()
        assert len(cert.steps) >= 2  # at least asserted + th-lemma

    def test_multi_constraint(self):
        """UNSAT core from a system with redundant constraints."""
        x, y = z3.Ints("x y")
        assertions = [x + y == 5, x > 3, y > 3, x >= 0, y >= 0]
        cert = _extract_unsat_core_certificate("M", ["p"], assertions)
        assert cert is not None
        assert cert.verify_locally()
        # Core should be smaller than all assertions
        assert len(cert.verification_conditions) <= len(assertions)

    def test_sat_returns_none(self):
        """UNSAT core strategy returns None on satisfiable formulas."""
        x = z3.Int("x")
        assertions = [x > 0, x < 10]
        cert = _extract_unsat_core_certificate("M", ["p"], assertions)
        assert cert is None

    def test_shape_like_constraints(self):
        """Constraints resembling TensorGuard shape checks."""
        batch, seq, hidden = z3.Ints("batch seq hidden")
        assertions = [
            batch > 0, seq > 0, hidden > 0,
            batch * seq == 128,
            batch > 128,  # contradicts batch * seq == 128 with seq > 0
        ]
        cert = _extract_unsat_core_certificate("ShapeModel", ["shape"], assertions)
        assert cert is not None
        assert cert.verify_locally()

    def test_certificate_hash_populated(self):
        x = z3.Int("x")
        cert = _extract_unsat_core_certificate("M", ["p"], [x > 0, x < 0])
        assert cert is not None
        assert len(cert.certificate_hash) == 64  # SHA-256 hex


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Replay certificate strategy tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
class TestReplayCertificate:
    def test_simple_contradiction(self):
        x = z3.Int("x")
        assertions = [x > 0, x < 0]
        cert = _extract_replay_certificate("M", ["p1"], assertions)
        assert cert is not None
        assert cert.strategy == CertificateStrategy.REPLAY
        assert cert.verify_locally()
        assert len(cert.steps) > 0

    def test_multi_constraint(self):
        x, y = z3.Ints("x y")
        assertions = [x + y == 5, x > 3, y > 3]
        cert = _extract_replay_certificate("M", ["p"], assertions)
        assert cert is not None
        assert cert.verify_locally()

    def test_sat_returns_none(self):
        x = z3.Int("x")
        assertions = [x > 0, x < 10]
        cert = _extract_replay_certificate("M", ["p"], assertions)
        assert cert is None

    def test_lia_constraints(self):
        """QF_LIA constraints (the majority of TensorGuard) should work."""
        a, b, c = z3.Ints("a b c")
        assertions = [
            a == b + c,
            a > 100,
            b <= 50,
            c <= 50,
        ]
        cert = _extract_replay_certificate("LIA", ["shape"], assertions)
        assert cert is not None
        assert cert.verify_locally()

    def test_has_proof_steps_from_z3(self):
        """Replay should produce real Z3 proof steps, not synthetic ones."""
        x = z3.Int("x")
        assertions = [x >= 0, x <= -1]
        cert = _extract_replay_certificate("M", ["p"], assertions)
        assert cert is not None
        # Should have asserted steps (from real proof tree)
        asserted = [s for s in cert.steps if s.rule == "asserted"]
        assert len(asserted) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Dual-solver certificate strategy tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
class TestDualSolverCertificate:
    def test_simple_contradiction(self):
        x = z3.Int("x")
        assertions = [x > 0, x < 0]
        cert = _extract_dual_solver_certificate("M", ["p1"], assertions)
        assert cert is not None
        assert cert.strategy == CertificateStrategy.DUAL_SOLVER
        assert cert.verify_locally()

    def test_sat_returns_none(self):
        x = z3.Int("x")
        assertions = [x > 0, x < 10]
        cert = _extract_dual_solver_certificate("M", ["p"], assertions)
        assert cert is None

    def test_core_minimization(self):
        """Dual solver should use UNSAT core to minimize assertions."""
        x, y = z3.Ints("x y")
        assertions = [x > 0, y > 0, x < 0, y < 100]
        cert = _extract_dual_solver_certificate("M", ["p"], assertions)
        assert cert is not None
        assert cert.verify_locally()
        # Core should exclude y > 0 and y < 100
        assert len(cert.verification_conditions) <= len(assertions)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Fallback chain tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
class TestFallbackChain:
    def test_extract_succeeds_without_proof_mode(self):
        """extract_proof_certificate should succeed even without proof mode
        by falling back to UNSAT-core or replay strategies."""
        s = z3.Solver()
        x = z3.Int("x")
        s.add(x > 0)
        s.add(x < 0)
        assert s.check() == z3.unsat
        cert = extract_proof_certificate("M", ["p"], s, list(s.assertions()))
        assert cert is not None
        assert cert.verify_locally()

    def test_extract_with_proof_mode_uses_native(self):
        """With proof mode enabled, should use Z3 native proof."""
        z3.set_param("proof", True)
        try:
            s = z3.Solver()
            x = z3.Int("x")
            s.add(x > 0)
            s.add(x < 0)
            assert s.check() == z3.unsat
            cert = extract_proof_certificate("M", ["p"], s, list(s.assertions()))
            assert cert is not None
            assert cert.verify_locally()
        finally:
            z3.set_param("proof", False)

    def test_fallback_on_complex_constraints(self):
        """Even complex constraints should get a certificate via fallback."""
        a, b, c, d = z3.Ints("a b c d")
        assertions = [
            a + b == 10,
            c + d == 20,
            a > 5, b > 5,  # contradicts a + b == 10
            c > 0, d > 0,
        ]
        s = z3.Solver()
        for asrt in assertions:
            s.add(asrt)
        assert s.check() == z3.unsat
        cert = extract_proof_certificate("Complex", ["p"], s, assertions)
        assert cert is not None
        assert cert.verify_locally()

    def test_sat_still_returns_none(self):
        """All strategies should correctly return None for SAT."""
        s = z3.Solver()
        x = z3.Int("x")
        s.add(x > 0)
        s.add(x < 10)
        assert s.check() == z3.sat
        cert = extract_proof_certificate("M", ["p"], s, list(s.assertions()))
        assert cert is None


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Coverage test — sample models should achieve >50% certificate rate
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
class TestCertificateCoverage:
    """Verify that multiple certificate strategies achieve high coverage."""

    @staticmethod
    def _make_shape_constraints(i: int) -> List:
        """Generate various UNSAT constraint sets mimicking TensorGuard."""
        x, y, z = z3.Ints(f"x{i} y{i} z{i}")
        scenarios = [
            # 0: Simple dimension mismatch
            [x > 0, x == 10, x == 20],
            # 1: Matmul incompatibility
            [x == 32, y == 64, x == y],
            # 2: Broadcast shape contradiction
            [x > 0, y > 0, x * y == 100, x > 100],
            # 3: Reshape size mismatch
            [x * y == 128, x == 16, y == 16],  # 16*16=256≠128
            # 4: Negative dimension
            [x > 0, y > 0, z == x - y, z < 0, x > y],
            # 5: Batch size overflow
            [x >= 1, x <= 64, x > 128],
            # 6: Channel count mismatch
            [x == 3, y == x * 2, y == 3],
            # 7: Stride constraint
            [x > 0, y > 0, x * y == 1024, x > 1024],
            # 8: Padding contradiction
            [x >= 0, y >= 0, x + y == 5, x > 5],
            # 9: Multi-dim shape
            [x > 0, y > 0, z > 0, x + y + z == 10, x > 10],
            # 10: Linear arithmetic
            [2 * x + 3 * y == 10, x > 5, y > 5],
            # 11: Equality chain
            [x == y, y == z, x > z],
            # 12: Division constraint
            [x > 0, x * 4 == 100],  # no int solution
            # 13: Output shape mismatch
            [x == 32, y == 64, z == x + y, z < 90],
            # 14: Kernel size > input
            [x > 0, x < 3, y == 5, y <= x],
        ]
        return scenarios[i % len(scenarios)]

    def test_coverage_above_50_percent(self):
        """At least 50% of sample UNSAT models should get certificates."""
        total = 15
        certified = 0
        for i in range(total):
            constraints = self._make_shape_constraints(i)
            # Verify they're actually UNSAT first
            s = z3.Solver()
            for c in constraints:
                s.add(c)
            if s.check() != z3.unsat:
                continue  # skip any accidentally SAT scenarios
            cert = extract_proof_certificate(
                f"Model_{i}", [f"prop_{i}"], s, constraints,
            )
            if cert is not None and cert.verify_locally():
                certified += 1
        coverage = certified / total
        assert coverage >= 0.50, (
            f"Certificate coverage {coverage:.1%} ({certified}/{total}) "
            f"is below 50% threshold"
        )

    def test_coverage_details(self):
        """Report which strategies were used for coverage."""
        total = 15
        strategy_counts: dict = {}
        for i in range(total):
            constraints = self._make_shape_constraints(i)
            s = z3.Solver()
            for c in constraints:
                s.add(c)
            if s.check() != z3.unsat:
                continue
            cert = extract_proof_certificate(
                f"Model_{i}", [f"prop_{i}"], s, constraints,
            )
            if cert is not None and cert.verify_locally():
                strat = cert.strategy.value if cert.strategy else "unknown"
                strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
        assert len(strategy_counts) > 0, "No certificates were produced"
