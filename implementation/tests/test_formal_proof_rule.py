"""Tests for formal proof rule and cache invalidation in assume_guarantee."""

from __future__ import annotations

import pytest

from src.assume_guarantee import (
    CacheInvalidationSpec,
    CompositionProofRule,
    CompositionalResult,
    DecompositionStrategy,
    FormalProofRule,
    InterfaceCheck,
    InterfaceContract,
    ProofObligation,
    SubModule,
    VerificationCache,
    compute_cache_invalidation,
    validate_proof_rule,
)
from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    Device,
    OpKind,
    Phase,
    VerificationResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _make_step(op: OpKind, output: str, inputs: list = None) -> ComputationStep:
    return ComputationStep(
        op=op,
        output=output,
        inputs=inputs or ["x"],
        layer_ref=None,
        params={},
    )


def _make_submodule(name: str, steps: list = None) -> SubModule:
    s = steps or [_make_step(OpKind.ACTIVATION, f"{name}_out")]
    graph = ComputationGraph(
        class_name="TestModel",
        steps=s,
        input_names=["x"],
        output_names=[f"{name}_out"],
        layers={},
    )
    return SubModule(
        name=name,
        graph=graph,
        input_contract=InterfaceContract(
            name=f"I_{name}",
            input_shapes={"x": ("batch", 10)},
        ),
        output_contract=InterfaceContract(
            name=f"O_{name}",
            output_shapes={f"{name}_out": ("batch", 10)},
        ),
    )


def _make_verification_result(safe: bool = True) -> VerificationResult:
    return VerificationResult(
        safe=safe,
        errors=[] if safe else ["Shape mismatch"],
        verification_time_ms=10.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ProofObligation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestProofObligation:
    def test_creation(self):
        obl = ProofObligation(
            kind="submodule_safety",
            submodule="block_0",
            description="{I_0} block_0 {O_0}",
            satisfied=True,
            evidence="Verified safe",
        )
        assert obl.satisfied
        assert obl.kind == "submodule_safety"


# ═══════════════════════════════════════════════════════════════════════════════
# FormalProofRule tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormalProofRule:
    def test_pretty_output(self):
        rule = FormalProofRule(
            obligations=[
                ProofObligation("submodule_safety", "M0", "{I0} M0 {O0}", True),
                ProofObligation("interface_compatibility", "M0→M1", "O0 ⊑ I1", True),
                ProofObligation("input_precondition", "M0", "I0 ok", True),
            ],
            conclusion_holds=True,
        )
        text = rule.pretty()
        assert "Abadi-Lamport" in text
        assert "SAFE" in text

    def test_failed_conclusion(self):
        rule = FormalProofRule(
            obligations=[
                ProofObligation("submodule_safety", "M0", "test", False),
            ],
            conclusion_holds=False,
        )
        text = rule.pretty()
        assert "NOT ESTABLISHED" in text


# ═══════════════════════════════════════════════════════════════════════════════
# validate_proof_rule tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateProofRule:
    def test_all_safe_all_compatible(self):
        sms = [_make_submodule("M0"), _make_submodule("M1")]
        results = {
            "M0": _make_verification_result(True),
            "M1": _make_verification_result(True),
        }
        checks = [InterfaceCheck("M0", "M1", True, "OK")]
        rule = validate_proof_rule(sms, results, checks)
        assert rule.conclusion_holds

    def test_unsafe_submodule_fails_conclusion(self):
        sms = [_make_submodule("M0"), _make_submodule("M1")]
        results = {
            "M0": _make_verification_result(True),
            "M1": _make_verification_result(False),
        }
        checks = [InterfaceCheck("M0", "M1", True, "OK")]
        rule = validate_proof_rule(sms, results, checks)
        assert not rule.conclusion_holds

    def test_incompatible_interface_fails(self):
        sms = [_make_submodule("M0"), _make_submodule("M1")]
        results = {
            "M0": _make_verification_result(True),
            "M1": _make_verification_result(True),
        }
        checks = [InterfaceCheck("M0", "M1", False, "Shape mismatch")]
        rule = validate_proof_rule(sms, results, checks)
        assert not rule.conclusion_holds

    def test_obligations_count(self):
        sms = [_make_submodule("M0"), _make_submodule("M1"), _make_submodule("M2")]
        results = {
            "M0": _make_verification_result(True),
            "M1": _make_verification_result(True),
            "M2": _make_verification_result(True),
        }
        checks = [
            InterfaceCheck("M0", "M1", True, "OK"),
            InterfaceCheck("M1", "M2", True, "OK"),
        ]
        rule = validate_proof_rule(sms, results, checks)
        # 3 submodule safety + 2 interface + 1 input = 6
        assert len(rule.obligations) == 6
        assert rule.conclusion_holds

    def test_single_submodule(self):
        sms = [_make_submodule("M0")]
        results = {"M0": _make_verification_result(True)}
        rule = validate_proof_rule(sms, results, [])
        assert rule.conclusion_holds

    def test_missing_result_fails(self):
        sms = [_make_submodule("M0")]
        rule = validate_proof_rule(sms, {}, [])
        assert not rule.conclusion_holds


# ═══════════════════════════════════════════════════════════════════════════════
# CacheInvalidationSpec tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheInvalidation:
    def test_middle_module_change(self):
        sms = [_make_submodule("M0"), _make_submodule("M1"), _make_submodule("M2")]
        spec = compute_cache_invalidation(sms, "M1")
        assert spec.changed_index == 1
        assert "M1" in spec.modules_to_reverify
        assert "M2" in spec.modules_to_reverify
        assert "M0" in spec.modules_cached
        assert "M0" not in spec.modules_to_reverify

    def test_first_module_change(self):
        sms = [_make_submodule("M0"), _make_submodule("M1"), _make_submodule("M2")]
        spec = compute_cache_invalidation(sms, "M0")
        assert spec.changed_index == 0
        assert len(spec.modules_to_reverify) == 3
        assert len(spec.modules_cached) == 0

    def test_last_module_change(self):
        sms = [_make_submodule("M0"), _make_submodule("M1"), _make_submodule("M2")]
        spec = compute_cache_invalidation(sms, "M2")
        assert spec.changed_index == 2
        assert spec.modules_to_reverify == ["M2"]
        assert spec.modules_cached == ["M0", "M1"]

    def test_unknown_module(self):
        sms = [_make_submodule("M0"), _make_submodule("M1")]
        spec = compute_cache_invalidation(sms, "M_unknown")
        assert spec.changed_index == -1
        assert len(spec.modules_to_reverify) == 0
        assert len(spec.modules_cached) == 2

    def test_interface_recheck_middle(self):
        sms = [_make_submodule("M0"), _make_submodule("M1"), _make_submodule("M2")]
        spec = compute_cache_invalidation(sms, "M1")
        # Should recheck (M0, M1) and (M1, M2)
        assert ("M0", "M1") in spec.interfaces_to_recheck
        assert ("M1", "M2") in spec.interfaces_to_recheck

    def test_interface_recheck_first(self):
        sms = [_make_submodule("M0"), _make_submodule("M1")]
        spec = compute_cache_invalidation(sms, "M0")
        assert ("M0", "M1") in spec.interfaces_to_recheck
