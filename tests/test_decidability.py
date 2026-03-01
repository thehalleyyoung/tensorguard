"""Tests for the decidability characterization module."""

from __future__ import annotations

import pytest

from src.decidability import (
    ComplexityClass,
    DecidabilitySummary,
    NIAAnalysisResult,
    RelationalConstraintClass,
    RelationalConstraintClassifier,
    RelationalConstraintInfo,
    TheoryFragment,
    VerificationQuery,
    analyze_nia_fragment,
    classify_query_complexity,
    classify_relational_constraint,
    identify_fragments,
    summarize_decidability,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _q(ops=(), device=False, phase=False) -> VerificationQuery:
    return VerificationQuery(
        operations=frozenset(ops),
        has_device_constraints=device,
        has_phase_constraints=phase,
    )


# ── classify_query_complexity ────────────────────────────────────────────────

class TestClassifyQueryComplexity:
    """Verify that the complexity classifier returns the correct class."""

    def test_empty_query_is_p(self):
        assert classify_query_complexity(_q()) == ComplexityClass.P

    def test_linear_ops_only_is_p(self):
        q = _q(["MATMUL", "ADD", "ACTIVATION", "LAYER_CALL"])
        assert classify_query_complexity(q) == ComplexityClass.P

    def test_reshape_is_np_hard(self):
        q = _q(["RESHAPE"])
        assert classify_query_complexity(q) == ComplexityClass.NP_HARD

    def test_flatten_is_np_hard(self):
        q = _q(["FLATTEN"])
        assert classify_query_complexity(q) == ComplexityClass.NP_HARD

    def test_reshape_with_linear_ops_is_np_hard(self):
        q = _q(["MATMUL", "ADD", "RESHAPE", "ACTIVATION"])
        assert classify_query_complexity(q) == ComplexityClass.NP_HARD

    def test_device_only_is_p(self):
        q = _q(["TO_DEVICE"], device=True)
        assert classify_query_complexity(q) == ComplexityClass.P

    def test_phase_only_is_p(self):
        q = _q(["DROPOUT"], phase=True)
        assert classify_query_complexity(q) == ComplexityClass.P

    def test_all_theories_without_reshape_is_p(self):
        q = _q(["MATMUL", "ADD", "TO_DEVICE", "DROPOUT"], device=True, phase=True)
        assert classify_query_complexity(q) == ComplexityClass.P

    def test_all_theories_with_reshape_is_np_hard(self):
        q = _q(["MATMUL", "RESHAPE", "TO_DEVICE", "DROPOUT"], device=True, phase=True)
        assert classify_query_complexity(q) == ComplexityClass.NP_HARD


# ── identify_fragments ──────────────────────────────────────────────────────

class TestIdentifyFragments:
    """Verify that the correct theory fragments are identified."""

    def test_empty_query_has_linear(self):
        frags = identify_fragments(_q())
        assert TheoryFragment.T_SHAPE_LINEAR in frags
        assert len(frags) == 1

    def test_matmul_adds_matmul_fragment(self):
        frags = identify_fragments(_q(["MATMUL"]))
        assert TheoryFragment.T_SHAPE_MATMUL in frags

    def test_add_adds_broadcast_fragment(self):
        frags = identify_fragments(_q(["ADD"]))
        assert TheoryFragment.T_SHAPE_BROADCAST in frags

    def test_multiply_adds_broadcast_fragment(self):
        frags = identify_fragments(_q(["MULTIPLY"]))
        assert TheoryFragment.T_SHAPE_BROADCAST in frags

    def test_reshape_adds_reshape_fragment(self):
        frags = identify_fragments(_q(["RESHAPE"]))
        assert TheoryFragment.T_SHAPE_RESHAPE in frags

    def test_flatten_adds_reshape_fragment(self):
        frags = identify_fragments(_q(["FLATTEN"]))
        assert TheoryFragment.T_SHAPE_RESHAPE in frags

    def test_to_device_adds_device_fragment(self):
        frags = identify_fragments(_q(["TO_DEVICE"]))
        assert TheoryFragment.T_DEVICE in frags

    def test_device_flag_adds_device_fragment(self):
        frags = identify_fragments(_q(device=True))
        assert TheoryFragment.T_DEVICE in frags

    def test_dropout_adds_phase_fragment(self):
        frags = identify_fragments(_q(["DROPOUT"]))
        assert TheoryFragment.T_PHASE in frags

    def test_phase_flag_adds_phase_fragment(self):
        frags = identify_fragments(_q(phase=True))
        assert TheoryFragment.T_PHASE in frags

    def test_conditional_adds_phase_fragment(self):
        frags = identify_fragments(_q(["CONDITIONAL"]))
        assert TheoryFragment.T_PHASE in frags

    def test_full_pipeline_fragments(self):
        q = _q(["MATMUL", "ADD", "RESHAPE", "TO_DEVICE", "DROPOUT"])
        frags = identify_fragments(q)
        expected = {
            TheoryFragment.T_SHAPE_LINEAR,
            TheoryFragment.T_SHAPE_MATMUL,
            TheoryFragment.T_SHAPE_BROADCAST,
            TheoryFragment.T_SHAPE_RESHAPE,
            TheoryFragment.T_DEVICE,
            TheoryFragment.T_PHASE,
        }
        assert frags == expected


# ── summarize_decidability ───────────────────────────────────────────────────

class TestSummarizeDecidability:
    """Verify that the decidability summary is well-formed."""

    def test_linear_summary_mentions_p(self):
        s = summarize_decidability(_q(["MATMUL", "ADD"]))
        assert s.complexity == ComplexityClass.P
        assert "QF_LIA" in s.explanation
        assert "decidable in P" in s.explanation

    def test_reshape_summary_mentions_np(self):
        s = summarize_decidability(_q(["RESHAPE"]))
        assert s.complexity == ComplexityClass.NP_HARD
        assert "NP-hard" in s.explanation
        assert "SUBSET-PRODUCT" in s.explanation

    def test_device_summary_mentions_finite(self):
        s = summarize_decidability(_q(device=True))
        assert "finite domain" in s.explanation
        assert "5 elements" in s.explanation

    def test_phase_summary_mentions_finite(self):
        s = summarize_decidability(_q(phase=True))
        assert "finite domain" in s.explanation
        assert "2 elements" in s.explanation

    def test_combined_summary_mentions_tinelli_zarba(self):
        q = _q(["MATMUL", "TO_DEVICE"], device=True, phase=True)
        s = summarize_decidability(q)
        assert "Tinelli-Zarba" in s.explanation

    def test_summary_returns_dataclass(self):
        s = summarize_decidability(_q())
        assert isinstance(s, DecidabilitySummary)
        assert isinstance(s.fragments, set)
        assert isinstance(s.complexity, ComplexityClass)
        assert isinstance(s.explanation, str)


# ── classify_relational_constraint ──────────────────────────────────────────

class TestClassifyRelationalConstraint:
    """Tests for the relational constraint classification API."""

    def test_concrete_int_is_lia(self):
        info = classify_relational_constraint("embed_dim", 512)
        assert info.classification == RelationalConstraintClass.QF_LIA_REDUCIBLE
        assert info.symbolic_vars == []

    def test_single_symbolic_times_constant_is_lia(self):
        info = classify_relational_constraint("ffn_dim", "4 * embed_dim")
        assert info.classification == RelationalConstraintClass.QF_LIA_REDUCIBLE
        assert info.symbolic_vars == ["embed_dim"]

    def test_two_symbolic_mult_is_nia(self):
        info = classify_relational_constraint("embed_dim", "heads * head_dim")
        assert info.classification == RelationalConstraintClass.QF_NIA
        assert set(info.symbolic_vars) == {"heads", "head_dim"}

    def test_concrete_dims_make_mult_lia(self):
        info = classify_relational_constraint(
            "embed_dim", "heads * head_dim",
            concrete_dims={"heads": 8},
        )
        assert info.classification == RelationalConstraintClass.QF_LIA_REDUCIBLE

    def test_addition_is_lia(self):
        info = classify_relational_constraint("total", "a + b")
        assert info.classification == RelationalConstraintClass.QF_LIA_REDUCIBLE

    def test_triple_product_is_nia(self):
        info = classify_relational_constraint("vol", "d * h * w")
        assert info.classification == RelationalConstraintClass.QF_NIA

    def test_identity_is_lia(self):
        info = classify_relational_constraint("out_dim", "in_dim")
        assert info.classification == RelationalConstraintClass.QF_LIA_REDUCIBLE

    def test_info_fields(self):
        info = classify_relational_constraint("embed_dim", "heads * head_dim")
        assert info.lhs == "embed_dim"
        assert info.expression == "heads * head_dim"
        assert isinstance(info.reason, str) and len(info.reason) > 0


# ── RelationalConstraintClassifier ──────────────────────────────────────────

class TestRelationalConstraintClassifier:
    """Tests for the classifier class directly."""

    def test_classify_all(self):
        c = RelationalConstraintClassifier()
        infos = c.classify_all({
            "embed_dim": "heads * head_dim",
            "ffn_dim": "4 * embed_dim",
        })
        assert len(infos) == 2
        classes = {i.lhs: i.classification for i in infos}
        assert classes["embed_dim"] == RelationalConstraintClass.QF_NIA
        assert classes["ffn_dim"] == RelationalConstraintClass.QF_LIA_REDUCIBLE

    def test_concrete_dims_respected(self):
        c = RelationalConstraintClassifier(concrete_dims={"heads": 8, "head_dim": 64})
        info = c.classify("embed_dim", "heads * head_dim")
        assert info.classification == RelationalConstraintClass.QF_LIA_REDUCIBLE
        assert info.symbolic_vars == []

    def test_partial_concrete(self):
        c = RelationalConstraintClassifier(concrete_dims={"heads": 8})
        info = c.classify("embed_dim", "heads * head_dim")
        assert info.classification == RelationalConstraintClass.QF_LIA_REDUCIBLE
        assert info.symbolic_vars == ["head_dim"]

    def test_mixed_add_mult_nia(self):
        c = RelationalConstraintClassifier()
        info = c.classify("out", "a * b + c")
        assert info.classification == RelationalConstraintClass.QF_NIA

    def test_subtraction_is_lia(self):
        c = RelationalConstraintClassifier()
        info = c.classify("out_h", "in_h - 2")
        assert info.classification == RelationalConstraintClass.QF_LIA_REDUCIBLE

    def test_squared_single_var_is_nia(self):
        c = RelationalConstraintClassifier()
        info = c.classify("area", "side * side")
        assert info.classification == RelationalConstraintClass.QF_NIA


# ── analyze_nia_fragment ────────────────────────────────────────────────────

class TestAnalyzeNIAFragment:
    """Tests for Z3-based NIA fragment analysis."""

    def test_concrete_sat(self):
        result = analyze_nia_fragment({"x": 10})
        assert result.status == "sat"
        assert result.model is not None
        assert result.model["x"] == 10

    def test_linear_constraint_sat(self):
        result = analyze_nia_fragment({"y": "2 * x"})
        assert result.status == "sat"
        assert result.model is not None
        assert result.model["y"] == 2 * result.model["x"]

    def test_nonlinear_constraint_sat(self):
        result = analyze_nia_fragment({"z": "a * b"})
        assert result.status == "sat"
        assert result.model is not None
        assert result.model["z"] == result.model["a"] * result.model["b"]

    def test_unsat_system(self):
        result = analyze_nia_fragment(
            {"x": 5},
            extra_bounds={"x": (10, 20)},
        )
        assert result.status == "unsat"
        assert result.model is None

    def test_returns_timing(self):
        result = analyze_nia_fragment({"x": "2 * y"})
        assert result.elapsed_s >= 0

    def test_result_fields(self):
        result = analyze_nia_fragment({"x": 42})
        assert isinstance(result, NIAAnalysisResult)
        assert isinstance(result.status, str)
        assert isinstance(result.elapsed_s, float)
        assert isinstance(result.timed_out, bool)
