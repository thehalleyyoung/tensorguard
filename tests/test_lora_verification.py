"""Tests for LoRA/fine-tuning verification support."""

from collections import OrderedDict
import importlib.util

import pytest
import torch
import torch.nn as nn

from src.lora_verification import (
    LoRAAdapter,
    LoRACompatibilityIssue,
    LoRACompatibilityResult,
    LoRAConfig,
    LoRAShapeContract,
    LoRAVerificationResult,
    LoRAVerifier,
    QuantizationBits,
    QuantizationVerifier,
    RankViolation,
    verify_lora_adapter_compatibility,
    verify_lora_model,
)
from src.model_checker import VerificationResult


# ═══════════════════════════════════════════════════════════════════════════════
# Test helpers — simple models with manual LoRA layers
# ═══════════════════════════════════════════════════════════════════════════════


class SimpleLinear(nn.Module):
    """Plain linear model — no LoRA."""

    def __init__(self, in_f=768, out_f=768):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.linear(x)


class LoRALinear(nn.Module):
    """A single linear layer with manual LoRA adapter."""

    def __init__(self, in_features=768, out_features=768, rank=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))
        self.scaling = 1.0

    def forward(self, x):
        base = x @ self.weight.T
        lora = x @ self.lora_A.T @ self.lora_B.T
        return base + self.scaling * lora


class LoRALinearModule(nn.Module):
    """LoRA using nn.Linear sub-modules for A and B."""

    def __init__(self, in_features=768, out_features=768, rank=8):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = 1.0

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


class TwoLayerLoRA(nn.Module):
    """Two-layer MLP with LoRA on both layers."""

    def __init__(self, d_model=256, d_hidden=512, rank=4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc1.lora_A = nn.Parameter(torch.randn(rank, d_model))
        self.fc1.lora_B = nn.Parameter(torch.randn(d_hidden, rank))
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.fc2.lora_A = nn.Parameter(torch.randn(rank, d_hidden))
        self.fc2.lora_B = nn.Parameter(torch.randn(d_model, rank))
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.fc1(x) + x @ self.fc1.lora_A.T @ self.fc1.lora_B.T
        h = self.relu(h)
        out = self.fc2(h) + h @ self.fc2.lora_A.T @ self.fc2.lora_B.T
        return out


class BadRankLoRA(nn.Module):
    """LoRA with rank > min(in, out) — should fail verification."""

    def __init__(self):
        super().__init__()
        # rank=100 but min(32, 64)=32, so rank > min(d, k)
        self.linear = nn.Linear(32, 64)
        self.linear.lora_A = nn.Parameter(torch.randn(100, 32))
        self.linear.lora_B = nn.Parameter(torch.randn(64, 100))

    def forward(self, x):
        return self.linear(x)


class MismatchedLoRA(nn.Module):
    """LoRA with mismatched inner dimensions — merge is unsafe."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 128)
        # A is (8, 64), B is (128, 16) — inner dim mismatch: 8 != 16
        self.linear.lora_A = nn.Parameter(torch.randn(8, 64))
        self.linear.lora_B = nn.Parameter(torch.randn(128, 16))

    def forward(self, x):
        return self.linear(x)


class TransformerWithLoRA(nn.Module):
    """Simple transformer attention with LoRA on Q and V projections."""

    def __init__(self, d_model=512, rank=8):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.q_proj.lora_A = nn.Parameter(torch.randn(rank, d_model))
        self.q_proj.lora_B = nn.Parameter(torch.randn(d_model, rank))
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.v_proj.lora_A = nn.Parameter(torch.randn(rank, d_model))
        self.v_proj.lora_B = nn.Parameter(torch.randn(d_model, rank))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(q + k + v)


class Linear4bit(nn.Linear):
    """Small bitsandbytes-like quantized-linear stand-in for offline tests."""


class PeftLikeLinear(nn.Module):
    """PEFT-shaped LoRA wrapper: base_layer plus lora_A/B ModuleDicts."""

    def __init__(
        self,
        in_features=8,
        out_features=8,
        rank=2,
        *,
        base_cls=nn.Linear,
    ):
        super().__init__()
        self.base_layer = base_cls(in_features, out_features, bias=False)
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(in_features, rank, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(rank, out_features, bias=False)}
        )
        self.merged = False
        self.merged_adapters = []
        self.disable_adapters = False

    def forward(self, x):
        return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))


class PeftLikeBlock(nn.Module):
    """Tiny HuggingFace-style block with PEFT target names."""

    def __init__(self, *, quantized=False):
        super().__init__()
        base_cls = Linear4bit if quantized else nn.Linear
        self.q_proj = PeftLikeLinear(8, 8, 2, base_cls=base_cls)
        self.v_proj = PeftLikeLinear(8, 4, 2, base_cls=base_cls)

    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x)


# ═══════════════════════════════════════════════════════════════════════════════
# LoRAConfig tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoRAConfig:
    def test_default_config(self):
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16.0
        assert cfg.dropout == 0.0
        assert cfg.target_modules == []

    def test_scaling(self):
        cfg = LoRAConfig(rank=4, alpha=8.0)
        assert cfg.scaling == 2.0

    def test_custom_targets(self):
        cfg = LoRAConfig(target_modules=["q_proj", "v_proj"])
        assert len(cfg.target_modules) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# LoRAAdapter tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoRAAdapter:
    def test_adapter_creation(self):
        a = LoRAAdapter(
            base_module_name="fc1",
            in_features=768,
            out_features=768,
            rank=8,
        )
        assert a.lora_A_shape == (8, 768)
        assert a.lora_B_shape == (768, 8)

    def test_adapter_symbolic_dims(self):
        a = LoRAAdapter(
            base_module_name="fc1",
            in_features="d_model",
            out_features="d_model",
            rank=8,
        )
        assert a.lora_A_shape == (8, "d_model")
        assert a.lora_B_shape == ("d_model", 8)

    def test_adapter_custom_shapes(self):
        a = LoRAAdapter(
            base_module_name="fc1",
            in_features=256,
            out_features=512,
            rank=16,
            lora_A_shape=(16, 256),
            lora_B_shape=(512, 16),
        )
        assert a.lora_A_shape == (16, 256)
        assert a.lora_B_shape == (512, 16)


# ═══════════════════════════════════════════════════════════════════════════════
# LoRAShapeContract tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoRAShapeContract:
    def test_valid_contract(self):
        a = LoRAAdapter("fc", 768, 768, rank=8)
        contract = LoRAShapeContract(adapter=a)
        violations = contract.check_concrete()
        assert violations == []

    def test_invalid_rank_too_large(self):
        a = LoRAAdapter("fc", 32, 64, rank=100)
        contract = LoRAShapeContract(adapter=a)
        violations = contract.check_concrete()
        assert len(violations) > 0
        assert "rank" in violations[0].lower()

    def test_invalid_rank_zero(self):
        a = LoRAAdapter("fc", 768, 768, rank=0)
        contract = LoRAShapeContract(adapter=a)
        violations = contract.check_concrete()
        assert len(violations) > 0

    def test_invalid_rank_negative(self):
        a = LoRAAdapter("fc", 768, 768, rank=-1)
        contract = LoRAShapeContract(adapter=a)
        violations = contract.check_concrete()
        assert any("rank" in v.lower() for v in violations)

    def test_mismatched_A_shape(self):
        a = LoRAAdapter(
            "fc", 768, 768, rank=8,
            lora_A_shape=(16, 768),  # rank mismatch: 16 != 8
        )
        contract = LoRAShapeContract(adapter=a)
        violations = contract.check_concrete()
        assert len(violations) > 0

    def test_human_readable_constraints(self):
        a = LoRAAdapter("fc", 768, 256, rank=8)
        contract = LoRAShapeContract(adapter=a)
        constraints = contract.constraints_human()
        assert len(constraints) == 5
        assert any("rank" in c for c in constraints)

    def test_z3_constraints(self):
        a = LoRAAdapter("fc", 768, 768, rank=8)
        contract = LoRAShapeContract(adapter=a)
        z3_cs = contract.to_z3_constraints()
        assert len(z3_cs) >= 3  # rank>0, rank<=in, rank<=out, positivity

    def test_z3_verify_valid(self):
        a = LoRAAdapter("fc", 768, 768, rank=8)
        contract = LoRAShapeContract(adapter=a)
        safe, cex = contract.verify_z3()
        assert safe is True
        assert cex is None

    def test_z3_verify_invalid_rank(self):
        a = LoRAAdapter("fc", 32, 64, rank=100)
        contract = LoRAShapeContract(adapter=a)
        safe, cex = contract.verify_z3()
        assert safe is False

    def test_z3_symbolic_dims(self):
        a = LoRAAdapter("fc", "d_model", "d_model", rank=8)
        contract = LoRAShapeContract(adapter=a)
        z3_cs = contract.to_z3_constraints()
        assert len(z3_cs) >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA detection tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoRADetection:
    def test_detect_no_lora(self):
        model = SimpleLinear()
        verifier = LoRAVerifier(model)
        adapters = verifier.detect_lora_modules()
        assert adapters == []

    def test_detect_parameter_lora(self):
        model = LoRALinear(768, 768, rank=8)
        verifier = LoRAVerifier(model)
        adapters = verifier.detect_lora_modules()
        assert len(adapters) >= 1
        assert adapters[0].rank == 8
        assert adapters[0].in_features == 768

    def test_detect_module_lora(self):
        model = LoRALinearModule(768, 768, rank=8)
        verifier = LoRAVerifier(model)
        adapters = verifier.detect_lora_modules()
        assert len(adapters) >= 1

    def test_detect_two_layer_lora(self):
        model = TwoLayerLoRA(256, 512, rank=4)
        verifier = LoRAVerifier(model)
        adapters = verifier.detect_lora_modules()
        assert len(adapters) == 2

    def test_detect_transformer_lora(self):
        model = TransformerWithLoRA(512, rank=8)
        verifier = LoRAVerifier(model)
        adapters = verifier.detect_lora_modules()
        # q_proj and v_proj have LoRA
        assert len(adapters) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Shape verification tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestShapeVerification:
    def test_valid_shapes(self):
        model = LoRALinear(768, 768, rank=8)
        verifier = LoRAVerifier(model)
        result = verifier.verify_adapter_shapes()
        assert result.safe

    def test_invalid_rank(self):
        model = BadRankLoRA()
        verifier = LoRAVerifier(model)
        result = verifier.verify_adapter_shapes()
        assert not result.safe

    def test_two_layer_valid(self):
        model = TwoLayerLoRA(256, 512, rank=4)
        verifier = LoRAVerifier(model)
        result = verifier.verify_adapter_shapes()
        assert result.safe

    def test_no_lora_model(self):
        model = SimpleLinear()
        verifier = LoRAVerifier(model)
        result = verifier.verify_adapter_shapes()
        assert result.safe  # No adapters = vacuously safe
        assert "No LoRA" in result.errors[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Rank constraint tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRankConstraints:
    def test_valid_rank(self):
        model = LoRALinear(768, 768, rank=8)
        verifier = LoRAVerifier(model)
        violations = verifier.verify_rank_constraints()
        assert violations == []

    def test_rank_exceeds_min(self):
        model = BadRankLoRA()
        verifier = LoRAVerifier(model)
        violations = verifier.verify_rank_constraints()
        assert len(violations) > 0
        assert violations[0].rank == 100

    def test_rank_equals_min(self):
        # rank == min(in, out) is ok
        model = LoRALinear(64, 128, rank=64)
        verifier = LoRAVerifier(model)
        violations = verifier.verify_rank_constraints()
        assert violations == []

    def test_rank_one(self):
        model = LoRALinear(768, 768, rank=1)
        verifier = LoRAVerifier(model)
        violations = verifier.verify_rank_constraints()
        assert violations == []


# ═══════════════════════════════════════════════════════════════════════════════
# Merge safety tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMergeSafety:
    def test_valid_merge(self):
        model = LoRALinear(768, 768, rank=8)
        verifier = LoRAVerifier(model)
        assert verifier.verify_merge_safety() is True

    def test_mismatched_merge(self):
        model = MismatchedLoRA()
        verifier = LoRAVerifier(model)
        assert verifier.verify_merge_safety() is False

    def test_two_layer_merge(self):
        model = TwoLayerLoRA(256, 512, rank=4)
        verifier = LoRAVerifier(model)
        assert verifier.verify_merge_safety() is True


# ═══════════════════════════════════════════════════════════════════════════════
# Composition verification tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestComposition:
    def test_composition_without_source(self):
        model = LoRALinear(768, 768, rank=8)
        verifier = LoRAVerifier(model)
        result = verifier.verify_composition(
            input_shapes={"x": ("batch", 768)}
        )
        assert result.safe

    def test_composition_with_source(self):
        source = """\
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 768)

    def forward(self, x):
        return self.fc(x)
"""
        model = LoRALinear(768, 768, rank=8)
        verifier = LoRAVerifier(model)
        result = verifier.verify_composition(
            input_shapes={"x": ("batch", 768)},
            source=source,
        )
        assert result.safe


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level API tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVerifyLoRAModel:
    def test_model_with_lora(self):
        model = LoRALinear(768, 768, rank=8)
        result = verify_lora_model(model)
        assert isinstance(result, LoRAVerificationResult)
        assert result.has_lora
        assert result.safe

    def test_model_without_lora(self):
        model = SimpleLinear()
        result = verify_lora_model(model)
        assert not result.has_lora
        assert result.safe

    def test_model_bad_rank(self):
        model = BadRankLoRA()
        result = verify_lora_model(model)
        assert result.has_lora
        assert not result.safe
        assert len(result.rank_violations) > 0

    def test_model_mismatched(self):
        model = MismatchedLoRA()
        result = verify_lora_model(model)
        assert not result.merge_safe
        assert not result.safe

    def test_pretty_output(self):
        model = LoRALinear(768, 768, rank=8)
        result = verify_lora_model(model)
        text = result.pretty()
        assert "SAFE" in text
        assert "LoRA detected" in text

    def test_pretty_output_unsafe(self):
        model = BadRankLoRA()
        result = verify_lora_model(model)
        text = result.pretty()
        assert "UNSAFE" in text


# ═══════════════════════════════════════════════════════════════════════════════
# Quantization verifier tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuantizationVerifier:
    def test_detect_no_quantization(self):
        model = SimpleLinear()
        qv = QuantizationVerifier(model)
        assert qv.detect_quantization() == QuantizationBits.FULL

    def test_verify_shapes_no_lora(self):
        model = SimpleLinear()
        qv = QuantizationVerifier(model)
        result = qv.verify_shapes_preserved()
        assert "No LoRA" in result.errors[0]

    def test_verify_shapes_with_lora(self):
        model = LoRALinear(768, 768, rank=8)
        qv = QuantizationVerifier(model)
        result = qv.verify_shapes_preserved()
        assert result.safe


# ═══════════════════════════════════════════════════════════════════════════════
# Additional edge-case tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_very_small_rank(self):
        model = LoRALinear(768, 768, rank=1)
        result = verify_lora_model(model)
        assert result.safe

    def test_rank_equals_dims(self):
        # rank == in == out (full rank LoRA — unusual but valid)
        model = LoRALinear(16, 16, rank=16)
        result = verify_lora_model(model)
        assert result.safe

    def test_asymmetric_dims(self):
        model = LoRALinear(256, 512, rank=8)
        result = verify_lora_model(model)
        assert result.safe

    def test_transformer_lora_verification(self):
        model = TransformerWithLoRA(512, rank=8)
        result = verify_lora_model(model)
        assert result.safe
        assert len(result.adapters) == 2

    def test_config_with_verifier(self):
        model = LoRALinear(768, 768, rank=8)
        config = LoRAConfig(
            rank=8,
            alpha=16.0,
            target_modules=["linear"],
        )
        result = verify_lora_model(model, lora_config=config)
        assert result.safe

    def test_contract_z3_valid_symmetric(self):
        a = LoRAAdapter("fc", 512, 512, rank=16)
        contract = LoRAShapeContract(adapter=a)
        safe, _ = contract.verify_z3()
        assert safe

    def test_rank_violation_dataclass(self):
        v = RankViolation(
            module_name="fc",
            rank=100,
            in_features=32,
            out_features=64,
            message="rank too large",
        )
        assert v.rank == 100
        assert "too large" in v.message


class TestPEFTCompatibility:
    def test_live_peft_module_dict_adapters_are_verified(self):
        model = PeftLikeBlock()

        result = verify_lora_adapter_compatibility(
            model,
            target_modules=["q_proj", "v_proj"],
        )

        assert isinstance(result, LoRACompatibilityResult)
        assert result.ok
        assert result.issues == ()
        assert set(result.checked_targets) == {"q_proj", "v_proj"}
        assert set(result.matched_target_modules) == {"q_proj", "v_proj"}
        assert {adapter.base_module_name for adapter in result.adapters} == {
            "q_proj",
            "v_proj",
        }

    def test_peft_state_dict_keys_resolve_to_base_layer_weight(self):
        model = PeftLikeBlock()
        adapter_state = OrderedDict(
            {
                "base_model.model.q_proj.lora_A.default.weight": torch.randn(2, 8),
                "base_model.model.q_proj.lora_B.default.weight": torch.randn(8, 2),
                "base_model.model.v_proj.lora_A.default.weight": torch.randn(2, 8),
                "base_model.model.v_proj.lora_B.default.weight": torch.randn(4, 2),
            }
        )

        result = verify_lora_adapter_compatibility(
            model,
            adapter_state,
            peft_config={"target_modules": {"q_proj", "v_proj"}},
        )

        assert result.ok
        assert result.issues == ()
        assert set(result.checked_targets) == {"q_proj", "v_proj"}

    def test_shape_rank_and_missing_target_issues_are_actionable(self):
        model = PeftLikeBlock()
        adapter_state = OrderedDict(
            {
                "base_model.model.q_proj.lora_A.default.weight": torch.randn(3, 9),
                "base_model.model.q_proj.lora_B.default.weight": torch.randn(8, 2),
                "base_model.model.missing.lora_A.default.weight": torch.randn(2, 8),
                "base_model.model.missing.lora_B.default.weight": torch.randn(8, 2),
                "base_model.model.v_proj.lora_A.default.weight": torch.randn(2, 8),
            }
        )

        result = verify_lora_adapter_compatibility(
            model,
            adapter_state,
            target_modules=["q_proj", "v_proj"],
        )
        categories = {issue.category for issue in result.issues}

        assert {
            "lora_rank_mismatch",
            "lora_input_mismatch",
            "lora_pair_incomplete",
            "lora_target_missing",
            "lora_target_unexpected",
        } <= categories

    def test_target_module_matching_uses_dot_segment_boundaries(self):
        class PrefixTrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.additional_q_proj = PeftLikeLinear(8, 8, 2)

        result = verify_lora_adapter_compatibility(
            PrefixTrap(),
            target_modules=["q_proj"],
        )
        categories = {issue.category for issue in result.issues}

        assert "lora_target_unexpected" in categories
        assert "lora_target_pattern_missing" in categories

    def test_merged_peft_state_checks_flag_only_expected_state_mismatch(self):
        model = PeftLikeBlock()
        model.q_proj.merged = True
        model.q_proj.merged_adapters = ["default"]

        merged_result = verify_lora_adapter_compatibility(
            model,
            target_modules=["q_proj", "v_proj"],
            expected_merged=True,
        )
        unmerged_result = verify_lora_adapter_compatibility(
            model,
            target_modules=["q_proj", "v_proj"],
            expected_merged=False,
        )

        assert not any(
            issue.category == "lora_merge_state_mismatch"
            for issue in merged_result.issues
            if issue.module_name == "q_proj"
        )
        assert any(
            issue.category == "lora_merge_state_mismatch"
            and issue.module_name == "q_proj"
            for issue in unmerged_result.issues
        )

    def test_quantized_base_accepts_float_adapters_and_rejects_int_adapters(self):
        model = PeftLikeBlock(quantized=True)
        float_result = verify_lora_adapter_compatibility(
            model,
            target_modules=["q_proj", "v_proj"],
            allow_quantized_base=True,
        )
        disallowed_result = verify_lora_adapter_compatibility(
            model,
            target_modules=["q_proj", "v_proj"],
            allow_quantized_base=False,
        )
        bad_state = OrderedDict(
            {
                "base_model.model.q_proj.lora_A.default.weight": torch.ones(
                    2, 8, dtype=torch.int8
                ),
                "base_model.model.q_proj.lora_B.default.weight": torch.randn(8, 2),
            }
        )
        int_adapter_result = verify_lora_adapter_compatibility(
            model,
            bad_state,
            target_modules=["q_proj"],
        )

        assert float_result.ok
        assert set(float_result.quantized_targets) == {"q_proj", "v_proj"}
        assert any(
            issue.category == "lora_quantized_base_disallowed"
            for issue in disallowed_result.issues
        )
        assert any(
            issue.category == "lora_adapter_not_floating"
            for issue in int_adapter_result.issues
        )

    def test_adapter_only_state_reports_base_checks_as_skipped_warning(self):
        adapter_state = OrderedDict(
            {
                "base_model.model.q_proj.lora_A.default.weight": torch.randn(2, 8),
                "base_model.model.q_proj.lora_B.default.weight": torch.randn(8, 2),
            }
        )

        result = verify_lora_adapter_compatibility(adapter_state=adapter_state)

        assert result.ok
        assert result.skipped_checks == ("base_shape:q_proj",)
        assert any(
            warning.category == "lora_base_unverified"
            for warning in result.warnings
        )

    def test_verify_lora_model_includes_compatibility_result(self):
        model = PeftLikeBlock()

        result = verify_lora_model(model, target_modules=["q_proj", "v_proj"])

        assert result.safe
        assert result.compatibility_result is not None
        assert result.compatibility_result.ok
        assert "PEFT issues" in result.pretty()

    def test_public_exports_lora_compatibility_gate(self):
        import tensorguard
        from tensorguard.torch import (
            LoRACompatibilityIssue as PublicIssue,
            LoRACompatibilityResult as PublicResult,
            verify_lora_adapter_compatibility as public_verify,
        )

        assert public_verify is verify_lora_adapter_compatibility
        assert PublicIssue is LoRACompatibilityIssue
        assert PublicResult is LoRACompatibilityResult
        assert tensorguard.verify_lora_adapter_compatibility is verify_lora_adapter_compatibility

    @pytest.mark.skipif(
        importlib.util.find_spec("peft") is None,
        reason="optional HuggingFace PEFT dependency is not installed",
    )
    def test_real_huggingface_peft_linear_example_when_installed(self):
        from peft import LoraConfig, get_peft_model

        class TinyHFStyle(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8, bias=False)

            def forward(self, x):
                return self.q_proj(x)

        peft_model = get_peft_model(
            TinyHFStyle(),
            LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]),
        )

        result = verify_lora_adapter_compatibility(
            peft_model,
            peft_config={"target_modules": ["q_proj"]},
        )

        assert result.ok
        assert result.adapters[0].rank == 2
