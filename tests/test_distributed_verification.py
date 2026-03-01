"""Tests for distributed verification (FSDP, DeepSpeed, adapter composition)."""

import math
import pytest

from src.distributed_verification import (
    AdapterComposition,
    AdapterCompositionVerifier,
    AdapterMergeStrategy,
    DeepSpeedConfig,
    DeepSpeedVerifier,
    DistributedVerificationResult,
    FSDPConfig,
    FSDPShardingVerifier,
    ParamShardInfo,
    ShardingResult,
    WrapPolicy,
    ZeROStage,
    verify_distributed,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FSDP Config tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFSDPConfig:
    def test_default_config(self):
        cfg = FSDPConfig()
        assert cfg.world_size == 1
        assert cfg.auto_wrap_policy == WrapPolicy.NONE
        assert cfg.min_num_params == 100_000
        assert cfg.sharding_strategy == "FULL_SHARD"

    def test_custom_config(self):
        cfg = FSDPConfig(
            world_size=8,
            auto_wrap_policy=WrapPolicy.SIZE_BASED,
            min_num_params=50_000,
        )
        assert cfg.world_size == 8
        assert cfg.auto_wrap_policy == WrapPolicy.SIZE_BASED
        assert cfg.min_num_params == 50_000

    def test_transformer_wrap_config(self):
        cfg = FSDPConfig(
            world_size=4,
            auto_wrap_policy=WrapPolicy.TRANSFORMER_BASED,
            transformer_layer_cls=["TransformerEncoderLayer"],
        )
        assert len(cfg.transformer_layer_cls) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSpeed Config tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeepSpeedConfig:
    def test_default_config(self):
        cfg = DeepSpeedConfig()
        assert cfg.stage == ZeROStage.STAGE_1
        assert cfg.dp_world_size == 1

    def test_stage_3_config(self):
        cfg = DeepSpeedConfig(
            stage=ZeROStage.STAGE_3,
            dp_world_size=8,
            offload_param=True,
        )
        assert cfg.stage == ZeROStage.STAGE_3
        assert cfg.offload_param is True


# ═══════════════════════════════════════════════════════════════════════════════
# FSDP Sharding Verifier tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFSDPShardingVerifier:
    def test_single_rank_no_sharding(self):
        cfg = FSDPConfig(world_size=1)
        verifier = FSDPShardingVerifier(cfg)
        params = {"fc.weight": (768, 512)}
        result = verifier.verify_shard_consistency(params)
        assert result.safe
        assert result.params_checked == 1

    def test_multi_rank_consistent(self):
        cfg = FSDPConfig(world_size=4)
        verifier = FSDPShardingVerifier(cfg)
        params = {
            "fc1.weight": (256, 128),
            "fc2.weight": (128, 256),
        }
        result = verifier.verify_shard_consistency(params)
        assert result.safe
        assert result.params_checked == 2

    def test_world_size_zero(self):
        cfg = FSDPConfig(world_size=0)
        verifier = FSDPShardingVerifier(cfg)
        params = {"fc.weight": (64, 64)}
        result = verifier.verify_shard_consistency(params)
        assert not result.safe
        assert any("positive" in v for v in result.violations)

    def test_negative_world_size(self):
        cfg = FSDPConfig(world_size=-1)
        verifier = FSDPShardingVerifier(cfg)
        params = {"fc.weight": (64, 64)}
        result = verifier.verify_shard_consistency(params)
        assert not result.safe

    def test_shard_info_populated(self):
        cfg = FSDPConfig(world_size=8)
        verifier = FSDPShardingVerifier(cfg)
        params = {"fc.weight": (1024, 512)}
        result = verifier.verify_shard_consistency(params)
        assert result.safe
        assert len(result.shard_info) == 1
        info = result.shard_info[0]
        assert info.name == "fc.weight"
        assert info.original_shape == (1024, 512)
        assert info.world_size == 8
        numel = 1024 * 512
        assert info.numel == numel
        assert info.shard_size == math.ceil(numel / 8)

    def test_large_model_many_params(self):
        cfg = FSDPConfig(world_size=16)
        verifier = FSDPShardingVerifier(cfg)
        params = {
            f"layer.{i}.weight": (512, 512) for i in range(24)
        }
        result = verifier.verify_shard_consistency(params)
        assert result.safe
        assert result.params_checked == 24

    def test_symbolic_dimensions(self):
        cfg = FSDPConfig(world_size=4)
        verifier = FSDPShardingVerifier(cfg)
        params = {"fc.weight": ("d_model", 256)}
        result = verifier.verify_shard_consistency(params)
        # Symbolic dims go through Z3 path — should still be safe
        assert result.safe

    def test_gather_scatter_shapes(self):
        cfg = FSDPConfig(world_size=4)
        verifier = FSDPShardingVerifier(cfg)
        params = {"fc.weight": (256, 128)}
        result = verifier.verify_gather_scatter_shapes(params)
        assert result.safe

    def test_size_based_wrapping(self):
        cfg = FSDPConfig(
            world_size=4,
            auto_wrap_policy=WrapPolicy.SIZE_BASED,
            min_num_params=10_000,
        )
        verifier = FSDPShardingVerifier(cfg)
        params = {
            "big_layer.weight": (256, 256),     # 65536 >= 10000
            "small_layer.weight": (10, 10),     # 100 < 10000
        }
        wrap = verifier.detect_wrapping(params)
        assert wrap["big_layer.weight"] is True
        assert wrap["small_layer.weight"] is False

    def test_use_orig_params(self):
        cfg = FSDPConfig(world_size=2, use_orig_params=True)
        verifier = FSDPShardingVerifier(cfg)
        params = {"fc.weight": (100, 50)}
        result = verifier.verify_shard_consistency(params)
        assert result.safe

    def test_shard_grad_op_strategy(self):
        cfg = FSDPConfig(
            world_size=4,
            sharding_strategy="SHARD_GRAD_OP",
        )
        verifier = FSDPShardingVerifier(cfg)
        params = {"fc.weight": (512, 256)}
        result = verifier.verify_shard_consistency(params)
        assert result.safe


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSpeed Verifier tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeepSpeedVerifier:
    def test_stage_0_valid(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_0, dp_world_size=4)
        verifier = DeepSpeedVerifier(cfg)
        params = {"fc.weight": (512, 256)}
        result = verifier.verify_stage(params)
        assert result.safe

    def test_stage_0_empty_shape(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_0, dp_world_size=4)
        verifier = DeepSpeedVerifier(cfg)
        params = {"fc.weight": ()}
        result = verifier.verify_stage(params)
        assert not result.safe
        assert any("empty" in v for v in result.violations)

    def test_stage_0_negative_dim(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_0, dp_world_size=4)
        verifier = DeepSpeedVerifier(cfg)
        params = {"fc.weight": (512, -1)}
        result = verifier.verify_stage(params)
        assert not result.safe
        assert any("non-positive" in v for v in result.violations)

    def test_stage_1_valid(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_1, dp_world_size=8)
        verifier = DeepSpeedVerifier(cfg)
        params = {
            "fc1.weight": (256, 128),
            "fc2.weight": (128, 256),
        }
        result = verifier.verify_stage(params)
        assert result.safe

    def test_stage_1_invalid_dp(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_1, dp_world_size=0)
        verifier = DeepSpeedVerifier(cfg)
        params = {"fc.weight": (256, 128)}
        result = verifier.verify_stage(params)
        assert not result.safe
        assert any("positive" in v for v in result.violations)

    def test_stage_2_valid(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_2, dp_world_size=4)
        verifier = DeepSpeedVerifier(cfg)
        params = {"fc.weight": (1024, 512)}
        result = verifier.verify_stage(params)
        assert result.safe

    def test_stage_2_invalid_dp(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_2, dp_world_size=0)
        verifier = DeepSpeedVerifier(cfg)
        params = {"fc.weight": (256, 128)}
        result = verifier.verify_stage(params)
        assert not result.safe

    def test_stage_2_symbolic(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_2, dp_world_size=4)
        verifier = DeepSpeedVerifier(cfg)
        params = {"fc.weight": ("d_model", 256)}
        result = verifier.verify_stage(params)
        assert result.safe

    def test_stage_3_valid(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_3, dp_world_size=8)
        verifier = DeepSpeedVerifier(cfg)
        params = {
            "fc1.weight": (512, 256),
            "fc2.weight": (256, 512),
        }
        result = verifier.verify_stage(params)
        assert result.safe

    def test_stage_3_shard_info(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_3, dp_world_size=4)
        verifier = DeepSpeedVerifier(cfg)
        params = {"fc.weight": (100, 50)}
        result = verifier.verify_stage(params)
        assert result.safe
        assert len(result.shard_info) == 1
        info = result.shard_info[0]
        assert info.name == "fc.weight"
        assert info.numel == 5000
        assert info.shard_size == math.ceil(5000 / 4)

    def test_stage_3_invalid_dp(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_3, dp_world_size=-1)
        verifier = DeepSpeedVerifier(cfg)
        params = {"fc.weight": (256, 128)}
        result = verifier.verify_stage(params)
        assert not result.safe

    def test_stage_3_symbolic(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_3, dp_world_size=8)
        verifier = DeepSpeedVerifier(cfg)
        params = {"fc.weight": ("hidden", "d_model")}
        result = verifier.verify_stage(params)
        assert result.safe

    def test_stage_3_many_params(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_3, dp_world_size=16)
        verifier = DeepSpeedVerifier(cfg)
        params = {f"layer.{i}.weight": (768, 768) for i in range(12)}
        result = verifier.verify_stage(params)
        assert result.safe
        assert result.params_checked == 12


# ═══════════════════════════════════════════════════════════════════════════════
# Adapter Composition tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdapterComposition:
    def test_single_adapter_add(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.ADD,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert result.safe

    def test_two_adapters_add_compatible(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
                {"name": "A2", "in_features": 768, "out_features": 768, "rank": 16},
            ],
            strategy=AdapterMergeStrategy.ADD,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert result.safe

    def test_two_adapters_add_incompatible(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
                {"name": "A2", "in_features": 512, "out_features": 768, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.ADD,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert not result.safe
        assert any("in_features" in v for v in result.violations)

    def test_adapter_rank_too_large(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 32, "out_features": 64, "rank": 100},
            ],
            strategy=AdapterMergeStrategy.ADD,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert not result.safe
        assert any("rank" in v for v in result.violations)

    def test_adapter_rank_zero(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 0},
            ],
            strategy=AdapterMergeStrategy.ADD,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert not result.safe

    def test_stack_compatible(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 256, "rank": 8},
                {"name": "A2", "in_features": 256, "out_features": 128, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.STACK,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert result.safe

    def test_stack_incompatible(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 256, "rank": 8},
                {"name": "A2", "in_features": 512, "out_features": 128, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.STACK,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert not result.safe
        assert any("out_features" in v or "in_features" in v
                    for v in result.violations)

    def test_stack_three_adapters(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 512, "rank": 8},
                {"name": "A2", "in_features": 512, "out_features": 256, "rank": 8},
                {"name": "A3", "in_features": 256, "out_features": 128, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.STACK,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert result.safe

    def test_switch_compatible(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
                {"name": "A2", "in_features": 768, "out_features": 768, "rank": 16},
                {"name": "A3", "in_features": 768, "out_features": 768, "rank": 4},
            ],
            strategy=AdapterMergeStrategy.SWITCH,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert result.safe

    def test_switch_incompatible_output(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
                {"name": "A2", "in_features": 768, "out_features": 512, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.SWITCH,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert not result.safe
        assert any("switching" in v for v in result.violations)

    def test_weighted_add_correct_weights(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
                {"name": "A2", "in_features": 768, "out_features": 768, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.WEIGHTED_ADD,
            weights=[0.5, 0.5],
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert result.safe

    def test_weighted_add_wrong_weight_count(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
                {"name": "A2", "in_features": 768, "out_features": 768, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.WEIGHTED_ADD,
            weights=[0.5],  # Should be 2 weights
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert not result.safe
        assert any("Weight count" in v for v in result.violations)

    def test_empty_composition(self):
        comp = AdapterComposition(
            adapters=[],
            strategy=AdapterMergeStrategy.ADD,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert result.safe


# ═══════════════════════════════════════════════════════════════════════════════
# Z3 Composition Verification tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestZ3Composition:
    def test_z3_add_valid(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
                {"name": "A2", "in_features": 768, "out_features": 768, "rank": 16},
            ],
            strategy=AdapterMergeStrategy.ADD,
        )
        verifier = AdapterCompositionVerifier(comp)
        safe, cex = verifier.verify_z3_composition()
        assert safe is True
        assert cex is None

    def test_z3_stack_valid(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 256, "rank": 8},
                {"name": "A2", "in_features": 256, "out_features": 128, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.STACK,
        )
        verifier = AdapterCompositionVerifier(comp)
        safe, cex = verifier.verify_z3_composition()
        assert safe is True

    def test_z3_stack_invalid(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 256, "rank": 8},
                {"name": "A2", "in_features": 512, "out_features": 128, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.STACK,
        )
        verifier = AdapterCompositionVerifier(comp)
        safe, cex = verifier.verify_z3_composition()
        assert safe is False
        assert cex is not None

    def test_z3_switch_invalid(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
                {"name": "A2", "in_features": 768, "out_features": 512, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.SWITCH,
        )
        verifier = AdapterCompositionVerifier(comp)
        safe, cex = verifier.verify_z3_composition()
        assert safe is False


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: verify_distributed tests
# ═══════════════════════════════════════════════════════════════════════════════


SIMPLE_MODEL = '''
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)
'''

CONV_MODEL = '''
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        h = self.conv(x)
        return self.fc(h.view(h.size(0), -1))
'''


class TestVerifyDistributed:
    def test_base_only(self):
        result = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
        )
        assert isinstance(result, DistributedVerificationResult)
        assert result.base_result is not None

    def test_with_fsdp(self):
        result = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
            fsdp_config=FSDPConfig(world_size=4),
        )
        assert result.fsdp_result is not None
        assert result.fsdp_result.safe

    def test_with_deepspeed_stage_1(self):
        result = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
            deepspeed_config=DeepSpeedConfig(
                stage=ZeROStage.STAGE_1,
                dp_world_size=4,
            ),
        )
        assert result.deepspeed_result is not None
        assert result.deepspeed_result.safe

    def test_with_deepspeed_stage_3(self):
        result = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
            deepspeed_config=DeepSpeedConfig(
                stage=ZeROStage.STAGE_3,
                dp_world_size=8,
            ),
        )
        assert result.deepspeed_result is not None
        assert result.deepspeed_result.safe

    def test_with_adapter_composition(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 256, "out_features": 256, "rank": 8},
                {"name": "A2", "in_features": 256, "out_features": 256, "rank": 4},
            ],
            strategy=AdapterMergeStrategy.ADD,
        )
        result = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
            adapter_composition=comp,
        )
        assert result.adapter_result is not None
        assert result.adapter_result.safe

    def test_fsdp_and_deepspeed_together(self):
        result = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
            fsdp_config=FSDPConfig(world_size=4),
            deepspeed_config=DeepSpeedConfig(
                stage=ZeROStage.STAGE_2,
                dp_world_size=4,
            ),
        )
        assert result.fsdp_result is not None
        assert result.deepspeed_result is not None

    def test_all_checks_together(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 256, "out_features": 256, "rank": 4},
            ],
            strategy=AdapterMergeStrategy.ADD,
        )
        result = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
            fsdp_config=FSDPConfig(world_size=8),
            deepspeed_config=DeepSpeedConfig(
                stage=ZeROStage.STAGE_3,
                dp_world_size=8,
            ),
            adapter_composition=comp,
        )
        assert result.fsdp_result is not None
        assert result.deepspeed_result is not None
        assert result.adapter_result is not None

    def test_pretty_output(self):
        result = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
            fsdp_config=FSDPConfig(world_size=4),
        )
        pretty = result.pretty()
        assert "DistributedVerificationResult" in pretty

    def test_invalid_source(self):
        result = verify_distributed(
            source="this is not valid python",
            input_shapes={},
        )
        assert result.base_result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases & misc
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_param_shard_info_dataclass(self):
        info = ParamShardInfo(
            name="fc.weight",
            original_shape=(256, 128),
            numel=32768,
            shard_size=8192,
            world_size=4,
        )
        assert info.name == "fc.weight"
        assert info.is_flat is False

    def test_sharding_result_pretty(self):
        r = ShardingResult(
            safe=False,
            violations=["test violation"],
            params_checked=3,
        )
        pretty = r.pretty()
        assert "UNSAFE" in pretty
        assert "test violation" in pretty

    def test_fsdp_numel_computation(self):
        cfg = FSDPConfig(world_size=2)
        v = FSDPShardingVerifier(cfg)
        assert v._compute_numel((10, 20, 30)) == 6000
        assert v._compute_numel(()) == 0
        sym = v._compute_numel(("batch", 256))
        assert isinstance(sym, str)
        assert "batch" in sym

    def test_fsdp_shard_size_computation(self):
        cfg = FSDPConfig(world_size=2)
        v = FSDPShardingVerifier(cfg)
        assert v._compute_shard_size(100, 4) == 25
        assert v._compute_shard_size(101, 4) == 26
        sym = v._compute_shard_size("N", 4)
        assert isinstance(sym, str)

    def test_deepspeed_partition_size(self):
        cfg = DeepSpeedConfig(stage=ZeROStage.STAGE_3, dp_world_size=4)
        v = DeepSpeedVerifier(cfg)
        assert v._compute_partition_size(100, 4) == 25
        assert v._compute_partition_size(101, 4) == 26

    def test_single_adapter_stack(self):
        """A single adapter in stack mode is vacuously safe."""
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.STACK,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert result.safe

    def test_single_adapter_switch(self):
        comp = AdapterComposition(
            adapters=[
                {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
            ],
            strategy=AdapterMergeStrategy.SWITCH,
        )
        verifier = AdapterCompositionVerifier(comp)
        result = verifier.verify()
        assert result.safe

    def test_z3_empty_composition(self):
        comp = AdapterComposition(adapters=[], strategy=AdapterMergeStrategy.ADD)
        verifier = AdapterCompositionVerifier(comp)
        safe, cex = verifier.verify_z3_composition()
        assert safe is True

    def test_conv_model_fsdp(self):
        result = verify_distributed(
            source=CONV_MODEL,
            input_shapes={"x": ("batch", 3, 32, 32)},
            fsdp_config=FSDPConfig(world_size=2),
        )
        assert result.fsdp_result is not None
