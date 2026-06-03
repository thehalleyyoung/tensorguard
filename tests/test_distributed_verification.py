"""Tests for distributed verification (FSDP, DeepSpeed, adapter composition)."""

import math
import pytest

from src.distributed_verification import (
    AdapterComposition,
    AdapterCompositionVerifier,
    AdapterMergeStrategy,
    DeepSpeedConfig,
    DeepSpeedVerifier,
    DistributedPlacement,
    DistributedVerificationResult,
    DTensorPlacement,
    DTensorSpec,
    DTensorVerifier,
    FSDPConfig,
    FSDP2Config,
    FSDP2Verifier,
    FSDPShardingVerifier,
    ParameterShardingSpec,
    ParameterShardingStrategy,
    ParameterShardingVerifier,
    ParamShardInfo,
    PipelineBoundarySpec,
    PipelineParallelVerifier,
    PipelineStageSpec,
    ShardingResult,
    WrapPolicy,
    ZeROStage,
    verify_dtensor_specs,
    verify_distributed,
    verify_parameter_sharding,
    verify_pipeline_boundaries,
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
# DTensor / FSDP2 / per-parameter sharding tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDTensorVerifier:
    def test_unpadded_shard_local_shapes_are_rank_specific(self):
        spec = DTensorSpec(
            name="w",
            global_shape=(10, 6),
            mesh_shape=(3,),
            placements=(DTensorPlacement.shard(0),),
        )
        result = DTensorVerifier().verify_specs([spec])
        assert result.safe
        assert result.local_shapes["w"] == {
            (0,): (4, 6),
            (1,): (4, 6),
            (2,): (2, 6),
        }
        tiny = verify_dtensor_specs([
            DTensorSpec(
                name="tiny",
                global_shape=(2, 6),
                mesh_shape=(3,),
                placements=(DTensorPlacement.shard(0),),
            )
        ])
        assert tiny.safe
        assert tiny.local_shapes["tiny"][(2,)] == (0, 6)

    def test_negative_shard_dim_and_rank_coordinate_expected_shape(self):
        spec = DTensorSpec(
            name="proj.weight",
            global_shape=(8, 5),
            mesh_shape=(2,),
            placements=(DTensorPlacement.shard(-1),),
            rank_coordinate=(1,),
            expected_local_shape=(8, 2),
        )
        result = verify_dtensor_specs([spec])
        assert result.safe
        assert result.local_shapes["proj.weight"] == {(1,): (8, 2)}

    def test_invalid_dtensor_mesh_and_placements_refuted(self):
        spec = DTensorSpec(
            name="bad",
            global_shape=(8, 5),
            mesh_shape=(2, 2),
            placements=(DTensorPlacement.shard(0),),
        )
        result = verify_dtensor_specs([spec])
        assert not result.safe
        assert any("placements length" in v for v in result.violations)

    def test_expected_shape_without_rank_abstains_for_uneven_shards(self):
        spec = DTensorSpec(
            name="uneven",
            global_shape=(10, 6),
            mesh_shape=(3,),
            placements=(DTensorPlacement.shard(0),),
            expected_local_shape=(4, 6),
        )
        result = verify_dtensor_specs([spec])
        assert result.safe
        assert any("provide rank_coordinate" in w for w in result.warnings)

    def test_partial_placement_warns_but_preserves_shape(self):
        spec = DTensorSpec(
            name="partial_sum",
            global_shape=(4, 4),
            mesh_shape=(2,),
            placements=(DTensorPlacement.partial(),),
        )
        result = verify_dtensor_specs([spec])
        assert result.safe
        assert result.local_shapes["partial_sum"][(0,)] == (4, 4)
        assert any("Partial placement" in w for w in result.warnings)

    def test_public_placement_enum_values(self):
        assert DTensorPlacement.replicate().kind is DistributedPlacement.REPLICATE
        assert DTensorPlacement.shard(1).dim == 1

    def test_public_package_exports_dtensor_helpers(self):
        import tensorguard

        assert tensorguard.FSDP2Config is FSDP2Config
        assert tensorguard.DTensorSpec is DTensorSpec
        assert tensorguard.verify_dtensor_specs is verify_dtensor_specs

    def test_real_torch_shard_split_oracle_when_available(self):
        torch = pytest.importorskip("torch")
        placement_types = pytest.importorskip(
            "torch.distributed.tensor.placement_types"
        )
        shard = placement_types.Shard(0)
        for rows in (10, 2):
            chunks, _ = shard._split_tensor(
                torch.empty(rows, 6),
                3,
                with_padding=False,
                contiguous=True,
            )
            oracle_shapes = [tuple(chunk.shape) for chunk in chunks]

            spec = DTensorSpec(
                name=f"torch_oracle_{rows}",
                global_shape=(rows, 6),
                mesh_shape=(3,),
                placements=(DTensorPlacement.shard(0),),
            )
            result = verify_dtensor_specs([spec])
            static_shapes = [
                result.local_shapes[f"torch_oracle_{rows}"][(rank,)]
                for rank in range(3)
            ]
            assert static_shapes == oracle_shapes


class TestParameterShardingVerifier:
    def test_rank_specific_expected_shape_mismatch_refuted(self):
        spec = ParameterShardingSpec(
            name="fc.weight",
            shape=(10, 5),
            strategy=ParameterShardingStrategy.SHARD,
            world_size=3,
            shard_dim=0,
            rank_coordinate=(2,),
            expected_local_shape=(4, 5),
        )
        result = verify_parameter_sharding([spec])
        assert not result.safe
        assert any("local shape at rank" in v for v in result.violations)

    def test_dtensor_strategy_on_2d_mesh(self):
        spec = ParameterShardingSpec(
            name="mlp.weight",
            shape=(8, 9),
            strategy=ParameterShardingStrategy.DTENSOR,
            world_size=6,
            mesh_shape=(2, 3),
            placements=(DTensorPlacement.shard(0), DTensorPlacement.shard(1)),
            rank_coordinate=(1, 2),
            expected_local_shape=(4, 3),
        )
        result = ParameterShardingVerifier().verify_specs([spec])
        assert result.safe
        assert result.local_shapes["mlp.weight"] == {(1, 2): (4, 3)}

    def test_mesh_product_must_match_world_size(self):
        spec = ParameterShardingSpec(
            name="bad_mesh.weight",
            shape=(8, 9),
            strategy=ParameterShardingStrategy.DTENSOR,
            world_size=8,
            mesh_shape=(2, 3),
            placements=(DTensorPlacement.shard(0), DTensorPlacement.shard(1)),
        )
        result = verify_parameter_sharding([spec])
        assert not result.safe
        assert any("mesh product" in v for v in result.violations)


class TestFSDP2Verifier:
    def test_fsdp2_default_per_parameter_shapes_from_source(self):
        result = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
            fsdp2_config=FSDP2Config(world_size=3),
        )
        assert result.fsdp2_result is not None
        assert result.fsdp2_result.safe
        assert result.fsdp2_result.local_shapes["fc1.weight"][(2,)] == (42, 256)

    def test_fsdp2_override_invalid_expected_shape_flips_top_level(self):
        overrides = {
            "fc1.weight": ParameterShardingSpec(
                name="fc1.weight",
                strategy=ParameterShardingStrategy.FULLY_SHARD,
                world_size=3,
                shard_dim=0,
                rank_coordinate=(2,),
                expected_local_shape=(43, 256),
            )
        }
        result = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
            fsdp2_config=FSDP2Config(world_size=3, parameter_overrides=overrides),
        )
        assert result.fsdp2_result is not None
        assert not result.fsdp2_result.safe
        assert not result.safe
        assert any("local shape at rank" in v
                   for v in result.fsdp2_result.violations)

    def test_fsdp2_mesh_product_must_equal_world_size(self):
        verifier = FSDP2Verifier(FSDP2Config(world_size=4, mesh_shape=(3,)))
        result = verifier.verify_sharding({"w": (8, 8)})
        assert not result.safe
        assert any("mesh product" in v for v in result.violations)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline-parallel boundary tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipelineParallelVerifier:
    def test_valid_stage_boundary_with_microbatch_contract(self):
        stages = [
            PipelineStageSpec(index=0, name="embed", output_shape=(8, 128)),
            PipelineStageSpec(index=1, name="mlp", input_shape=(8, 128)),
        ]
        boundary = PipelineBoundarySpec(
            name="embed_to_mlp",
            producer_stage=0,
            consumer_stage=1,
            activation_shape=(8, 128),
            microbatch_dim=0,
            microbatches=4,
            global_batch_size=32,
            activation_dtype="torch.float16",
            expected_dtype="torch.float16",
            activation_device="cuda:1",
            expected_device="cuda:1",
        )
        result = PipelineParallelVerifier(stages).verify_boundaries([boundary])
        assert result.safe
        assert result.params_checked == 1
        assert result.local_shapes["embed_to_mlp"][(0, 1)] == (8, 128)

    def test_shape_mismatch_refutes_boundary_and_top_level(self):
        boundary = PipelineBoundarySpec(
            name="bad_hidden",
            producer_stage=0,
            consumer_stage=1,
            activation_shape=(8, 128),
            expected_input_shape=(8, 256),
        )
        direct = verify_pipeline_boundaries([boundary])
        assert not direct.safe
        assert any("consumer expected input" in v for v in direct.violations)

        aggregate = verify_distributed(
            source=SIMPLE_MODEL,
            input_shapes={"x": ("batch", 256)},
            pipeline_boundaries=[boundary],
        )
        assert aggregate.pipeline_result is not None
        assert not aggregate.pipeline_result.safe
        assert not aggregate.safe
        assert "Pipeline" in aggregate.pretty()

    def test_microbatch_mismatch_and_invalid_axis_refute(self):
        mismatch = PipelineBoundarySpec(
            name="bad_microbatch",
            producer_stage=0,
            consumer_stage=1,
            activation_shape=(15, 128),
            microbatches=4,
            global_batch_size=64,
        )
        bad_axis = PipelineBoundarySpec(
            name="bad_axis",
            producer_stage=0,
            consumer_stage=1,
            activation_shape=(8, 128),
            microbatch_dim=3,
        )
        result = verify_pipeline_boundaries([mismatch, bad_axis])
        assert not result.safe
        assert any("per-microbatch dimension" in v for v in result.violations)
        assert any("microbatch_dim" in v for v in result.violations)

    def test_uneven_final_microbatch_warns_without_refuting(self):
        boundary = PipelineBoundarySpec(
            name="uneven",
            producer_stage=0,
            consumer_stage=1,
            activation_shape=(8, 128),
            microbatches=4,
            global_batch_size=30,
        )
        result = verify_pipeline_boundaries([boundary])
        assert result.safe
        assert any("uneven final microbatches" in w for w in result.warnings)

    def test_dtype_and_device_mismatch_refute_only_when_expected_is_explicit(self):
        implicit = PipelineBoundarySpec(
            name="implicit_runtime_transfer",
            producer_stage=0,
            consumer_stage=1,
            activation_shape=(8, 128),
            activation_dtype="torch.float16",
            activation_device="cuda:0",
        )
        explicit = PipelineBoundarySpec(
            name="bad_boundary_contract",
            producer_stage=0,
            consumer_stage=1,
            activation_shape=(8, 128),
            activation_dtype="torch.float16",
            expected_dtype="torch.float32",
            activation_device="cuda:0",
            expected_device="cuda:1",
        )
        implicit_result = verify_pipeline_boundaries([implicit])
        assert implicit_result.safe

        stage_expected = PipelineParallelVerifier([
            PipelineStageSpec(index=0),
            PipelineStageSpec(index=1, dtype="torch.float32", device="cuda:1"),
        ]).verify_boundaries([implicit])
        assert not stage_expected.safe
        assert any("activation dtype" in v for v in stage_expected.violations)
        assert any("activation device" in v for v in stage_expected.violations)

        explicit_result = verify_pipeline_boundaries([explicit])
        assert not explicit_result.safe
        assert any("activation dtype" in v for v in explicit_result.violations)
        assert any("activation device" in v for v in explicit_result.violations)

    def test_checkpoint_recompute_contracts_are_checked(self):
        boundary = PipelineBoundarySpec(
            name="checkpointed_block",
            producer_stage=0,
            consumer_stage=1,
            activation_shape=(8, 128),
            checkpoint_boundary=True,
            recompute_shape=(8, 64),
            activation_dtype="bf16",
            recompute_dtype="fp32",
            activation_device="cuda:0",
            recompute_device="cuda:1",
        )
        result = verify_pipeline_boundaries([boundary])
        assert not result.safe
        assert any("checkpoint recompute_shape" in v for v in result.violations)
        assert any("checkpoint recompute dtype" in v for v in result.violations)
        assert any("checkpoint recompute device" in v for v in result.violations)

    def test_symbolic_dims_and_interleaved_schedule_warn_not_refute(self):
        boundary = PipelineBoundarySpec(
            name="symbolic_virtual_stage",
            producer_stage=2,
            consumer_stage=1,
            activation_shape=("microbatch", "hidden"),
            expected_input_shape=("microbatch", 128),
            microbatches=4,
            global_batch_size=32,
        )
        result = verify_pipeline_boundaries([boundary])
        assert result.safe
        assert any("symbolic dimensions" in w for w in result.warnings)
        assert any("interleaved" in w for w in result.warnings)
        assert any("symbolic microbatch extent" in w for w in result.warnings)

    def test_stage_specs_refute_duplicate_missing_and_negative_shape(self):
        stages = [
            PipelineStageSpec(index=0, output_shape=(8, 128)),
            PipelineStageSpec(index=0, input_shape=(8, -1)),
        ]
        boundary = PipelineBoundarySpec(
            name="missing_consumer",
            producer_stage=0,
            consumer_stage=2,
            activation_shape=(8, 128),
        )
        result = PipelineParallelVerifier(stages).verify_boundaries([boundary])
        assert not result.safe
        assert any("duplicate stage index" in v for v in result.violations)
        assert any("negative" in v for v in result.violations)
        assert any("consumer_stage 2 is not declared" in v for v in result.violations)

    def test_real_torch_split_stage_shape_oracle_when_available(self):
        torch = pytest.importorskip("torch")
        nn = torch.nn

        stage0 = nn.Sequential(nn.Linear(16, 32), nn.GELU())
        stage1 = nn.Sequential(nn.Linear(32, 4))
        x = torch.randn(8, 16)
        activation = stage0(x)
        output = stage1(activation)
        assert tuple(output.shape) == (8, 4)

        boundary = PipelineBoundarySpec(
            name="real_stage0_to_stage1",
            producer_stage=0,
            consumer_stage=1,
            activation_shape=tuple(activation.shape),
            expected_input_shape=(8, 32),
            microbatches=4,
            global_batch_size=32,
            activation_dtype=str(activation.dtype),
            expected_dtype="torch.float32",
            activation_device=str(activation.device),
            expected_device="cpu",
            checkpoint_boundary=True,
            recompute_shape=tuple(stage0(x).shape),
            recompute_dtype=str(stage0(x).dtype),
            recompute_device=str(stage0(x).device),
        )
        ok = verify_pipeline_boundaries([boundary])
        assert ok.safe

        broken = verify_pipeline_boundaries([
            PipelineBoundarySpec(
                name="real_bad_hidden",
                producer_stage=0,
                consumer_stage=1,
                activation_shape=tuple(activation.shape),
                expected_input_shape=(8, 31),
            )
        ])
        assert not broken.safe

    def test_public_package_exports_pipeline_helpers(self):
        import tensorguard

        assert tensorguard.PipelineBoundarySpec is PipelineBoundarySpec
        assert tensorguard.PipelineStageSpec is PipelineStageSpec
        assert tensorguard.PipelineParallelVerifier is PipelineParallelVerifier
        assert tensorguard.verify_pipeline_boundaries is verify_pipeline_boundaries


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
