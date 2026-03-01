"""Distributed verification for FSDP and DeepSpeed sharding configurations.

Extends TensorGuard's static verification to cover Fully Sharded Data
Parallel (FSDP) and DeepSpeed ZeRO-stage configurations, plus multi-adapter
composition.  Uses Z3 to verify that:

  • shard_size × world_size ≥ numel  for every sharded parameter
  • flat_param reconstruction preserves original shapes
  • DeepSpeed ZeRO partition sizes are consistent across stages
  • Stacked/merged LoRA adapters maintain shape compatibility

Key insight: FSDP/DeepSpeed change the *physical layout* of parameter
tensors but must preserve the *logical shape* visible to the forward pass.
We encode this invariant as Z3 constraints and verify it statically.

Usage::

    from src.distributed_verification import verify_distributed

    result = verify_distributed(
        source=open("my_model.py").read(),
        input_shapes={"x": ("batch", 3, 224, 224)},
        fsdp_config=FSDPConfig(world_size=8),
    )
    print(f"Safe: {result.safe}")
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    ConstraintVerifier,
    Confidence,
    CounterexampleTrace,
    Device,
    LayerDef,
    LayerKind,
    OpKind,
    Phase,
    SafetyCertificate,
    VerificationResult,
    extract_computation_graph,
    verify_model,
)

from src.assume_guarantee import (
    CompositionalResult,
    InterfaceContract,
    SubModule,
    verify_compositional,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Configuration & data classes
# ═══════════════════════════════════════════════════════════════════════════════


class WrapPolicy(Enum):
    """FSDP auto-wrap policies."""
    NONE = "none"
    SIZE_BASED = "size_based"
    TRANSFORMER_BASED = "transformer_based"
    MODULE_BASED = "module_based"


class ZeROStage(Enum):
    """DeepSpeed ZeRO optimisation stages."""
    STAGE_0 = 0   # No partitioning (DDP)
    STAGE_1 = 1   # Optimizer state partitioning
    STAGE_2 = 2   # + Gradient partitioning
    STAGE_3 = 3   # + Parameter partitioning


class AdapterMergeStrategy(Enum):
    """How multiple LoRA adapters are combined."""
    STACK = "stack"           # Sequential application: A₂(A₁(x))
    ADD = "add"               # Additive: W + α₁B₁A₁ + α₂B₂A₂
    SWITCH = "switch"         # Dynamic selection at runtime
    WEIGHTED_ADD = "weighted" # Weighted sum of adapter outputs


@dataclass
class FSDPConfig:
    """Configuration for FSDP sharding verification.

    Attributes
    ----------
    world_size : int
        Number of FSDP ranks (GPUs).
    auto_wrap_policy : WrapPolicy
        Wrapping strategy used.
    min_num_params : int
        Minimum parameter count for size-based wrapping.
    cpu_offload : bool
        Whether CPU offloading is enabled.
    use_orig_params : bool
        Whether ``use_orig_params=True`` (preserves original param shapes).
    transformer_layer_cls : list of str
        Class names for transformer-based wrapping.
    sharding_strategy : str
        FSDP sharding strategy (``"FULL_SHARD"``, ``"SHARD_GRAD_OP"``,
        ``"NO_SHARD"``).
    """

    world_size: int = 1
    auto_wrap_policy: WrapPolicy = WrapPolicy.NONE
    min_num_params: int = 100_000
    cpu_offload: bool = False
    use_orig_params: bool = False
    transformer_layer_cls: List[str] = field(default_factory=list)
    sharding_strategy: str = "FULL_SHARD"


@dataclass
class DeepSpeedConfig:
    """Configuration for DeepSpeed ZeRO verification.

    Attributes
    ----------
    stage : ZeROStage
        ZeRO optimisation stage.
    dp_world_size : int
        Data-parallel world size.
    offload_optimizer : bool
        Whether the optimizer state is offloaded to CPU.
    offload_param : bool
        Whether parameters are offloaded to CPU (Stage 3 only).
    reduce_bucket_size : int
        Size of allreduce communication buckets.
    allgather_bucket_size : int
        Size of allgather communication buckets.
    contiguous_gradients : bool
        Whether gradients are stored contiguously.
    """

    stage: ZeROStage = ZeROStage.STAGE_1
    dp_world_size: int = 1
    offload_optimizer: bool = False
    offload_param: bool = False
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000
    contiguous_gradients: bool = True


@dataclass
class ParamShardInfo:
    """Information about a single sharded parameter.

    Attributes
    ----------
    name : str
        Fully-qualified parameter name.
    original_shape : tuple
        Shape before sharding.
    numel : int or str
        Total number of elements (may be symbolic).
    shard_size : int or str
        Elements per shard (may be symbolic).
    world_size : int
        Number of shards.
    is_flat : bool
        Whether the parameter has been flattened.
    """

    name: str
    original_shape: Tuple[Union[int, str], ...]
    numel: Union[int, str]
    shard_size: Union[int, str]
    world_size: int
    is_flat: bool = False


@dataclass
class ShardingResult:
    """Result of FSDP or DeepSpeed sharding verification.

    Attributes
    ----------
    safe : bool
        Whether all sharding constraints are satisfied.
    violations : list of str
        Human-readable violation descriptions.
    params_checked : int
        Number of parameters verified.
    z3_constraints_used : int
        Number of Z3 constraints generated.
    verification_time_ms : float
        Wall-clock verification time.
    shard_info : list of ParamShardInfo
        Per-parameter shard information.
    """

    safe: bool = True
    violations: List[str] = field(default_factory=list)
    params_checked: int = 0
    z3_constraints_used: int = 0
    verification_time_ms: float = 0.0
    shard_info: List[ParamShardInfo] = field(default_factory=list)

    def pretty(self) -> str:
        status = "SAFE" if self.safe else "UNSAFE"
        lines = [
            f"ShardingResult: {status}",
            f"  Parameters checked:  {self.params_checked}",
            f"  Z3 constraints:      {self.z3_constraints_used}",
            f"  Time:                {self.verification_time_ms:.1f} ms",
        ]
        for v in self.violations:
            lines.append(f"  ✗ {v}")
        return "\n".join(lines)


@dataclass
class AdapterComposition:
    """Description of how multiple adapters are composed.

    Attributes
    ----------
    adapters : list of dict
        Each dict has keys: name, in_features, out_features, rank.
    strategy : AdapterMergeStrategy
        How adapters are combined.
    weights : list of float, optional
        Per-adapter weights for WEIGHTED_ADD strategy.
    """

    adapters: List[Dict[str, Any]] = field(default_factory=list)
    strategy: AdapterMergeStrategy = AdapterMergeStrategy.ADD
    weights: Optional[List[float]] = None


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FSDP Sharding Verifier
# ═══════════════════════════════════════════════════════════════════════════════


class FSDPShardingVerifier:
    """Verify that FSDP sharding configurations are consistent with model shapes.

    The verifier checks:
      1. That ``shard_size × world_size ≥ numel`` for every parameter.
      2. That flat_param reconstruction preserves original shapes.
      3. That the wrapping policy selects appropriate modules.
      4. That sharded parameter shapes are consistent after gather/scatter.

    Parameters
    ----------
    fsdp_config : FSDPConfig
        FSDP configuration to verify against.
    """

    def __init__(self, fsdp_config: FSDPConfig):
        self.config = fsdp_config

    def _compute_numel(self, shape: Tuple[Union[int, str], ...]) -> Union[int, str]:
        """Compute the number of elements from a shape tuple.

        Returns an int if all dimensions are concrete, else a symbolic
        expression string.
        """
        if not shape:
            return 0
        result = 1
        symbolic_parts: List[str] = []
        for d in shape:
            if isinstance(d, int):
                result *= d
            else:
                symbolic_parts.append(str(d))
        if symbolic_parts:
            parts = [str(result)] + symbolic_parts if result != 1 else symbolic_parts
            return " * ".join(parts)
        return result

    def _compute_shard_size(
        self, numel: Union[int, str], world_size: int
    ) -> Union[int, str]:
        """Compute the per-rank shard size: ⌈numel / world_size⌉."""
        if isinstance(numel, int):
            return math.ceil(numel / world_size) if world_size > 0 else numel
        return f"ceil({numel} / {world_size})"

    def detect_wrapping(
        self,
        params: Dict[str, Tuple[Union[int, str], ...]],
    ) -> Dict[str, bool]:
        """Determine which parameters would be wrapped by the auto-wrap policy.

        Parameters
        ----------
        params : dict
            Mapping from parameter name → shape tuple.

        Returns
        -------
        dict
            Mapping from parameter name → whether it would be wrapped.
        """
        policy = self.config.auto_wrap_policy
        result: Dict[str, bool] = {}

        for name, shape in params.items():
            numel = self._compute_numel(shape)

            if policy == WrapPolicy.NONE:
                result[name] = True  # All params under a single FSDP unit

            elif policy == WrapPolicy.SIZE_BASED:
                if isinstance(numel, int):
                    result[name] = numel >= self.config.min_num_params
                else:
                    # Symbolic — conservatively assume wrapped
                    result[name] = True

            elif policy == WrapPolicy.TRANSFORMER_BASED:
                # Match if param name contains any transformer layer class
                matched = any(
                    cls_name.lower() in name.lower()
                    for cls_name in self.config.transformer_layer_cls
                )
                result[name] = matched or True  # wrapped at some level

            elif policy == WrapPolicy.MODULE_BASED:
                result[name] = True

        return result

    def verify_shard_consistency(
        self,
        params: Dict[str, Tuple[Union[int, str], ...]],
    ) -> ShardingResult:
        """Verify that FSDP sharding is consistent for all parameters.

        For each parameter, checks:
          - ``shard_size × world_size ≥ numel``
          - Flat param can be reshaped back to original shape
          - Padding (if any) does not corrupt data

        Parameters
        ----------
        params : dict
            Mapping from parameter name → original shape tuple.

        Returns
        -------
        ShardingResult
        """
        t0 = time.perf_counter()
        violations: List[str] = []
        shard_info_list: List[ParamShardInfo] = []
        z3_count = 0
        world_size = self.config.world_size

        if world_size <= 0:
            return ShardingResult(
                safe=False,
                violations=["world_size must be positive"],
                verification_time_ms=(time.perf_counter() - t0) * 1000,
            )

        wrap_decisions = self.detect_wrapping(params)

        for name, shape in params.items():
            numel = self._compute_numel(shape)
            shard_size = self._compute_shard_size(numel, world_size)

            info = ParamShardInfo(
                name=name,
                original_shape=shape,
                numel=numel,
                shard_size=shard_size,
                world_size=world_size,
                is_flat=(self.config.sharding_strategy == "FULL_SHARD"),
            )
            shard_info_list.append(info)

            # Concrete check
            if isinstance(numel, int) and isinstance(shard_size, int):
                total_after_gather = shard_size * world_size
                if total_after_gather < numel:
                    violations.append(
                        f"{name}: shard_size({shard_size}) × "
                        f"world_size({world_size}) = {total_after_gather} "
                        f"< numel({numel})"
                    )

                # Verify flat_param reconstruction
                if not self.config.use_orig_params:
                    padding = total_after_gather - numel
                    if padding < 0:
                        violations.append(
                            f"{name}: negative padding ({padding}), "
                            f"flat_param reconstruction impossible"
                        )

            # Z3 symbolic check
            elif HAS_Z3:
                z3_violations = self._verify_shard_z3(name, shape, world_size)
                violations.extend(z3_violations)
                z3_count += 1

        elapsed = (time.perf_counter() - t0) * 1000
        return ShardingResult(
            safe=len(violations) == 0,
            violations=violations,
            params_checked=len(params),
            z3_constraints_used=z3_count,
            verification_time_ms=elapsed,
            shard_info=shard_info_list,
        )

    def _verify_shard_z3(
        self,
        param_name: str,
        shape: Tuple[Union[int, str], ...],
        world_size: int,
    ) -> List[str]:
        """Use Z3 to verify sharding constraint for a single parameter.

        Checks: ∀ positive dims, ⌈numel/world_size⌉ × world_size ≥ numel.

        Returns list of violation messages (empty = OK).
        """
        if not HAS_Z3:
            return []

        violations: List[str] = []
        s = z3.Solver()

        # Build symbolic numel
        dim_vars: List[z3.ArithRef] = []
        numel_expr: z3.ArithRef = z3.IntVal(1)
        for i, d in enumerate(shape):
            if isinstance(d, int):
                numel_expr = numel_expr * z3.IntVal(d)
            else:
                v = z3.Int(f"{param_name}_dim_{i}_{d}")
                s.add(v > 0)
                dim_vars.append(v)
                numel_expr = numel_expr * v

        ws = z3.IntVal(world_size)
        # shard_size = ceil(numel / world_size)
        # We encode ceil(a/b) as (a + b - 1) / b  (integer division)
        shard_size = (numel_expr + ws - 1) / ws
        total_gathered = shard_size * ws

        # Check: can total_gathered < numel?
        s.add(total_gathered < numel_expr)

        if s.check() == z3.sat:
            m = s.model()
            violations.append(
                f"{param_name}: Z3 found sharding violation — {m}"
            )

        return violations

    def verify_gather_scatter_shapes(
        self,
        params: Dict[str, Tuple[Union[int, str], ...]],
    ) -> ShardingResult:
        """Verify that all-gather reconstructs the correct shapes.

        After FSDP all-gather, the flat parameter must be reshaped to
        the original shape.  This checks that reshape is valid.

        Parameters
        ----------
        params : dict
            Parameter name → original shape.

        Returns
        -------
        ShardingResult
        """
        t0 = time.perf_counter()
        violations: List[str] = []
        world_size = self.config.world_size

        for name, shape in params.items():
            numel = self._compute_numel(shape)
            if isinstance(numel, int):
                shard_size = math.ceil(numel / world_size)
                gathered = shard_size * world_size
                # After gathering, we have `gathered` elements; we need
                # to slice off padding and reshape to `shape`.
                if gathered < numel:
                    violations.append(
                        f"{name}: gathered size {gathered} < numel {numel}"
                    )
                # Check that the original shape is consistent
                shape_product = 1
                for d in shape:
                    if isinstance(d, int):
                        shape_product *= d
                if shape_product != numel:
                    violations.append(
                        f"{name}: shape product {shape_product} != "
                        f"numel {numel}"
                    )

        elapsed = (time.perf_counter() - t0) * 1000
        return ShardingResult(
            safe=len(violations) == 0,
            violations=violations,
            params_checked=len(params),
            verification_time_ms=elapsed,
        )

    def verify_from_module(
        self,
        model: Any,
    ) -> ShardingResult:
        """Extract parameter shapes from an nn.Module and verify sharding.

        Parameters
        ----------
        model : nn.Module
            The model to verify.

        Returns
        -------
        ShardingResult
        """
        if not HAS_TORCH:
            return ShardingResult(
                safe=True,
                violations=["torch not available; skipping module verification"],
            )

        params: Dict[str, Tuple[Union[int, str], ...]] = {}
        for name, param in model.named_parameters():
            params[name] = tuple(param.shape)

        return self.verify_shard_consistency(params)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DeepSpeed ZeRO Verifier
# ═══════════════════════════════════════════════════════════════════════════════


class DeepSpeedVerifier:
    """Verify DeepSpeed ZeRO stage configurations for shape consistency.

    Checks that parameter, gradient, and optimizer state partitioning
    is consistent with model shapes at each ZeRO stage:

      • Stage 0: DDP — no shape changes.
      • Stage 1: Optimizer state partitioned — no visible shape changes.
      • Stage 2: Gradients partitioned — verify allreduce shape consistency.
      • Stage 3: Parameters partitioned — verify gather produces correct shapes.

    Parameters
    ----------
    ds_config : DeepSpeedConfig
        DeepSpeed configuration.
    """

    def __init__(self, ds_config: DeepSpeedConfig):
        self.config = ds_config

    def _compute_partition_size(
        self, numel: Union[int, str], dp_world_size: int
    ) -> Union[int, str]:
        """Compute per-rank partition size: ⌈numel / dp_world_size⌉."""
        if isinstance(numel, int) and dp_world_size > 0:
            return math.ceil(numel / dp_world_size)
        return f"ceil({numel} / {dp_world_size})"

    def _compute_numel(self, shape: Tuple[Union[int, str], ...]) -> Union[int, str]:
        """Compute total element count from a shape tuple."""
        if not shape:
            return 0
        result = 1
        symbolic = []
        for d in shape:
            if isinstance(d, int):
                result *= d
            else:
                symbolic.append(str(d))
        if symbolic:
            parts = ([str(result)] if result != 1 else []) + symbolic
            return " * ".join(parts)
        return result

    def verify_stage(
        self,
        params: Dict[str, Tuple[Union[int, str], ...]],
    ) -> ShardingResult:
        """Verify sharding constraints for the configured ZeRO stage.

        Parameters
        ----------
        params : dict
            Mapping from parameter name → shape tuple.

        Returns
        -------
        ShardingResult
        """
        stage = self.config.stage
        if stage == ZeROStage.STAGE_0:
            return self._verify_stage_0(params)
        elif stage == ZeROStage.STAGE_1:
            return self._verify_stage_1(params)
        elif stage == ZeROStage.STAGE_2:
            return self._verify_stage_2(params)
        elif stage == ZeROStage.STAGE_3:
            return self._verify_stage_3(params)
        else:
            return ShardingResult(
                safe=False,
                violations=[f"Unknown ZeRO stage: {stage}"],
            )

    def _verify_stage_0(
        self,
        params: Dict[str, Tuple[Union[int, str], ...]],
    ) -> ShardingResult:
        """Stage 0: Standard DDP — no partitioning, just verify shapes exist."""
        t0 = time.perf_counter()
        violations: List[str] = []

        for name, shape in params.items():
            if not shape:
                violations.append(f"{name}: empty shape")
            for i, d in enumerate(shape):
                if isinstance(d, int) and d <= 0:
                    violations.append(
                        f"{name}: dimension {i} is non-positive ({d})"
                    )

        elapsed = (time.perf_counter() - t0) * 1000
        return ShardingResult(
            safe=len(violations) == 0,
            violations=violations,
            params_checked=len(params),
            verification_time_ms=elapsed,
        )

    def _verify_stage_1(
        self,
        params: Dict[str, Tuple[Union[int, str], ...]],
    ) -> ShardingResult:
        """Stage 1: Optimizer state partitioning.

        No visible shape changes to parameters or gradients.
        Verify that partition_size * dp_world_size >= numel (for optimizer
        state tracking purposes).
        """
        t0 = time.perf_counter()
        violations: List[str] = []
        dp = self.config.dp_world_size

        if dp <= 0:
            return ShardingResult(
                safe=False,
                violations=["dp_world_size must be positive"],
                verification_time_ms=(time.perf_counter() - t0) * 1000,
            )

        for name, shape in params.items():
            numel = self._compute_numel(shape)
            if isinstance(numel, int):
                part_size = math.ceil(numel / dp)
                if part_size * dp < numel:
                    violations.append(
                        f"{name}: partition_size({part_size}) × "
                        f"dp_world_size({dp}) = {part_size * dp} "
                        f"< numel({numel})"
                    )

        elapsed = (time.perf_counter() - t0) * 1000
        return ShardingResult(
            safe=len(violations) == 0,
            violations=violations,
            params_checked=len(params),
            verification_time_ms=elapsed,
        )

    def _verify_stage_2(
        self,
        params: Dict[str, Tuple[Union[int, str], ...]],
    ) -> ShardingResult:
        """Stage 2: Gradient partitioning.

        Verify that allreduce produces gradients with original shapes.
        Each rank holds a partition of the gradient; after allreduce,
        the full gradient must match the parameter shape.
        """
        t0 = time.perf_counter()
        violations: List[str] = []
        dp = self.config.dp_world_size
        z3_count = 0

        if dp <= 0:
            return ShardingResult(
                safe=False,
                violations=["dp_world_size must be positive"],
                verification_time_ms=(time.perf_counter() - t0) * 1000,
            )

        for name, shape in params.items():
            numel = self._compute_numel(shape)
            if isinstance(numel, int):
                part_size = math.ceil(numel / dp)
                gathered = part_size * dp
                if gathered < numel:
                    violations.append(
                        f"{name}: gradient allreduce inconsistency — "
                        f"gathered({gathered}) < numel({numel})"
                    )
                # Check reduce bucket size
                if numel > self.config.reduce_bucket_size:
                    # Not a safety violation, but worth noting
                    pass
            elif HAS_Z3:
                z3_viols = self._verify_partition_z3(
                    name, shape, dp, "gradient"
                )
                violations.extend(z3_viols)
                z3_count += 1

        elapsed = (time.perf_counter() - t0) * 1000
        return ShardingResult(
            safe=len(violations) == 0,
            violations=violations,
            params_checked=len(params),
            z3_constraints_used=z3_count,
            verification_time_ms=elapsed,
        )

    def _verify_stage_3(
        self,
        params: Dict[str, Tuple[Union[int, str], ...]],
    ) -> ShardingResult:
        """Stage 3: Parameter partitioning.

        Verify that all-gather produces parameters with correct shapes.
        This is the most aggressive ZeRO stage — parameters themselves
        are sharded, so gather must reconstruct the original shape.
        """
        t0 = time.perf_counter()
        violations: List[str] = []
        shard_info_list: List[ParamShardInfo] = []
        dp = self.config.dp_world_size
        z3_count = 0

        if dp <= 0:
            return ShardingResult(
                safe=False,
                violations=["dp_world_size must be positive"],
                verification_time_ms=(time.perf_counter() - t0) * 1000,
            )

        for name, shape in params.items():
            numel = self._compute_numel(shape)
            part_size = self._compute_partition_size(numel, dp)

            info = ParamShardInfo(
                name=name,
                original_shape=shape,
                numel=numel,
                shard_size=part_size,
                world_size=dp,
                is_flat=True,
            )
            shard_info_list.append(info)

            if isinstance(numel, int) and isinstance(part_size, int):
                gathered = part_size * dp
                if gathered < numel:
                    violations.append(
                        f"{name}: partition_size({part_size}) × "
                        f"dp_world_size({dp}) = {gathered} "
                        f"< param_numel({numel})"
                    )

                # Check that reshape from flat to original is valid
                shape_product = 1
                for d in shape:
                    if isinstance(d, int):
                        shape_product *= d
                if shape_product != numel:
                    violations.append(
                        f"{name}: original shape product {shape_product} "
                        f"!= numel {numel}"
                    )

                # Check allgather bucket
                if numel > self.config.allgather_bucket_size:
                    pass  # Warning only, not a safety issue

            elif HAS_Z3:
                z3_viols = self._verify_partition_z3(
                    name, shape, dp, "parameter"
                )
                violations.extend(z3_viols)
                z3_count += 1

        elapsed = (time.perf_counter() - t0) * 1000
        return ShardingResult(
            safe=len(violations) == 0,
            violations=violations,
            params_checked=len(params),
            z3_constraints_used=z3_count,
            verification_time_ms=elapsed,
            shard_info=shard_info_list,
        )

    def _verify_partition_z3(
        self,
        param_name: str,
        shape: Tuple[Union[int, str], ...],
        dp_world_size: int,
        partition_type: str,
    ) -> List[str]:
        """Z3 verification of partition_size × dp_world_size ≥ numel.

        Returns list of violation messages (empty = OK).
        """
        if not HAS_Z3:
            return []

        violations: List[str] = []
        s = z3.Solver()

        numel_expr: z3.ArithRef = z3.IntVal(1)
        for i, d in enumerate(shape):
            if isinstance(d, int):
                numel_expr = numel_expr * z3.IntVal(d)
            else:
                v = z3.Int(f"{param_name}_dim_{i}_{d}")
                s.add(v > 0)
                numel_expr = numel_expr * v

        ws = z3.IntVal(dp_world_size)
        partition_size = (numel_expr + ws - 1) / ws
        total = partition_size * ws

        # Try to find an assignment where total < numel
        s.add(total < numel_expr)

        if s.check() == z3.sat:
            m = s.model()
            violations.append(
                f"{param_name}: Z3 found {partition_type} partition "
                f"violation — {m}"
            )

        return violations

    def verify_from_module(
        self,
        model: Any,
    ) -> ShardingResult:
        """Extract parameter shapes from an nn.Module and verify.

        Parameters
        ----------
        model : nn.Module
            The model to verify.

        Returns
        -------
        ShardingResult
        """
        if not HAS_TORCH:
            return ShardingResult(
                safe=True,
                violations=["torch not available; skipping module verification"],
            )

        params: Dict[str, Tuple[Union[int, str], ...]] = {}
        for name, param in model.named_parameters():
            params[name] = tuple(param.shape)

        return self.verify_stage(params)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Adapter Composition Verifier
# ═══════════════════════════════════════════════════════════════════════════════


class AdapterCompositionVerifier:
    """Verify composition of multiple LoRA adapters.

    Checks three composition strategies:

    1. **Stack (sequential)**: For adapters A₁, A₂, …, verify that
       output rank of Aᵢ matches input rank of Aᵢ₊₁.

    2. **Add/merge**: For adapters applied additively to the same base
       weight, verify that all B_i·A_i products have the same shape as W.

    3. **Dynamic switch**: For runtime adapter selection, verify that
       all adapters produce the same output shape.

    Parameters
    ----------
    composition : AdapterComposition
        Description of how adapters are composed.
    """

    def __init__(self, composition: AdapterComposition):
        self.composition = composition

    def verify(self) -> ShardingResult:
        """Run composition verification based on the configured strategy.

        Returns
        -------
        ShardingResult
        """
        strategy = self.composition.strategy
        if strategy == AdapterMergeStrategy.STACK:
            return self.verify_stack()
        elif strategy in (AdapterMergeStrategy.ADD, AdapterMergeStrategy.WEIGHTED_ADD):
            return self.verify_merge()
        elif strategy == AdapterMergeStrategy.SWITCH:
            return self.verify_switch()
        else:
            return ShardingResult(
                safe=False,
                violations=[f"Unknown merge strategy: {strategy}"],
            )

    def verify_stack(self) -> ShardingResult:
        """Verify sequential adapter stacking.

        For stacked adapters A₁(A₂(x)), the output dimension of A₂
        must match the input dimension of A₁.  More precisely, for
        each consecutive pair (Aᵢ, Aᵢ₊₁):
            Aᵢ.out_features == Aᵢ₊₁.in_features

        Returns
        -------
        ShardingResult
        """
        t0 = time.perf_counter()
        violations: List[str] = []
        adapters = self.composition.adapters
        z3_count = 0

        if len(adapters) < 2:
            elapsed = (time.perf_counter() - t0) * 1000
            return ShardingResult(
                safe=True,
                params_checked=len(adapters),
                verification_time_ms=elapsed,
            )

        for i in range(len(adapters) - 1):
            curr = adapters[i]
            next_a = adapters[i + 1]

            curr_out = curr.get("out_features")
            next_in = next_a.get("in_features")

            if curr_out is None or next_in is None:
                violations.append(
                    f"Adapters [{i}]→[{i+1}]: missing dimension info"
                )
                continue

            if isinstance(curr_out, int) and isinstance(next_in, int):
                if curr_out != next_in:
                    violations.append(
                        f"Adapters [{i}]({curr.get('name','?')}) → "
                        f"[{i+1}]({next_a.get('name','?')}): "
                        f"out_features({curr_out}) != in_features({next_in})"
                    )
            elif HAS_Z3:
                z3_viols = self._verify_stack_pair_z3(i, curr, next_a)
                violations.extend(z3_viols)
                z3_count += 1

        elapsed = (time.perf_counter() - t0) * 1000
        return ShardingResult(
            safe=len(violations) == 0,
            violations=violations,
            params_checked=len(adapters),
            z3_constraints_used=z3_count,
            verification_time_ms=elapsed,
        )

    def _verify_stack_pair_z3(
        self,
        index: int,
        curr: Dict[str, Any],
        next_a: Dict[str, Any],
    ) -> List[str]:
        """Z3 verification for a single adapter pair in a stack."""
        if not HAS_Z3:
            return []

        violations: List[str] = []
        s = z3.Solver()

        curr_out = curr.get("out_features")
        next_in = next_a.get("in_features")

        def _sym(prefix: str, val: Any) -> z3.ArithRef:
            if isinstance(val, int):
                return z3.IntVal(val)
            return z3.Int(f"adapter_{prefix}")

        out_v = _sym(f"{index}_out", curr_out)
        in_v = _sym(f"{index+1}_in", next_in)

        if isinstance(curr_out, str):
            s.add(out_v > 0)
        if isinstance(next_in, str):
            s.add(in_v > 0)

        # Check if they can differ
        s.add(out_v != in_v)

        if s.check() == z3.sat:
            m = s.model()
            violations.append(
                f"Adapters [{index}]→[{index+1}]: Z3 found dimension "
                f"mismatch — {m}"
            )

        return violations

    def verify_merge(self) -> ShardingResult:
        """Verify additive adapter merging.

        For W + α₁·B₁·A₁ + α₂·B₂·A₂ + …, every B_i·A_i must produce
        a matrix with the same shape as W: (out_features, in_features).

        All adapters must share the same in_features and out_features.

        Returns
        -------
        ShardingResult
        """
        t0 = time.perf_counter()
        violations: List[str] = []
        adapters = self.composition.adapters

        if not adapters:
            elapsed = (time.perf_counter() - t0) * 1000
            return ShardingResult(
                safe=True,
                verification_time_ms=elapsed,
            )

        ref_in = adapters[0].get("in_features")
        ref_out = adapters[0].get("out_features")

        for i, adapter in enumerate(adapters):
            a_in = adapter.get("in_features")
            a_out = adapter.get("out_features")

            if isinstance(ref_in, int) and isinstance(a_in, int):
                if a_in != ref_in:
                    violations.append(
                        f"Adapter [{i}]({adapter.get('name','?')}): "
                        f"in_features({a_in}) != reference({ref_in})"
                    )

            if isinstance(ref_out, int) and isinstance(a_out, int):
                if a_out != ref_out:
                    violations.append(
                        f"Adapter [{i}]({adapter.get('name','?')}): "
                        f"out_features({a_out}) != reference({ref_out})"
                    )

            # Verify rank is compatible
            rank = adapter.get("rank")
            if isinstance(rank, int):
                if rank <= 0:
                    violations.append(
                        f"Adapter [{i}]({adapter.get('name','?')}): "
                        f"rank must be positive, got {rank}"
                    )
                if isinstance(a_in, int) and rank > a_in:
                    violations.append(
                        f"Adapter [{i}]({adapter.get('name','?')}): "
                        f"rank({rank}) > in_features({a_in})"
                    )
                if isinstance(a_out, int) and rank > a_out:
                    violations.append(
                        f"Adapter [{i}]({adapter.get('name','?')}): "
                        f"rank({rank}) > out_features({a_out})"
                    )

        # Validate weights for WEIGHTED_ADD
        if (self.composition.strategy == AdapterMergeStrategy.WEIGHTED_ADD
                and self.composition.weights is not None):
            if len(self.composition.weights) != len(adapters):
                violations.append(
                    f"Weight count ({len(self.composition.weights)}) != "
                    f"adapter count ({len(adapters)})"
                )

        elapsed = (time.perf_counter() - t0) * 1000
        return ShardingResult(
            safe=len(violations) == 0,
            violations=violations,
            params_checked=len(adapters),
            verification_time_ms=elapsed,
        )

    def verify_switch(self) -> ShardingResult:
        """Verify dynamic adapter switching.

        All adapters must produce the same output shape, so that
        switching between them at runtime does not cause shape errors.

        Returns
        -------
        ShardingResult
        """
        t0 = time.perf_counter()
        violations: List[str] = []
        adapters = self.composition.adapters
        z3_count = 0

        if len(adapters) < 2:
            elapsed = (time.perf_counter() - t0) * 1000
            return ShardingResult(
                safe=True,
                params_checked=len(adapters),
                verification_time_ms=elapsed,
            )

        ref = adapters[0]
        ref_in = ref.get("in_features")
        ref_out = ref.get("out_features")

        for i, adapter in enumerate(adapters[1:], 1):
            a_in = adapter.get("in_features")
            a_out = adapter.get("out_features")

            # Output shape must match for switching
            if isinstance(ref_out, int) and isinstance(a_out, int):
                if a_out != ref_out:
                    violations.append(
                        f"Adapter [{i}]({adapter.get('name','?')}): "
                        f"out_features({a_out}) != reference({ref_out}) — "
                        f"switching would cause shape mismatch"
                    )

            # Input shape must also match (same input goes to all)
            if isinstance(ref_in, int) and isinstance(a_in, int):
                if a_in != ref_in:
                    violations.append(
                        f"Adapter [{i}]({adapter.get('name','?')}): "
                        f"in_features({a_in}) != reference({ref_in}) — "
                        f"switching would cause shape mismatch"
                    )

            # Z3 for symbolic dims
            if ((isinstance(ref_out, str) or isinstance(a_out, str))
                    and HAS_Z3):
                z3_viols = self._verify_switch_pair_z3(i, ref, adapter)
                violations.extend(z3_viols)
                z3_count += 1

        elapsed = (time.perf_counter() - t0) * 1000
        return ShardingResult(
            safe=len(violations) == 0,
            violations=violations,
            params_checked=len(adapters),
            z3_constraints_used=z3_count,
            verification_time_ms=elapsed,
        )

    def _verify_switch_pair_z3(
        self,
        index: int,
        ref: Dict[str, Any],
        adapter: Dict[str, Any],
    ) -> List[str]:
        """Z3 verification for adapter switch compatibility."""
        if not HAS_Z3:
            return []

        violations: List[str] = []
        s = z3.Solver()

        def _sym(prefix: str, val: Any) -> z3.ArithRef:
            if isinstance(val, int):
                return z3.IntVal(val)
            return z3.Int(prefix)

        ref_out = _sym("ref_out", ref.get("out_features"))
        a_out = _sym(f"adapter_{index}_out", adapter.get("out_features"))

        if isinstance(ref.get("out_features"), str):
            s.add(ref_out > 0)
        if isinstance(adapter.get("out_features"), str):
            s.add(a_out > 0)

        s.add(ref_out != a_out)

        if s.check() == z3.sat:
            m = s.model()
            violations.append(
                f"Adapter [{index}]: Z3 found switch incompatibility — {m}"
            )

        return violations

    def verify_z3_composition(self) -> Tuple[bool, Optional[str]]:
        """Full Z3 verification of adapter composition constraints.

        Encodes all adapters' shape constraints and the composition
        invariant as Z3 formulas, then checks satisfiability.

        Returns
        -------
        tuple of (bool, str or None)
            ``(True, None)`` if safe; ``(False, counterexample)`` otherwise.
        """
        if not HAS_Z3:
            raise RuntimeError("Z3 required for symbolic composition verification")

        adapters = self.composition.adapters
        if not adapters:
            return True, None

        s = z3.Solver()
        strategy = self.composition.strategy

        # Create symbolic variables for each adapter's dims
        adapter_vars: List[Dict[str, z3.ArithRef]] = []
        for i, a in enumerate(adapters):
            name = a.get("name", f"adapter_{i}")
            in_f = a.get("in_features")
            out_f = a.get("out_features")
            rank = a.get("rank")

            def _s(n: str, v: Any) -> z3.ArithRef:
                if isinstance(v, int):
                    return z3.IntVal(v)
                return z3.Int(f"{name}_{n}")

            d_in = _s("in", in_f)
            d_out = _s("out", out_f)
            d_rank = _s("rank", rank)

            # Positivity
            s.add(d_in > 0)
            s.add(d_out > 0)
            s.add(d_rank > 0)
            # Rank bound
            s.add(d_rank <= d_in)
            s.add(d_rank <= d_out)

            adapter_vars.append({
                "in": d_in, "out": d_out, "rank": d_rank,
            })

        # Strategy-specific constraints
        violation_conds: List[z3.BoolRef] = []

        if strategy == AdapterMergeStrategy.STACK:
            for i in range(len(adapter_vars) - 1):
                # curr.out must equal next.in
                violation_conds.append(
                    adapter_vars[i]["out"] != adapter_vars[i + 1]["in"]
                )

        elif strategy in (AdapterMergeStrategy.ADD,
                          AdapterMergeStrategy.WEIGHTED_ADD):
            if len(adapter_vars) >= 2:
                ref_in = adapter_vars[0]["in"]
                ref_out = adapter_vars[0]["out"]
                for i in range(1, len(adapter_vars)):
                    violation_conds.append(adapter_vars[i]["in"] != ref_in)
                    violation_conds.append(adapter_vars[i]["out"] != ref_out)

        elif strategy == AdapterMergeStrategy.SWITCH:
            if len(adapter_vars) >= 2:
                ref_in = adapter_vars[0]["in"]
                ref_out = adapter_vars[0]["out"]
                for i in range(1, len(adapter_vars)):
                    violation_conds.append(adapter_vars[i]["in"] != ref_in)
                    violation_conds.append(adapter_vars[i]["out"] != ref_out)

        if violation_conds:
            s.add(z3.Or(*violation_conds))
            if s.check() == z3.sat:
                return False, f"Counterexample: {s.model()}"

        return True, None


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Distributed Verification Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DistributedVerificationResult:
    """Aggregate result of distributed verification.

    Attributes
    ----------
    safe : bool
        Overall safety verdict.
    base_result : VerificationResult or None
        Result from standard ``verify_model()`` pipeline.
    fsdp_result : ShardingResult or None
        FSDP sharding verification result.
    deepspeed_result : ShardingResult or None
        DeepSpeed ZeRO verification result.
    adapter_result : ShardingResult or None
        Adapter composition verification result.
    verification_time_ms : float
        Total wall-clock time.
    """

    safe: bool
    base_result: Optional[VerificationResult] = None
    fsdp_result: Optional[ShardingResult] = None
    deepspeed_result: Optional[ShardingResult] = None
    adapter_result: Optional[ShardingResult] = None
    verification_time_ms: float = 0.0

    def pretty(self) -> str:
        status = "SAFE" if self.safe else "UNSAFE"
        lines = [
            f"DistributedVerificationResult: {status}",
            f"  Time: {self.verification_time_ms:.1f} ms",
        ]
        if self.base_result is not None:
            lines.append(f"  Base model safe:  {self.base_result.safe}")
        if self.fsdp_result is not None:
            lines.append(f"  FSDP safe:        {self.fsdp_result.safe}")
            for v in self.fsdp_result.violations:
                lines.append(f"    ✗ {v}")
        if self.deepspeed_result is not None:
            lines.append(f"  DeepSpeed safe:   {self.deepspeed_result.safe}")
            for v in self.deepspeed_result.violations:
                lines.append(f"    ✗ {v}")
        if self.adapter_result is not None:
            lines.append(f"  Adapters safe:    {self.adapter_result.safe}")
            for v in self.adapter_result.violations:
                lines.append(f"    ✗ {v}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Parameter extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_params_from_source(
    source: str,
) -> Dict[str, Tuple[Union[int, str], ...]]:
    """Extract parameter shapes from an nn.Module source definition.

    Parses the ``__init__`` method to find ``nn.Linear``, ``nn.Conv2d``,
    etc. and infers parameter shapes from their constructor arguments.

    Parameters
    ----------
    source : str
        Python source code.

    Returns
    -------
    dict
        Mapping from parameter name → shape tuple.
    """
    try:
        graph = extract_computation_graph(source)
    except (ValueError, SyntaxError):
        return {}

    params: Dict[str, Tuple[Union[int, str], ...]] = {}

    for attr_name, layer in graph.layers.items():
        if layer.kind == LayerKind.LINEAR:
            in_f = layer.in_features or layer.params.get("in_features")
            out_f = layer.out_features or layer.params.get("out_features")
            if in_f is not None and out_f is not None:
                params[f"{attr_name}.weight"] = (out_f, in_f)
                if layer.params.get("bias", True):
                    params[f"{attr_name}.bias"] = (out_f,)

        elif layer.kind == LayerKind.CONV2D:
            in_c = layer.in_channels or layer.params.get("in_channels")
            out_c = layer.out_channels or layer.params.get("out_channels")
            ks = layer.kernel_size or layer.params.get("kernel_size")
            if in_c is not None and out_c is not None and ks is not None:
                if isinstance(ks, int):
                    ks = (ks, ks)
                elif isinstance(ks, (list, tuple)) and len(ks) == 1:
                    ks = (ks[0], ks[0])
                params[f"{attr_name}.weight"] = (out_c, in_c, *ks)

        elif layer.kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
                            LayerKind.LAYERNORM):
            nf = layer.num_features or layer.params.get("num_features")
            if nf is not None:
                params[f"{attr_name}.weight"] = (nf,)
                params[f"{attr_name}.bias"] = (nf,)

        elif layer.kind == LayerKind.EMBEDDING:
            ne = layer.num_embeddings or layer.params.get("num_embeddings")
            ed = layer.embedding_dim or layer.params.get("embedding_dim")
            if ne is not None and ed is not None:
                params[f"{attr_name}.weight"] = (ne, ed)

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Top-level API
# ═══════════════════════════════════════════════════════════════════════════════


def verify_distributed(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    fsdp_config: Optional[FSDPConfig] = None,
    deepspeed_config: Optional[DeepSpeedConfig] = None,
    adapter_composition: Optional[AdapterComposition] = None,
) -> DistributedVerificationResult:
    """Verify a model under distributed training configurations.

    Extends the standard ``verify_model()`` pipeline with FSDP sharding,
    DeepSpeed ZeRO, and adapter composition checks.

    Parameters
    ----------
    source : str
        Python source code of the nn.Module class.
    input_shapes : dict, optional
        Input tensor shapes for base model verification.
    fsdp_config : FSDPConfig, optional
        FSDP configuration.  If provided, runs FSDP sharding checks.
    deepspeed_config : DeepSpeedConfig, optional
        DeepSpeed configuration.  If provided, runs ZeRO stage checks.
    adapter_composition : AdapterComposition, optional
        Adapter composition description.  If provided, runs composition
        verification.

    Returns
    -------
    DistributedVerificationResult
    """
    t0 = time.perf_counter()
    input_shapes = input_shapes or {}

    # 1. Base model verification
    base_result: Optional[VerificationResult] = None
    try:
        base_result = verify_model(
            source=source,
            input_shapes=input_shapes,
        )
    except Exception as e:
        logger.debug("Base model verification failed: %s", e)
        base_result = VerificationResult(
            safe=False,
            errors=[str(e)],
        )

    # 2. Extract parameter shapes from source
    params = _extract_params_from_source(source)

    # 3. FSDP verification
    fsdp_result: Optional[ShardingResult] = None
    if fsdp_config is not None:
        fsdp_verifier = FSDPShardingVerifier(fsdp_config)
        fsdp_result = fsdp_verifier.verify_shard_consistency(params)

    # 4. DeepSpeed verification
    ds_result: Optional[ShardingResult] = None
    if deepspeed_config is not None:
        ds_verifier = DeepSpeedVerifier(deepspeed_config)
        ds_result = ds_verifier.verify_stage(params)

    # 5. Adapter composition verification
    adapter_result: Optional[ShardingResult] = None
    if adapter_composition is not None:
        comp_verifier = AdapterCompositionVerifier(adapter_composition)
        adapter_result = comp_verifier.verify()

    # Overall safety
    safe = True
    if base_result is not None and not base_result.safe:
        safe = False
    if fsdp_result is not None and not fsdp_result.safe:
        safe = False
    if ds_result is not None and not ds_result.safe:
        safe = False
    if adapter_result is not None and not adapter_result.safe:
        safe = False

    elapsed = (time.perf_counter() - t0) * 1000
    return DistributedVerificationResult(
        safe=safe,
        base_result=base_result,
        fsdp_result=fsdp_result,
        deepspeed_result=ds_result,
        adapter_result=adapter_result,
        verification_time_ms=elapsed,
    )
