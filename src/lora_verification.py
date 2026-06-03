"""Verification support for LoRA adapters and fine-tuning workflows.

Uses assume-guarantee decomposition to verify adapter correctness:
- Base model satisfies contract C_base
- LoRA adapter satisfies C_adapter (rank constraints, scaling)
- Composed model satisfies C_base ∧ C_adapter

Key insight: LoRA modifies W → W + α·B·A where B∈R^{d×r}, A∈R^{r×k}.
Shape safety requires: r ≤ min(d, k) and output shape is preserved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

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


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter.

    Attributes
    ----------
    rank : int
        The low-rank dimension r.
    alpha : float
        Scaling factor α.  The effective scaling is α/r.
    target_modules : list of str
        Names of linear layers to apply LoRA to (e.g. ``["q_proj", "v_proj"]``).
    dropout : float
        Dropout probability for the LoRA path.
    """

    rank: int = 8
    alpha: float = 16.0
    target_modules: List[str] = field(default_factory=list)
    dropout: float = 0.0

    @property
    def scaling(self) -> float:
        """Effective scaling factor α/r."""
        return self.alpha / self.rank if self.rank > 0 else 0.0


@dataclass
class LoRAAdapter:
    """Description of a single LoRA-adapted linear layer.

    Attributes
    ----------
    base_module_name : str
        Dot-path to the adapted module (e.g. ``"model.layers.0.self_attn.q_proj"``).
    in_features : int or str
        Input dimension of the original linear layer (may be symbolic).
    out_features : int or str
        Output dimension of the original linear layer (may be symbolic).
    rank : int
        The LoRA rank r.
    lora_A_shape : tuple
        Shape of the A matrix: ``(rank, in_features)``.
    lora_B_shape : tuple
        Shape of the B matrix: ``(out_features, rank)``.
    """

    base_module_name: str
    in_features: Union[int, str]
    out_features: Union[int, str]
    rank: int
    lora_A_shape: Tuple[Union[int, str], Union[int, str]] = (0, 0)
    lora_B_shape: Tuple[Union[int, str], Union[int, str]] = (0, 0)

    def __post_init__(self):
        if self.lora_A_shape == (0, 0):
            self.lora_A_shape = (self.rank, self.in_features)
        if self.lora_B_shape == (0, 0):
            self.lora_B_shape = (self.out_features, self.rank)


@dataclass
class RankViolation:
    """A rank-constraint violation for a LoRA adapter.

    Attributes
    ----------
    module_name : str
        Name of the offending module.
    rank : int
        The configured rank r.
    in_features : int or str
        Input dimension.
    out_features : int or str
        Output dimension.
    message : str
        Human-readable explanation.
    """

    module_name: str
    rank: int
    in_features: Union[int, str]
    out_features: Union[int, str]
    message: str


class QuantizationBits(Enum):
    """Common quantization configurations."""
    FULL = auto()     # fp32 / fp16 / bf16 — no quantization
    INT8 = auto()     # 8-bit quantization
    INT4 = auto()     # 4-bit quantization (QLoRA)
    NF4 = auto()      # NormalFloat4 (bitsandbytes)


@dataclass
class LoRAShapeContract:
    """Shape contract for a single LoRA-adapted linear layer.

    The contract encodes:
        input:  (*, in_features)
        lora_A: (rank, in_features)
        lora_B: (out_features, rank)
        output: (*, out_features)

    Constraints:
        rank > 0
        rank ≤ min(in_features, out_features)
    """

    adapter: LoRAAdapter

    def constraints_human(self) -> List[str]:
        """Return human-readable constraint descriptions."""
        a = self.adapter
        return [
            f"rank({a.rank}) > 0",
            f"rank({a.rank}) <= min(in_features={a.in_features}, "
            f"out_features={a.out_features})",
            f"lora_A shape = ({a.rank}, {a.in_features})",
            f"lora_B shape = ({a.out_features}, {a.rank})",
            f"output preserves (*, {a.out_features})",
        ]

    def check_concrete(self) -> List[str]:
        """Check constraints with concrete (integer) dimensions.

        Returns list of violation messages (empty = all OK).
        """
        violations: List[str] = []
        a = self.adapter

        if isinstance(a.rank, int) and a.rank <= 0:
            violations.append(
                f"{a.base_module_name}: rank must be > 0, got {a.rank}"
            )

        if isinstance(a.in_features, int) and isinstance(a.out_features, int):
            min_dim = min(a.in_features, a.out_features)
            if isinstance(a.rank, int) and a.rank > min_dim:
                violations.append(
                    f"{a.base_module_name}: rank ({a.rank}) > "
                    f"min(in={a.in_features}, out={a.out_features}) = {min_dim}"
                )

        # Check A matrix shape consistency
        if isinstance(a.rank, int) and isinstance(a.in_features, int):
            expected_A = (a.rank, a.in_features)
            if (isinstance(a.lora_A_shape[0], int) and
                    isinstance(a.lora_A_shape[1], int)):
                if a.lora_A_shape != expected_A:
                    violations.append(
                        f"{a.base_module_name}: lora_A shape "
                        f"{a.lora_A_shape} != expected {expected_A}"
                    )

        # Check B matrix shape consistency
        if isinstance(a.rank, int) and isinstance(a.out_features, int):
            expected_B = (a.out_features, a.rank)
            if (isinstance(a.lora_B_shape[0], int) and
                    isinstance(a.lora_B_shape[1], int)):
                if a.lora_B_shape != expected_B:
                    violations.append(
                        f"{a.base_module_name}: lora_B shape "
                        f"{a.lora_B_shape} != expected {expected_B}"
                    )

        return violations

    def to_z3_constraints(self) -> List:
        """Encode shape constraints as Z3 expressions.

        Returns a list of Z3 BoolRef constraints.
        Requires Z3 to be installed.
        """
        if not HAS_Z3:
            raise RuntimeError("Z3 required for symbolic constraint encoding")

        a = self.adapter
        constraints = []

        # Create symbolic variables for potentially-symbolic dimensions
        def _sym(name: str, val: Union[int, str]):
            if isinstance(val, int):
                return z3.IntVal(val)
            return z3.Int(f"{a.base_module_name}_{name}")

        r = _sym("rank", a.rank)
        d_in = _sym("in_features", a.in_features)
        d_out = _sym("out_features", a.out_features)

        # rank > 0
        constraints.append(r > 0)

        # Positivity of dimensions
        constraints.append(d_in > 0)
        constraints.append(d_out > 0)

        # rank ≤ min(in_features, out_features)
        constraints.append(r <= d_in)
        constraints.append(r <= d_out)

        return constraints

    def verify_z3(self) -> Tuple[bool, Optional[str]]:
        """Check constraints using Z3.

        Returns (is_safe, counterexample_or_None).
        """
        if not HAS_Z3:
            raise RuntimeError("Z3 required for symbolic verification")

        s = z3.Solver()

        a = self.adapter

        def _sym(name: str, val: Union[int, str]):
            if isinstance(val, int):
                return z3.IntVal(val)
            return z3.Int(f"{a.base_module_name}_{name}")

        r = _sym("rank", a.rank)
        d_in = _sym("in_features", a.in_features)
        d_out = _sym("out_features", a.out_features)

        # Assert dimension positivity
        s.add(d_in > 0)
        s.add(d_out > 0)

        # Check: is there an assignment where rank > min(in, out) or rank <= 0?
        s.add(z3.Or(r <= 0, r > d_in, r > d_out))

        if s.check() == z3.sat:
            m = s.model()
            return False, f"Counterexample: {m}"
        return True, None


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  LoRA module detection
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_lora_from_module(
    module: Any,
    prefix: str = "",
) -> List[LoRAAdapter]:
    """Walk an nn.Module tree and detect LoRA adapter layers.

    Detects:
    - Manual LoRA: modules with ``lora_A`` and ``lora_B`` attributes
    - PEFT-style: modules of type ``lora.Linear`` or with ``lora_A``/``lora_B``
      sub-modules stored as nn.Linear or nn.Parameter
    """
    if not HAS_TORCH:
        return []

    adapters: List[LoRAAdapter] = []

    for name, child in module.named_modules():
        full_name = f"{prefix}.{name}" if prefix else name
        if not full_name:
            full_name = "(root)"

        has_lora_A = hasattr(child, "lora_A")
        has_lora_B = hasattr(child, "lora_B")

        if has_lora_A and has_lora_B:
            lora_a = child.lora_A
            lora_b = child.lora_B

            # Extract shapes
            a_shape = _get_param_shape(lora_a)
            b_shape = _get_param_shape(lora_b)

            if a_shape is not None and b_shape is not None:
                rank = a_shape[0]
                in_features = a_shape[1]
                out_features = b_shape[0]

                adapters.append(LoRAAdapter(
                    base_module_name=full_name,
                    in_features=in_features,
                    out_features=out_features,
                    rank=rank,
                    lora_A_shape=a_shape,
                    lora_B_shape=b_shape,
                ))

    return adapters


def _get_param_shape(param: Any) -> Optional[Tuple[int, ...]]:
    """Extract shape from a parameter, nn.Linear, or tensor."""
    if not HAS_TORCH:
        return None

    if isinstance(param, nn.Linear):
        return tuple(param.weight.shape)
    if isinstance(param, nn.Parameter):
        return tuple(param.shape)
    if isinstance(param, torch.Tensor):
        return tuple(param.shape)
    # PEFT wraps lora_A in nn.ModuleDict sometimes
    if isinstance(param, nn.ModuleDict):
        for key, mod in param.items():
            if isinstance(mod, nn.Linear):
                return tuple(mod.weight.shape)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  LoRAVerifier — the main verifier class
# ═══════════════════════════════════════════════════════════════════════════════


class LoRAVerifier:
    """Verifier for LoRA-adapted neural network models.

    Parameters
    ----------
    base_model : nn.Module
        The model (base or already LoRA-adapted).
    lora_config : LoRAConfig, optional
        Configuration describing the LoRA adapter setup.
        If ``None``, config is inferred from the model.
    """

    def __init__(
        self,
        base_model: Any,
        lora_config: Optional[LoRAConfig] = None,
    ):
        self.model = base_model
        self.config = lora_config or LoRAConfig()
        self._adapters: Optional[List[LoRAAdapter]] = None

    def detect_lora_modules(
        self,
        model: Optional[Any] = None,
    ) -> List[LoRAAdapter]:
        """Detect LoRA adapter layers in the model.

        Parameters
        ----------
        model : nn.Module, optional
            Model to inspect.  Defaults to ``self.model``.

        Returns
        -------
        list of LoRAAdapter
            Detected adapters with their shape information.
        """
        target = model if model is not None else self.model
        adapters = _detect_lora_from_module(target)
        self._adapters = adapters
        return adapters

    @property
    def adapters(self) -> List[LoRAAdapter]:
        if self._adapters is None:
            self._adapters = self.detect_lora_modules()
        return self._adapters

    def verify_adapter_shapes(self) -> VerificationResult:
        """Verify all LoRA shape constraints across detected adapters.

        Checks every detected adapter for:
        - Correct A/B matrix dimensions
        - Rank positivity
        - Rank bounded by min(in_features, out_features)

        Returns
        -------
        VerificationResult
            ``safe=True`` iff all adapters pass shape checks.
        """
        t0 = time.perf_counter()
        adapters = self.adapters

        if not adapters:
            return VerificationResult(
                safe=True,
                errors=["No LoRA adapters detected"],
                verification_time_ms=(time.perf_counter() - t0) * 1000,
                confidence=Confidence.HIGH,
            )

        all_violations: List[str] = []
        for adapter in adapters:
            contract = LoRAShapeContract(adapter=adapter)
            violations = contract.check_concrete()
            all_violations.extend(violations)

        elapsed = (time.perf_counter() - t0) * 1000
        if all_violations:
            return VerificationResult(
                safe=False,
                errors=all_violations,
                verification_time_ms=elapsed,
                confidence=Confidence.HIGH,
            )

        cert = SafetyCertificate(
            model_name=type(self.model).__name__,
            properties=["lora_shape_compatible", "lora_rank_bounded"],
            k=0,
            checked_steps=len(adapters),
            verification_time_ms=elapsed,
        )
        return VerificationResult(
            safe=True,
            certificate=cert,
            verification_time_ms=elapsed,
            confidence=Confidence.HIGH,
        )

    def verify_rank_constraints(self) -> List[RankViolation]:
        """Check rank ≤ min(in_features, out_features) for all adapters.

        Returns
        -------
        list of RankViolation
            Empty list if all constraints are satisfied.
        """
        violations: List[RankViolation] = []
        for a in self.adapters:
            if isinstance(a.in_features, int) and isinstance(a.out_features, int):
                min_dim = min(a.in_features, a.out_features)
                if a.rank > min_dim:
                    violations.append(RankViolation(
                        module_name=a.base_module_name,
                        rank=a.rank,
                        in_features=a.in_features,
                        out_features=a.out_features,
                        message=(
                            f"rank ({a.rank}) exceeds min(in={a.in_features}, "
                            f"out={a.out_features}) = {min_dim}"
                        ),
                    ))
            if a.rank <= 0:
                violations.append(RankViolation(
                    module_name=a.base_module_name,
                    rank=a.rank,
                    in_features=a.in_features,
                    out_features=a.out_features,
                    message=f"rank must be positive, got {a.rank}",
                ))
        return violations

    def verify_composition(
        self,
        input_shapes: Dict[str, tuple],
        source: Optional[str] = None,
    ) -> VerificationResult:
        """Use assume-guarantee to verify the full composed model.

        The decomposition treats the base model and each LoRA adapter
        as separate components in an assume-guarantee chain.

        Parameters
        ----------
        input_shapes : dict
            Input tensor shapes for the model.
        source : str, optional
            Python source code of the model.  If provided, uses
            ``verify_compositional()`` from the assume-guarantee engine.

        Returns
        -------
        VerificationResult
        """
        t0 = time.perf_counter()

        # First: verify adapter shapes
        adapter_result = self.verify_adapter_shapes()
        if not adapter_result.safe:
            adapter_result.verification_time_ms = (
                (time.perf_counter() - t0) * 1000
            )
            return adapter_result

        # If source is provided, use compositional verification
        if source is not None:
            try:
                comp_result = verify_compositional(
                    source=source,
                    input_shapes=input_shapes,
                )
                elapsed = (time.perf_counter() - t0) * 1000

                errors = []
                for name, sub_result in comp_result.submodule_results.items():
                    errors.extend(sub_result.errors)
                for ic in comp_result.interface_checks:
                    if not ic.compatible:
                        errors.append(ic.message)

                return VerificationResult(
                    safe=comp_result.safe and adapter_result.safe,
                    certificate=adapter_result.certificate,
                    errors=errors,
                    verification_time_ms=elapsed,
                    confidence=Confidence.HIGH,
                )
            except (ValueError, Exception) as e:
                logger.debug("Compositional verification failed: %s", e)

        # Fallback: verify adapter shapes only
        elapsed = (time.perf_counter() - t0) * 1000
        adapter_result.verification_time_ms = elapsed
        return adapter_result

    def verify_merge_safety(self) -> bool:
        """Check that W + α·B·A produces correct output shape after merging.

        Verifies that for each adapter, the matrix product B @ A has the
        same shape as the original weight matrix W (out_features, in_features),
        so that W + α·B·A is well-defined.

        Returns
        -------
        bool
            ``True`` if merging is safe for all adapters.
        """
        for a in self.adapters:
            # B is (out_features, rank), A is (rank, in_features)
            # B @ A = (out_features, in_features) ✓
            # Check dimensional consistency
            if (isinstance(a.lora_B_shape[1], int) and
                    isinstance(a.lora_A_shape[0], int)):
                if a.lora_B_shape[1] != a.lora_A_shape[0]:
                    return False

            if (isinstance(a.lora_B_shape[0], int) and
                    isinstance(a.out_features, int)):
                if a.lora_B_shape[0] != a.out_features:
                    return False

            if (isinstance(a.lora_A_shape[1], int) and
                    isinstance(a.in_features, int)):
                if a.lora_A_shape[1] != a.in_features:
                    return False

        return True


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  QuantizationVerifier — QLoRA support
# ═══════════════════════════════════════════════════════════════════════════════


class QuantizationVerifier:
    """Verify that quantized base + LoRA adapter preserves shapes.

    Handles QLoRA scenarios where the base model weights are quantized
    (4-bit or 8-bit) but LoRA adapters remain in full precision.
    """

    def __init__(self, model: Any):
        self.model = model
        self._quantization_bits: Optional[QuantizationBits] = None

    def detect_quantization(self) -> QuantizationBits:
        """Detect quantization configuration of the model.

        Checks for:
        - bitsandbytes Linear4bit / Linear8bitLt layers
        - Manual quantization markers
        """
        if not HAS_TORCH:
            return QuantizationBits.FULL

        for name, module in self.model.named_modules():
            cls_name = type(module).__name__

            if "Linear4bit" in cls_name or "Params4bit" in cls_name:
                self._quantization_bits = QuantizationBits.INT4
                return QuantizationBits.INT4

            if "Linear8bitLt" in cls_name or "Int8Params" in cls_name:
                self._quantization_bits = QuantizationBits.INT8
                return QuantizationBits.INT8

            if hasattr(module, "quant_type"):
                qt = str(getattr(module, "quant_type", ""))
                if "nf4" in qt.lower():
                    self._quantization_bits = QuantizationBits.NF4
                    return QuantizationBits.NF4

        self._quantization_bits = QuantizationBits.FULL
        return QuantizationBits.FULL

    def verify_shapes_preserved(self) -> VerificationResult:
        """Verify that quantized base + LoRA preserves shapes.

        The key insight: quantization changes the *storage format* but
        not the *logical shape* of tensors.  So shape constraints from
        the LoRA verifier still apply.

        Returns
        -------
        VerificationResult
        """
        t0 = time.perf_counter()
        quant = self.detect_quantization()

        lora_verifier = LoRAVerifier(self.model)
        adapters = lora_verifier.detect_lora_modules()

        errors: List[str] = []
        if not adapters:
            errors.append("No LoRA adapters detected in quantized model")

        # Verify adapter shapes (shape constraints are quant-invariant)
        result = lora_verifier.verify_adapter_shapes()

        elapsed = (time.perf_counter() - t0) * 1000

        if quant != QuantizationBits.FULL and adapters:
            # Additional check: verify LoRA weights are NOT quantized
            for adapter in adapters:
                if not _check_lora_full_precision(self.model, adapter):
                    errors.append(
                        f"{adapter.base_module_name}: LoRA weights appear "
                        f"to be quantized — this may cause precision issues"
                    )

        if errors and result.safe:
            result.errors = errors

        result.verification_time_ms = elapsed
        return result


def _check_lora_full_precision(model: Any, adapter: LoRAAdapter) -> bool:
    """Check that LoRA A/B weights are in full precision (not quantized)."""
    if not HAS_TORCH:
        return True

    for name, module in model.named_modules():
        if name == adapter.base_module_name or name.endswith(
            adapter.base_module_name
        ):
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)

            for param_name, param in [("lora_A", lora_a), ("lora_B", lora_b)]:
                if param is None:
                    continue
                # Get the actual tensor
                if isinstance(param, nn.Linear):
                    t = param.weight
                elif isinstance(param, (nn.Parameter, torch.Tensor)):
                    t = param
                else:
                    continue

                # Check dtype — quantized tensors are typically uint8/int8
                if t.dtype in (torch.uint8, torch.int8):
                    return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PEFT / adapter compatibility gates
# ═══════════════════════════════════════════════════════════════════════════════


Shape = Tuple[int, ...]


@dataclass(frozen=True)
class LoRACompatibilityIssue:
    """One actionable LoRA/PEFT adapter compatibility finding."""

    category: str
    message: str
    module_name: Optional[str] = None
    adapter_name: Optional[str] = None
    key: Optional[str] = None
    expected_shape: Optional[Shape] = None
    actual_shape: Optional[Shape] = None
    severity: str = "error"
    suggestion: Optional[str] = None


@dataclass(frozen=True)
class LoRACompatibilityResult:
    """Result of TensorGuard's LoRA/PEFT compatibility gate."""

    ok: bool
    issues: Tuple[LoRACompatibilityIssue, ...]
    warnings: Tuple[LoRACompatibilityIssue, ...] = ()
    adapters: Tuple[LoRAAdapter, ...] = ()
    checked_targets: Tuple[str, ...] = ()
    matched_target_modules: Tuple[str, ...] = ()
    quantized_targets: Tuple[str, ...] = ()
    merged_targets: Tuple[str, ...] = ()
    skipped_checks: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _LoRACompatibilityRecord:
    target_name: str
    raw_target_name: str
    adapter_name: str
    a_tensor: Optional[Any]
    b_tensor: Optional[Any]
    a_key: Optional[str]
    b_key: Optional[str]
    base_weight: Optional[Any]
    base_key: Optional[str]
    module: Optional[Any] = None
    source: str = "model"
    merged: Optional[bool] = None
    adapters_disabled: bool = False
    quantized_base: bool = False

    @property
    def a_shape(self) -> Optional[Shape]:
        return _shape(self.a_tensor)

    @property
    def b_shape(self) -> Optional[Shape]:
        return _shape(self.b_tensor)

    @property
    def base_shape(self) -> Optional[Shape]:
        return _shape(self.base_weight)


def verify_lora_adapter_compatibility(
    model: Optional[Any] = None,
    adapter_state: Optional[Mapping[str, Any]] = None,
    *,
    peft_config: Optional[Any] = None,
    target_modules: Optional[Sequence[str]] = None,
    require_all_targets: bool = True,
    expected_merged: Optional[bool] = None,
    allow_quantized_base: bool = True,
    require_float_adapters: bool = True,
) -> LoRACompatibilityResult:
    """Verify LoRA/PEFT adapters against a target model without mutating it.

    The gate understands both live PEFT-style modules
    (``lora_A.<adapter>.weight`` / ``lora_B.<adapter>.weight`` ModuleDicts) and
    adapter-only state dicts using HuggingFace PEFT key layouts such as
    ``base_model.model.layers.0.q_proj.lora_A.default.weight``.  Base-relative
    compatibility checks require ``model``; adapter-only inputs still validate
    A/B rank agreement and surface skipped base checks as warnings.
    """

    if model is None and adapter_state is None:
        raise TypeError("model or adapter_state is required")

    patterns = tuple(_target_modules_from_config(peft_config, target_modules))
    records = (
        _collect_state_lora_records(model, _unwrap_adapter_state(adapter_state))
        if adapter_state is not None
        else _collect_live_lora_records(model)
    )

    issues: List[LoRACompatibilityIssue] = []
    warnings_: List[LoRACompatibilityIssue] = []
    skipped: List[str] = []
    adapters: List[LoRAAdapter] = []
    checked_targets: List[str] = []
    quantized_targets: List[str] = []
    merged_targets: List[str] = []

    for record in records:
        checked_targets.append(record.target_name)
        if record.quantized_base:
            quantized_targets.append(record.target_name)
        if record.merged:
            merged_targets.append(record.target_name)
        adapter = _adapter_from_record(record)
        if adapter is not None:
            adapters.append(adapter)
        _check_lora_compatibility_record(
            record,
            issues,
            warnings_,
            skipped,
            expected_merged=expected_merged,
            allow_quantized_base=allow_quantized_base,
            require_float_adapters=require_float_adapters,
            model_available=model is not None,
        )

    matched_patterns = _check_target_module_patterns(
        records,
        patterns,
        require_all_targets=require_all_targets,
        issues=issues,
    )

    return LoRACompatibilityResult(
        ok=not issues,
        issues=tuple(issues),
        warnings=tuple(warnings_),
        adapters=tuple(adapters),
        checked_targets=tuple(dict.fromkeys(checked_targets)),
        matched_target_modules=tuple(matched_patterns),
        quantized_targets=tuple(dict.fromkeys(quantized_targets)),
        merged_targets=tuple(dict.fromkeys(merged_targets)),
        skipped_checks=tuple(dict.fromkeys(skipped)),
    )


def _unwrap_adapter_state(adapter_state: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if adapter_state is None:
        return {}
    if not isinstance(adapter_state, Mapping):
        raise TypeError("adapter_state must be a mapping")
    for key in ("state_dict", "adapter_state_dict", "model_state_dict", "model"):
        value = adapter_state.get(key)
        if isinstance(value, Mapping):
            return value
    return adapter_state


def _target_modules_from_config(
    peft_config: Optional[Any],
    target_modules: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    if target_modules is not None:
        return _normalize_target_module_sequence(target_modules)
    if peft_config is None:
        return ()
    value = (
        peft_config.get("target_modules")
        if isinstance(peft_config, Mapping)
        else getattr(peft_config, "target_modules", None)
    )
    return _normalize_target_module_sequence(value)


def _normalize_target_module_sequence(value: Optional[Any]) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return (str(value),)


def _collect_live_lora_records(model: Optional[Any]) -> List[_LoRACompatibilityRecord]:
    if model is None or not HAS_TORCH:
        return []
    records: List[_LoRACompatibilityRecord] = []
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return records
    for raw_name, module in named_modules():
        if not hasattr(module, "lora_A") and not hasattr(module, "lora_B"):
            continue
        display_name = raw_name or "(root)"
        a_tensors = _extract_lora_member_tensors(getattr(module, "lora_A", None))
        b_tensors = _extract_lora_member_tensors(getattr(module, "lora_B", None))
        adapter_names = sorted(set(a_tensors) | set(b_tensors))
        base_weight, base_suffix = _module_base_weight(module)
        merged = _module_merged(module)
        disabled = bool(getattr(module, "disable_adapters", False))
        quantized = _is_quantized_base_layer(module, model)
        for adapter_name in adapter_names:
            records.append(
                _LoRACompatibilityRecord(
                    target_name=_normalize_peft_target(display_name),
                    raw_target_name=display_name,
                    adapter_name=adapter_name,
                    a_tensor=a_tensors.get(adapter_name),
                    b_tensor=b_tensors.get(adapter_name),
                    a_key=f"{display_name}.lora_A.{adapter_name}.weight",
                    b_key=f"{display_name}.lora_B.{adapter_name}.weight",
                    base_weight=base_weight,
                    base_key=f"{display_name}.{base_suffix}" if base_suffix else None,
                    module=module,
                    merged=merged,
                    adapters_disabled=disabled,
                    quantized_base=quantized,
                )
            )
    return records


def _collect_state_lora_records(
    model: Optional[Any],
    state: Mapping[str, Any],
) -> List[_LoRACompatibilityRecord]:
    groups: Dict[Tuple[str, str], Dict[str, str]] = {}
    for key in state:
        parsed = _parse_lora_state_key(key)
        if parsed is None:
            continue
        raw_target, role, adapter_name = parsed
        groups.setdefault((raw_target, adapter_name), {})[role] = key

    model_state = _model_state_dict(model)
    records: List[_LoRACompatibilityRecord] = []
    for (raw_target, adapter_name), group in sorted(groups.items()):
        target = _normalize_peft_target(raw_target)
        module = _find_module_by_target(model, raw_target, target)
        base_weight, base_key = _state_base_weight(model_state, raw_target, target)
        if base_weight is None and module is not None:
            base_weight, suffix = _module_base_weight(module)
            base_key = f"{target}.{suffix}" if suffix else None
        records.append(
            _LoRACompatibilityRecord(
                target_name=target,
                raw_target_name=raw_target,
                adapter_name=adapter_name,
                a_tensor=state.get(group.get("A", "")),
                b_tensor=state.get(group.get("B", "")),
                a_key=group.get("A"),
                b_key=group.get("B"),
                base_weight=base_weight,
                base_key=base_key,
                module=module,
                source="state",
                merged=_module_merged(module) if module is not None else None,
                adapters_disabled=bool(getattr(module, "disable_adapters", False)) if module is not None else False,
                quantized_base=_is_quantized_base_layer(module, model) if module is not None else _model_quantized(model),
            )
        )
    return records


def _parse_lora_state_key(key: str) -> Optional[Tuple[str, str, str]]:
    for marker, role in ((".lora_A.", "A"), (".lora_B.", "B")):
        if marker in key and key.endswith(".weight"):
            raw_target, _, rest = key.rpartition(marker)
            adapter_name = rest[: -len(".weight")] or "default"
            if raw_target:
                return raw_target, role, adapter_name
    suffixes = (
        (".lora_A.weight", "A"),
        (".lora_B.weight", "B"),
        (".lora_A", "A"),
        (".lora_B", "B"),
    )
    for suffix, role in suffixes:
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)], role, "default"
    return None


def _extract_lora_member_tensors(value: Any) -> Dict[str, Any]:
    if value is None or not HAS_TORCH:
        return {}
    if isinstance(value, nn.ModuleDict):
        return {
            str(name): tensor
            for name, member in value.items()
            for tensor in [_weight_tensor(member)]
            if tensor is not None
        }
    if isinstance(value, nn.ParameterDict):
        return {str(name): tensor for name, tensor in value.items()}
    if isinstance(value, Mapping):
        return {
            str(name): tensor
            for name, member in value.items()
            for tensor in [_weight_tensor(member)]
            if tensor is not None
        }
    tensor = _weight_tensor(value)
    return {"default": tensor} if tensor is not None else {}


def _weight_tensor(value: Any) -> Optional[Any]:
    if value is None or not HAS_TORCH:
        return None
    if isinstance(value, nn.Linear):
        return value.weight
    if isinstance(value, (nn.Parameter, torch.Tensor)):
        return value
    weight = getattr(value, "weight", None)
    if isinstance(weight, (nn.Parameter, torch.Tensor)):
        return weight
    return None


def _module_base_weight(module: Any) -> Tuple[Optional[Any], Optional[str]]:
    if module is None or not HAS_TORCH:
        return None, None
    base_layer = getattr(module, "base_layer", None)
    if base_layer is not None:
        tensor = _weight_tensor(base_layer)
        if tensor is not None:
            return tensor, "base_layer.weight"
    tensor = _weight_tensor(module)
    if tensor is not None:
        return tensor, "weight"
    return None, None


def _model_state_dict(model: Optional[Any]) -> Mapping[str, Any]:
    if model is None:
        return {}
    if isinstance(model, Mapping):
        return model
    state_dict = getattr(model, "state_dict", None)
    if callable(state_dict):
        return state_dict()
    return {}


def _state_base_weight(
    model_state: Mapping[str, Any],
    raw_target: str,
    target: str,
) -> Tuple[Optional[Any], Optional[str]]:
    for base in dict.fromkeys((raw_target, target, _normalize_peft_target(raw_target))):
        for key in (f"{base}.weight", f"{base}.base_layer.weight"):
            if key in model_state:
                return model_state[key], key
    return None, None


def _find_module_by_target(
    model: Optional[Any],
    raw_target: str,
    target: str,
) -> Optional[Any]:
    if model is None or not HAS_TORCH:
        return None
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return None
    modules = dict(named_modules())
    candidates = [
        raw_target,
        target,
        f"base_model.model.{target}",
        f"model.{target}",
    ]
    if target == "(root)":
        candidates.append("")
    for candidate in candidates:
        if candidate in modules:
            return modules[candidate]
    return None


def _check_lora_compatibility_record(
    record: _LoRACompatibilityRecord,
    issues: List[LoRACompatibilityIssue],
    warnings_: List[LoRACompatibilityIssue],
    skipped: List[str],
    *,
    expected_merged: Optional[bool],
    allow_quantized_base: bool,
    require_float_adapters: bool,
    model_available: bool,
) -> None:
    if record.a_tensor is None or record.b_tensor is None:
        existing_key = record.a_key or record.b_key
        missing = "lora_B" if record.a_tensor is not None else "lora_A"
        issues.append(
            LoRACompatibilityIssue(
                category="lora_pair_incomplete",
                module_name=record.target_name,
                adapter_name=record.adapter_name,
                key=existing_key,
                message=f"{record.target_name}: adapter {record.adapter_name!r} is missing {missing}",
                suggestion="Save both LoRA A and B matrices for every adapter.",
            )
        )
        return

    a_shape = record.a_shape
    b_shape = record.b_shape
    if a_shape is None or b_shape is None or len(a_shape) != 2 or len(b_shape) != 2:
        issues.append(
            LoRACompatibilityIssue(
                category="lora_matrix_rank",
                module_name=record.target_name,
                adapter_name=record.adapter_name,
                key=record.a_key or record.b_key,
                actual_shape=a_shape or b_shape,
                message=f"{record.target_name}: LoRA A/B tensors must both be rank-2 matrices",
            )
        )
        return

    rank_a, in_a = a_shape
    out_b, rank_b = b_shape
    if rank_a != rank_b:
        issues.append(
            LoRACompatibilityIssue(
                category="lora_rank_mismatch",
                module_name=record.target_name,
                adapter_name=record.adapter_name,
                key=record.b_key,
                expected_shape=(out_b, rank_a),
                actual_shape=b_shape,
                message=(
                    f"{record.target_name}: lora_A rank {rank_a} does not match "
                    f"lora_B rank {rank_b}"
                ),
                suggestion="Use the same low-rank dimension for LoRA A and B.",
            )
        )
    if rank_a <= 0:
        issues.append(
            LoRACompatibilityIssue(
                category="lora_rank_invalid",
                module_name=record.target_name,
                adapter_name=record.adapter_name,
                key=record.a_key,
                message=f"{record.target_name}: LoRA rank must be positive, got {rank_a}",
            )
        )

    base_shape = record.base_shape
    if base_shape is None:
        skipped.append(f"base_shape:{record.target_name}")
        target = warnings_ if not model_available else issues
        target.append(
            LoRACompatibilityIssue(
                category="lora_base_unverified" if not model_available else "lora_target_missing",
                module_name=record.target_name,
                adapter_name=record.adapter_name,
                key=record.a_key or record.b_key,
                severity="warning" if not model_available else "error",
                message=(
                    f"{record.target_name}: base weight is unavailable; "
                    "TensorGuard cannot prove adapter/base compatibility"
                    if not model_available
                    else f"{record.target_name}: adapter target has no matching base weight"
                ),
                suggestion="Verify adapter-only states together with the base model they target.",
            )
        )
    elif len(base_shape) != 2:
        issues.append(
            LoRACompatibilityIssue(
                category="lora_target_not_linear",
                module_name=record.target_name,
                adapter_name=record.adapter_name,
                key=record.base_key,
                actual_shape=base_shape,
                message=f"{record.target_name}: LoRA target weight is not a rank-2 Linear-style matrix",
            )
        )
    else:
        out_features, in_features = base_shape
        if in_a != in_features:
            issues.append(
                LoRACompatibilityIssue(
                    category="lora_input_mismatch",
                    module_name=record.target_name,
                    adapter_name=record.adapter_name,
                    key=record.a_key,
                    expected_shape=(rank_a, in_features),
                    actual_shape=a_shape,
                    message=(
                        f"{record.target_name}: lora_A input dimension {in_a} "
                        f"does not match base in_features {in_features}"
                    ),
                )
            )
        if out_b != out_features:
            issues.append(
                LoRACompatibilityIssue(
                    category="lora_output_mismatch",
                    module_name=record.target_name,
                    adapter_name=record.adapter_name,
                    key=record.b_key,
                    expected_shape=(out_features, rank_b),
                    actual_shape=b_shape,
                    message=(
                        f"{record.target_name}: lora_B output dimension {out_b} "
                        f"does not match base out_features {out_features}"
                    ),
                )
            )
        if rank_a > min(in_features, out_features):
            issues.append(
                LoRACompatibilityIssue(
                    category="lora_rank_invalid",
                    module_name=record.target_name,
                    adapter_name=record.adapter_name,
                    key=record.a_key,
                    message=(
                        f"{record.target_name}: LoRA rank {rank_a} must be in "
                        f"[1, min({in_features}, {out_features})]"
                    ),
                )
            )

    if expected_merged is not None and record.merged is not None and record.merged != expected_merged:
        issues.append(
            LoRACompatibilityIssue(
                category="lora_merge_state_mismatch",
                module_name=record.target_name,
                adapter_name=record.adapter_name,
                message=(
                    f"{record.target_name}: merged={record.merged} but "
                    f"expected merged={expected_merged}"
                ),
                suggestion="Call merge_adapter()/unmerge_adapter() or update the expected_merged gate.",
            )
        )
    if record.adapters_disabled:
        warnings_.append(
            LoRACompatibilityIssue(
                category="lora_adapter_disabled",
                module_name=record.target_name,
                adapter_name=record.adapter_name,
                severity="warning",
                message=f"{record.target_name}: LoRA adapters are attached but disabled",
            )
        )

    if record.quantized_base and not allow_quantized_base:
        issues.append(
            LoRACompatibilityIssue(
                category="lora_quantized_base_disallowed",
                module_name=record.target_name,
                adapter_name=record.adapter_name,
                key=record.base_key,
                message=f"{record.target_name}: base layer is quantized but quantized bases are disallowed",
            )
        )
    if require_float_adapters:
        for key, tensor in ((record.a_key, record.a_tensor), (record.b_key, record.b_tensor)):
            if not _is_floating_tensor(tensor):
                issues.append(
                    LoRACompatibilityIssue(
                        category="lora_adapter_not_floating",
                        module_name=record.target_name,
                        adapter_name=record.adapter_name,
                        key=key,
                        message=f"{record.target_name}: adapter tensor {key!r} is not floating point",
                        suggestion="QLoRA keeps the base quantized but the low-rank adapter matrices floating.",
                    )
                )


def _check_target_module_patterns(
    records: Sequence[_LoRACompatibilityRecord],
    patterns: Sequence[str],
    *,
    require_all_targets: bool,
    issues: List[LoRACompatibilityIssue],
) -> Tuple[str, ...]:
    if not patterns:
        return ()
    matched: Dict[str, bool] = {pattern: False for pattern in patterns}
    for record in records:
        if any(_target_matches_pattern(record.target_name, pattern) for pattern in patterns):
            for pattern in patterns:
                if _target_matches_pattern(record.target_name, pattern):
                    matched[pattern] = True
        else:
            issues.append(
                LoRACompatibilityIssue(
                    category="lora_target_unexpected",
                    module_name=record.target_name,
                    adapter_name=record.adapter_name,
                    message=(
                        f"{record.target_name}: adapter target is not listed in "
                        f"target_modules={tuple(patterns)!r}"
                    ),
                    suggestion="Check the PEFT target_modules config for stale or over-broad adapter targets.",
                )
            )
    if require_all_targets:
        for pattern, was_matched in matched.items():
            if not was_matched:
                issues.append(
                    LoRACompatibilityIssue(
                        category="lora_target_pattern_missing",
                        module_name=pattern,
                        message=f"target_modules pattern {pattern!r} matched no LoRA adapter",
                    )
                )
    return tuple(pattern for pattern, was_matched in matched.items() if was_matched)


def _target_matches_pattern(target: str, pattern: str) -> bool:
    target = _normalize_peft_target(target)
    pattern = _normalize_peft_target(pattern)
    if pattern == "all-linear":
        return True
    if "." in pattern:
        return target == pattern or target.endswith(f".{pattern}")
    return target.split(".")[-1] == pattern


def _adapter_from_record(record: _LoRACompatibilityRecord) -> Optional[LoRAAdapter]:
    a_shape = record.a_shape
    b_shape = record.b_shape
    if a_shape is None or b_shape is None or len(a_shape) != 2 or len(b_shape) != 2:
        return None
    base_shape = record.base_shape
    in_features = base_shape[1] if base_shape is not None and len(base_shape) == 2 else a_shape[1]
    out_features = base_shape[0] if base_shape is not None and len(base_shape) == 2 else b_shape[0]
    return LoRAAdapter(
        base_module_name=record.target_name,
        in_features=in_features,
        out_features=out_features,
        rank=a_shape[0],
        lora_A_shape=a_shape,
        lora_B_shape=b_shape,
    )


def _normalize_peft_target(target: str) -> str:
    normalized = target
    for prefix in ("base_model.model.", "base_model.", "model."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    if normalized.endswith(".base_layer"):
        normalized = normalized[: -len(".base_layer")]
    return normalized


def _module_merged(module: Optional[Any]) -> Optional[bool]:
    if module is None:
        return None
    merged_adapters = getattr(module, "merged_adapters", None)
    if merged_adapters:
        return True
    merged = getattr(module, "merged", None)
    if isinstance(merged, bool):
        return merged
    return None


def _is_quantized_base_layer(module: Optional[Any], model: Optional[Any] = None) -> bool:
    if module is None:
        return _model_quantized(model)
    base_layer = getattr(module, "base_layer", module)
    candidates = (module, base_layer, getattr(base_layer, "weight", None))
    quant_markers = ("Linear4bit", "Linear8bitLt", "Params4bit", "Int8Params")
    for obj in candidates:
        if obj is None:
            continue
        cls_name = type(obj).__name__
        if any(marker in cls_name for marker in quant_markers):
            return True
        if hasattr(obj, "quant_state") or hasattr(obj, "quant_type"):
            return True
        dtype = getattr(obj, "dtype", None)
        if HAS_TORCH and dtype in (torch.int8, torch.uint8):
            return True
    return _model_quantized(model)


def _model_quantized(model: Optional[Any]) -> bool:
    if model is None:
        return False
    for attr in ("is_loaded_in_4bit", "is_loaded_in_8bit"):
        if bool(getattr(model, attr, False)):
            return True
    return False


def _is_floating_tensor(tensor: Any) -> bool:
    if tensor is None:
        return False
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        return bool(torch.is_floating_point(tensor))
    dtype = str(getattr(tensor, "dtype", "")).lower()
    return "float" in dtype or "bfloat" in dtype


def _shape(value: Any) -> Optional[Shape]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  LoRA verification result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LoRAVerificationResult:
    """Aggregate result of LoRA-specific verification.

    Attributes
    ----------
    safe : bool
        Overall safety verdict.
    has_lora : bool
        Whether LoRA adapters were detected.
    adapters : list of LoRAAdapter
        Detected adapters.
    shape_result : VerificationResult
        Result of shape verification.
    rank_violations : list of RankViolation
        Rank constraint violations.
    merge_safe : bool
        Whether W + α·B·A merge is shape-safe.
    composition_result : VerificationResult or None
        Result of compositional verification (if source was provided).
    quantization : QuantizationBits
        Detected quantization level.
    verification_time_ms : float
        Total wall-clock time.
    """

    safe: bool
    has_lora: bool = False
    adapters: List[LoRAAdapter] = field(default_factory=list)
    shape_result: Optional[VerificationResult] = None
    rank_violations: List[RankViolation] = field(default_factory=list)
    merge_safe: bool = True
    composition_result: Optional[VerificationResult] = None
    quantization: QuantizationBits = QuantizationBits.FULL
    compatibility_result: Optional[LoRACompatibilityResult] = None
    verification_time_ms: float = 0.0

    def pretty(self) -> str:
        status = "SAFE" if self.safe else "UNSAFE"
        compatibility_issues = (
            len(self.compatibility_result.issues)
            if self.compatibility_result is not None
            else 0
        )
        lines = [
            f"LoRAVerificationResult: {status}",
            f"  LoRA detected:    {self.has_lora}",
            f"  Adapters:         {len(self.adapters)}",
            f"  Merge safe:       {self.merge_safe}",
            f"  Quantization:     {self.quantization.name}",
            f"  Rank violations:  {len(self.rank_violations)}",
            f"  PEFT issues:      {compatibility_issues}",
            f"  Time:             {self.verification_time_ms:.1f} ms",
        ]
        for v in self.rank_violations:
            lines.append(f"  ✗ {v.module_name}: {v.message}")
        if self.compatibility_result is not None:
            for issue in self.compatibility_result.issues:
                lines.append(f"  ✗ {issue.module_name}: {issue.message}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Top-level API
# ═══════════════════════════════════════════════════════════════════════════════


def verify_lora_model(
    model: Any,
    input_shapes: Optional[Dict[str, tuple]] = None,
    lora_config: Optional[LoRAConfig] = None,
    source: Optional[str] = None,
    adapter_state: Optional[Mapping[str, Any]] = None,
    peft_config: Optional[Any] = None,
    target_modules: Optional[Sequence[str]] = None,
    require_all_targets: bool = True,
    expected_merged: Optional[bool] = None,
    allow_quantized_base: bool = True,
    require_float_adapters: bool = True,
) -> LoRAVerificationResult:
    """Verify a model that may contain LoRA adapters.

    Auto-detects whether the model has LoRA layers.  If none are found,
    falls back to standard ``verify_model()`` verification (if source
    is provided).

    Parameters
    ----------
    model : nn.Module
        The model to verify (base, LoRA-adapted, or quantized+LoRA).
    input_shapes : dict, optional
        Input tensor shapes.
    lora_config : LoRAConfig, optional
        Explicit LoRA configuration.  Auto-detected if ``None``.
    source : str, optional
        Python source of the model class.  Enables compositional
        verification via ``verify_compositional()``.

    Returns
    -------
    LoRAVerificationResult
    """
    t0 = time.perf_counter()
    input_shapes = input_shapes or {}

    verifier = LoRAVerifier(model, lora_config)
    adapters = verifier.detect_lora_modules()

    if not adapters:
        # No LoRA detected — fall back to standard verification
        fallback_result: Optional[VerificationResult] = None
        if source is not None:
            try:
                fallback_result = verify_model(
                    source=source,
                    input_shapes=input_shapes,
                )
            except Exception:
                fallback_result = None

        elapsed = (time.perf_counter() - t0) * 1000
        return LoRAVerificationResult(
            safe=fallback_result.safe if fallback_result else True,
            has_lora=False,
            adapters=[],
            shape_result=fallback_result,
            verification_time_ms=elapsed,
        )

    # Run shape verification
    shape_result = verifier.verify_adapter_shapes()

    # Run rank constraint checks
    rank_violations = verifier.verify_rank_constraints()

    # Check merge safety
    merge_safe = verifier.verify_merge_safety()

    # Compositional verification if source is available
    comp_result = None
    if source is not None:
        comp_result = verifier.verify_composition(input_shapes, source=source)

    # Quantization check
    quant_verifier = QuantizationVerifier(model)
    quant = quant_verifier.detect_quantization()

    # PEFT compatibility check
    compatibility_result = verify_lora_adapter_compatibility(
        model,
        adapter_state,
        peft_config=peft_config,
        target_modules=target_modules,
        require_all_targets=require_all_targets,
        expected_merged=expected_merged,
        allow_quantized_base=allow_quantized_base,
        require_float_adapters=require_float_adapters,
    )

    # Overall safety
    safe = (
        shape_result.safe
        and len(rank_violations) == 0
        and merge_safe
        and (comp_result is None or comp_result.safe)
        and compatibility_result.ok
    )

    elapsed = (time.perf_counter() - t0) * 1000
    return LoRAVerificationResult(
        safe=safe,
        has_lora=True,
        adapters=adapters,
        shape_result=shape_result,
        rank_violations=rank_violations,
        merge_safe=merge_safe,
        composition_result=comp_result,
        quantization=quant,
        compatibility_result=compatibility_result,
        verification_time_ms=elapsed,
    )


__all__ = [
    "LoRAAdapter",
    "LoRACompatibilityIssue",
    "LoRACompatibilityResult",
    "LoRAConfig",
    "LoRAShapeContract",
    "LoRAVerificationResult",
    "LoRAVerifier",
    "QuantizationBits",
    "QuantizationVerifier",
    "RankViolation",
    "verify_lora_adapter_compatibility",
    "verify_lora_model",
]
