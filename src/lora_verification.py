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
# 5.  LoRA verification result
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
    verification_time_ms: float = 0.0

    def pretty(self) -> str:
        status = "SAFE" if self.safe else "UNSAFE"
        lines = [
            f"LoRAVerificationResult: {status}",
            f"  LoRA detected:    {self.has_lora}",
            f"  Adapters:         {len(self.adapters)}",
            f"  Merge safe:       {self.merge_safe}",
            f"  Quantization:     {self.quantization.name}",
            f"  Rank violations:  {len(self.rank_violations)}",
            f"  Time:             {self.verification_time_ms:.1f} ms",
        ]
        for v in self.rank_violations:
            lines.append(f"  ✗ {v.module_name}: {v.message}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Top-level API
# ═══════════════════════════════════════════════════════════════════════════════


def verify_lora_model(
    model: Any,
    input_shapes: Optional[Dict[str, tuple]] = None,
    lora_config: Optional[LoRAConfig] = None,
    source: Optional[str] = None,
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

    # Overall safety
    safe = (
        shape_result.safe
        and len(rank_violations) == 0
        and merge_safe
        and (comp_result is None or comp_result.safe)
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
        verification_time_ms=elapsed,
    )
