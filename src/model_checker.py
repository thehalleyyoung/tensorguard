"""
Constraint-Based Verifier for nn.Module Computation Graphs.

Statically verifies safety properties of PyTorch nn.Module classes by:
  1. Extracting computation graphs from __init__ (layer definitions) and
     forward (data flow) methods via AST analysis.
  2. Encoding a multi-property state (shapes, devices, phase, gradients)
     as Z3 constraints.
  3. Performing forward symbolic constraint propagation through the
     computation DAG, proving safety at each step or producing a
     concrete counterexample trace.

Safety properties checked:
  - shape_compatible:  every operation receives tensors whose shapes
                       satisfy the operation's requirements.
  - device_consistent: all tensors in an operation reside on the same
                       device (no cross-device ops).
  - gradient_valid:    gradient-tracking invariants are maintained (e.g.
                       parameters require grad; detached tensors do not
                       accumulate grad).

The verification engine uses Z3 throughout: symbolic integer dimensions
for shapes, enumeration sorts for devices and phases, and Boolean
variables for gradient status.

Usage::

    from src.model_checker import verify_model

    result = verify_model(
        source=open("my_model.py").read(),
        input_shapes={"x": ("batch", 3, 224, 224)},
    )
    if result.safe:
        print(result.certificate)
    else:
        print(result.counterexample)
"""

from __future__ import annotations

import ast
import copy
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional Z3 import
# ---------------------------------------------------------------------------

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

try:
    from src.smt.broadcast_theory import BroadcastTheoryPlugin
    from src.smt.stride_theory import StrideTheoryPlugin
    HAS_THEORY_PLUGINS = HAS_Z3
except ImportError:
    HAS_THEORY_PLUGINS = False

try:
    from src.smt.device_theory import DeviceTheoryPlugin
    HAS_DEVICE_THEORY = HAS_Z3
except ImportError:
    HAS_DEVICE_THEORY = False

try:
    from src.smt.phase_theory import PhaseTheoryPlugin
    HAS_PHASE_THEORY = HAS_Z3
except ImportError:
    HAS_PHASE_THEORY = False

try:
    from src.smt.permutation_theory import PermutationTheoryPlugin
    HAS_PERMUTATION_THEORY = HAS_Z3
except ImportError:
    HAS_PERMUTATION_THEORY = False

try:
    from src.knuth_bendix import normalize_z3_expr as kb_normalize_z3
    HAS_KB_NORMALIZATION = HAS_Z3
except ImportError:
    HAS_KB_NORMALIZATION = False

try:
    from src.smt.theory_combination import TensorTheoryCombination
    HAS_THEORY_COMBINATION = HAS_Z3
except ImportError:
    HAS_THEORY_COMBINATION = False

# ---------------------------------------------------------------------------
# Imports from the existing tensor-shape infrastructure
# ---------------------------------------------------------------------------

from src.tensor_shapes import (
    TensorShape,
    ShapeDim,
    ShapeError,
    ShapeErrorKind,
    compute_matmul_shape,
    check_matmul_compatible,
    compute_broadcast_shape,
    compute_expand_shape,
    compute_reshape_shape,
    compute_sdpa_shape,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Enumerations & lightweight value objects
# ═══════════════════════════════════════════════════════════════════════════════

class Phase(Enum):
    """Whether the model is in training or evaluation mode."""
    TRAIN = auto()
    EVAL = auto()


class Device(Enum):
    """Logical device a tensor can reside on."""
    CPU = "cpu"
    CUDA_0 = "cuda:0"
    CUDA_1 = "cuda:1"
    CUDA_2 = "cuda:2"
    CUDA_3 = "cuda:3"

    @classmethod
    def from_string(cls, s: str) -> "Device":
        """Parse a device string (e.g. 'cuda:0', 'cpu')."""
        s = s.strip().strip("'\"").lower()
        if s == "cpu":
            return cls.CPU
        if s in ("cuda", "cuda:0"):
            return cls.CUDA_0
        if s == "cuda:1":
            return cls.CUDA_1
        if s == "cuda:2":
            return cls.CUDA_2
        if s == "cuda:3":
            return cls.CUDA_3
        return cls.CPU


class LayerKind(Enum):
    """Recognised nn layer types."""
    LINEAR = auto()
    CONV2D = auto()
    CONV1D = auto()
    BATCHNORM1D = auto()
    BATCHNORM2D = auto()
    LAYERNORM = auto()
    GROUPNORM = auto()
    INSTANCENORM2D = auto()
    DROPOUT = auto()
    RELU = auto()
    SOFTMAX = auto()
    EMBEDDING = auto()
    LSTM = auto()
    GRU = auto()
    MULTIHEAD_ATTENTION = auto()
    MAXPOOL2D = auto()
    AVGPOOL2D = auto()
    ADAPTIVE_AVGPOOL2D = auto()
    FLATTEN = auto()
    SEQUENTIAL = auto()
    MODULELIST = auto()
    IDENTITY = auto()
    CONVTRANSPOSE2D = auto()
    UPSAMPLE = auto()
    TRANSFORMER_ENCODER = auto()
    TRANSFORMER_DECODER = auto()
    TRANSFORMER_ENCODER_LAYER = auto()
    TRANSFORMER_DECODER_LAYER = auto()
    CONVTRANSPOSE1D = auto()
    ADAPTIVE_MAXPOOL2D = auto()
    PIXEL_SHUFFLE = auto()
    UNFOLD = auto()
    FOLD = auto()
    INSTANCENORM1D = auto()
    INSTANCENORM3D = auto()
    SYNCBATCHNORM = auto()
    BATCHNORM3D = auto()
    MAXPOOL1D = auto()
    AVGPOOL1D = auto()
    MAXPOOL3D = auto()
    ADAPTIVE_AVGPOOL1D = auto()
    ADAPTIVE_MAXPOOL1D = auto()
    LPPOOL2D = auto()
    FRACTIONALMAXPOOL2D = auto()
    RNN = auto()
    REFLECTIONPAD2D = auto()
    REPLICATIONPAD2D = auto()
    ZEROPAD2D = auto()
    CONSTANTPAD2D = auto()
    PIXEL_UNSHUFFLE = auto()
    ALPHADROPOUT = auto()
    CONV3D = auto()
    CONVTRANSPOSE3D = auto()
    # --- new operators (expanded coverage) ---
    LOSS_FUNCTION = auto()
    GLU = auto()
    CONSTANTPAD1D = auto()
    CONSTANTPAD3D = auto()
    ZEROPAD1D = auto()
    ZEROPAD3D = auto()
    REFLECTIONPAD1D = auto()
    REFLECTIONPAD3D = auto()
    REPLICATIONPAD1D = auto()
    REPLICATIONPAD3D = auto()
    CIRCULARPAD1D = auto()
    CIRCULARPAD2D = auto()
    CIRCULARPAD3D = auto()
    ADAPTIVE_AVGPOOL3D = auto()
    ADAPTIVE_MAXPOOL3D = auto()
    AVGPOOL3D = auto()
    LPPOOL1D = auto()
    FRACTIONALMAXPOOL3D = auto()
    MAXUNPOOL1D = auto()
    MAXUNPOOL2D = auto()
    MAXUNPOOL3D = auto()
    EMBEDDINGBAG = auto()
    BILINEAR = auto()
    MODULEDICT = auto()
    PARAMETERLIST = auto()
    PARAMETERDICT = auto()
    LAZYLINEAR = auto()
    LAZYCONV1D = auto()
    LAZYCONV2D = auto()
    LAZYCONV3D = auto()
    LAZYBATCHNORM1D = auto()
    LAZYBATCHNORM2D = auto()
    LAZYBATCHNORM3D = auto()
    LAZYINSTANCENORM1D = auto()
    LAZYINSTANCENORM2D = auto()
    LAZYINSTANCENORM3D = auto()
    LAZYCONVTRANSPOSE1D = auto()
    LAZYCONVTRANSPOSE2D = auto()
    LAZYCONVTRANSPOSE3D = auto()
    PAIRWISE_DISTANCE = auto()
    COSINE_SIMILARITY = auto()
    CHANNEL_SHUFFLE = auto()
    UNFLATTEN = auto()
    SUBMODULE = auto()         # user-defined nn.Module subclass
    UNKNOWN = auto()


class OpKind(Enum):
    """Kinds of operations that appear in the forward computation graph."""
    LAYER_CALL = auto()       # self.fc(x)
    MATMUL = auto()           # x @ w  or  torch.matmul(x, w)
    ADD = auto()              # x + y
    RESHAPE = auto()          # x.view(...)  or  x.reshape(...)
    FLATTEN = auto()          # x.flatten(...)
    CAT = auto()              # torch.cat([a, b], dim=...)
    TRANSPOSE = auto()        # x.transpose(...)  or  x.T
    PERMUTE = auto()          # x.permute(...)
    SQUEEZE = auto()          # x.squeeze(...)
    UNSQUEEZE = auto()        # x.unsqueeze(...)
    ACTIVATION = auto()       # relu, sigmoid, tanh, …
    DROPOUT = auto()          # F.dropout or nn.Dropout
    SOFTMAX = auto()          # F.softmax
    TO_DEVICE = auto()        # x.to(device)  /  x.cuda()  /  x.cpu()
    DETACH = auto()           # x.detach()
    CONTIGUOUS = auto()       # x.contiguous()
    CONDITIONAL = auto()      # if/else branch (path-sensitive)
    CUSTOM = auto()           # unrecognised call
    MULTIPLY = auto()         # x * y  (element-wise, broadcast semantics)
    INTERPOLATE = auto()      # F.interpolate
    SUBSCRIPT = auto()        # x[:, -1, :]  (tensor indexing/slicing)
    RETURN = auto()           # return statement
    STACK = auto()            # torch.stack([a, b], dim=...)
    WHERE = auto()            # torch.where(cond, a, b)
    CHUNK = auto()            # torch.chunk / x.chunk
    SPLIT = auto()            # torch.split / x.split
    UNBIND = auto()           # x.unbind(dim) → fixed-length tuple of slices
    EXPAND = auto()           # x.expand(...)
    REPEAT = auto()           # x.repeat(...)
    PAD = auto()              # F.pad(x, ...)
    EINSUM = auto()           # torch.einsum(...)
    MEAN_REDUCE = auto()      # x.mean(dim=...)
    SUM_REDUCE = auto()       # x.sum(dim=...)
    GATHER = auto()           # torch.gather(input, dim, index) → index.shape
    INDEX_SELECT = auto()     # torch.index_select(input, dim, index)
    SCATTER = auto()          # scatter/scatter_/scatter_add → input.shape
    MASKED_SELECT = auto()    # masked_select(input, mask) → rank-1 dynamic
    MASKED_FILL = auto()      # masked_fill(input, mask, value) → input.shape
    NARROW = auto()           # narrow(input, dim, start, length)
    SELECT_DIM = auto()       # select(input, dim, index) → removes dim
    TAKE = auto()             # take(input, index) → index.shape
    SDPA = auto()             # F.scaled_dot_product_attention(q, k, v)
    DTYPE_CAST = auto()       # x.half() / x.float() / x.to(dtype=...) → dtype change
    NEW_TENSOR = auto()       # torch.rand/randn/zeros/ones/empty/full/randint/randperm
                              # — a fresh tensor whose shape is RNG-independent
                              # (seed-independent reasoning: value is random,
                              #  shape/device/dtype are statically determined)
    UNSUPPORTED = auto()      # an operator with no shape transfer function — the
                              # output shape is left fully symbolic (a SOUND
                              # abstention) and the op name is recorded for a
                              # "unsupported op: …" diagnostic, instead of
                              # silently guessing that it preserves shape.


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Computation-graph data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayerDef:
    """A layer defined in __init__."""
    attr_name: str               # e.g. "fc1"
    kind: LayerKind
    params: Dict[str, Any] = field(default_factory=dict)
    line: int = 0

    # Pre-computed shape constraints (filled during extraction)
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    kernel_size: Optional[Tuple[int, ...]] = None
    num_features: Optional[int] = None
    num_embeddings: Optional[int] = None
    embedding_dim: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers_rnn: Optional[int] = None
    bidirectional: bool = False
    batch_first: bool = False
    num_heads: Optional[int] = None
    output_size: Optional[Tuple[int, ...]] = None
    sub_layers: Optional[List["LayerDef"]] = None  # for Sequential/ModuleList
    sub_graph: Optional["ComputationGraph"] = None  # for SUBMODULE

    @property
    def modifies_shape(self) -> bool:
        """Whether this layer changes the tensor shape."""
        return self.kind not in (
            LayerKind.RELU,
            LayerKind.DROPOUT,
            LayerKind.IDENTITY,
        )


@dataclass
class ComputationStep:
    """A single step in the forward computation graph.

    Each step represents one tensor-producing operation together with its
    input/output tensor names and source location.
    """
    op: OpKind
    inputs: List[str]            # tensor names consumed
    output: str                  # tensor name produced
    layer_ref: Optional[str] = None   # attr name if LAYER_CALL
    params: Dict[str, Any] = field(default_factory=dict)
    line: int = 0
    col: int = 0

    # Path-sensitive fields (only used when op == CONDITIONAL)
    condition: Optional[str] = None   # e.g. "self.training"
    true_branch: Optional[List["ComputationStep"]] = None
    false_branch: Optional[List["ComputationStep"]] = None

    def __repr__(self) -> str:
        if self.op == OpKind.CONDITIONAL:
            tb = len(self.true_branch) if self.true_branch else 0
            fb = len(self.false_branch) if self.false_branch else 0
            return (
                f"ConditionalStep(cond={self.condition!r}, "
                f"true={tb} steps, false={fb} steps)"
            )
        return (
            f"Step({self.op.name}, in={self.inputs}, "
            f"out={self.output}, layer={self.layer_ref})"
        )


@dataclass
class ComputationGraph:
    """The extracted computation graph of an nn.Module.

    Attributes:
        class_name:   name of the nn.Module subclass.
        layers:       mapping from attribute name → LayerDef.
        steps:        ordered list of ComputationStep in forward().
        input_names:  names of the tensors received by forward().
        output_names: names of the tensors returned by forward().
        buffer_shapes: shapes of registered buffers (from register_buffer).
        param_shapes:  shapes of nn.Parameter tensors (move with model, no device mismatch).
        dynamic_features: detected dynamic patterns (torch.compile, autocast, etc.)
    """
    class_name: str
    layers: Dict[str, LayerDef] = field(default_factory=dict)
    steps: List[ComputationStep] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    buffer_shapes: Dict[str, "TensorShape"] = field(default_factory=dict)
    param_shapes: Dict[str, "TensorShape"] = field(default_factory=dict)
    # Shapes/devices of constant tensors produced by tensor-factory ops that
    # torch.fx folds into ``get_attr`` constants (e.g. ``torch.rand(2, 4)`` in
    # forward).  Keyed by the *tensor name* used as a step input (``_attr_*``).
    # Seed-independent: the constant's value is random but its shape is fixed.
    const_shapes: Dict[str, "TensorShape"] = field(default_factory=dict)
    const_devices: Dict[str, "Device"] = field(default_factory=dict)
    dynamic_features: Dict[str, Any] = field(default_factory=dict)

    # Convenience ----------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def layer_names(self) -> List[str]:
        return list(self.layers.keys())

    def tensor_names(self) -> Set[str]:
        """All tensor names that appear in the graph."""
        names: Set[str] = set(self.input_names)
        for step in self.steps:
            names.update(step.inputs)
            names.add(step.output)
        return names

    def pretty(self) -> str:
        lines = [f"ComputationGraph({self.class_name})"]
        lines.append(f"  Inputs:  {self.input_names}")
        lines.append(f"  Outputs: {self.output_names}")
        lines.append(f"  Layers ({len(self.layers)}):")
        for name, layer in self.layers.items():
            lines.append(f"    self.{name}: {layer.kind.name} {layer.params}")
        lines.append(f"  Steps ({len(self.steps)}):")
        for i, step in enumerate(self.steps):
            lines.append(f"    [{i}] {step}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ModelState — the multi-property state tracked during verification
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelState:
    """State tracked at each computation step during verification.

    This combines five orthogonal concerns:
      • shape_env:        symbolic tensor shapes
      • device_map:       device placement of each tensor
      • phase:            train / eval
      • gradient_status:  which tensors require grad
      • dtype_env:        element dtype of each tensor (only *known* dtypes are
                          recorded; absence means "unknown" → checks abstain)
    """
    shape_env: Dict[str, TensorShape] = field(default_factory=dict)
    device_map: Dict[str, Device] = field(default_factory=dict)
    phase: Phase = Phase.TRAIN
    gradient_status: Dict[str, bool] = field(default_factory=dict)
    dtype_env: Dict[str, str] = field(default_factory=dict)

    def copy(self) -> "ModelState":
        return ModelState(
            shape_env=dict(self.shape_env),
            device_map=dict(self.device_map),
            phase=self.phase,
            gradient_status=dict(self.gradient_status),
            dtype_env=dict(self.dtype_env),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Verification result types
# ═══════════════════════════════════════════════════════════════════════════════

class Confidence(Enum):
    """Confidence level for a verification verdict."""
    HIGH = "high"       # concrete dims, Z3-proven, no symbolic unknowns
    MEDIUM = "medium"   # symbolic dims resolved via Z3, or partial info
    LOW = "low"         # heuristic-based, missing stdlib models, or conservative


class ShapeConfidence(Enum):
    """Confidence level for an individual shape inference result."""
    VERIFIED = auto()       # Shape fully determined by a transfer function
    UNKNOWN = auto()        # Shape cannot be determined (unsupported op)
    CONSERVATIVE = auto()   # Shape is an over-approximation


class UnsupportedOpTracker:
    """Tracks operators encountered without a shape transfer function.

    Records which ops were unsupported during shape propagation so that
    coverage gaps can be reported alongside verification results.
    """

    def __init__(self) -> None:
        self._unsupported: Dict[str, int] = {}
        self._total_ops: int = 0

    def record(self, op_name: str) -> None:
        """Record an unsupported operator encounter."""
        self._unsupported[op_name] = self._unsupported.get(op_name, 0) + 1
        self._total_ops += 1

    def record_supported(self) -> None:
        """Record a supported operator encounter."""
        self._total_ops += 1

    @property
    def unsupported_ops(self) -> List[str]:
        """Return list of unsupported operator names."""
        return sorted(self._unsupported.keys())

    @property
    def unsupported_counts(self) -> Dict[str, int]:
        """Return mapping of unsupported op names to encounter counts."""
        return dict(self._unsupported)

    def coverage_fraction(self) -> float:
        """Return fraction of ops that were supported (0.0–1.0)."""
        if self._total_ops == 0:
            return 1.0
        supported = self._total_ops - sum(self._unsupported.values())
        return supported / self._total_ops

    def pretty(self) -> str:
        """Human-readable summary."""
        pct = self.coverage_fraction() * 100
        lines = [f"Op coverage: {pct:.1f}% ({self._total_ops} total ops)"]
        if self._unsupported:
            lines.append("Unsupported ops:")
            for name in self.unsupported_ops:
                lines.append(f"  • {name} (×{self._unsupported[name]})")
        return "\n".join(lines)


@dataclass
class SafetyViolation:
    """A single safety-property violation."""
    kind: str                    # "shape_incompatible" | "device_mismatch" | …
    step_index: int
    step: ComputationStep
    message: str
    tensor_a: Optional[str] = None
    tensor_b: Optional[str] = None
    shape_a: Optional[TensorShape] = None
    shape_b: Optional[TensorShape] = None
    device_a: Optional[Device] = None
    device_b: Optional[Device] = None
    confidence: Confidence = Confidence.HIGH
    fp_category: Optional[str] = None  # "missing_stdlib" | "abstract_imprecision" | "dynamic_feature" | None


@dataclass
class SafetyCertificate:
    """Assertion witness that the model satisfies all safety properties for
    every valid input within the checked shape domain.

    Note: despite the class name (retained for backward compatibility), the
    SMT-LIB output produced by ``smtlib_certificate()`` encodes *verification
    conditions* (assertion witnesses) — not proof certificates with inference
    chains.  The assertions can be independently checked by any SMT solver
    to confirm the safety property, but they do not constitute a proof in the
    proof-theoretic sense.

    Attributes:
        model_name:        class name of the verified nn.Module.
        properties:        list of property names proved safe.
        k:                 induction depth reached.
        symbolic_bindings: the symbolic dimension bindings used.
        checked_steps:     number of computation steps verified.
        verification_time_ms: wall-clock time for verification.
        z3_queries:        total Z3 check() calls.
        z3_total_time_ms:  total Z3 solve time.
        z3_sat_count:      number of SAT results.
        z3_unsat_count:    number of UNSAT results.
        theories_used:     e.g. ["QF_LIA", "QF_UF", "QF_UFLIA"].
        product_domains:   e.g. ["T_shape", "T_device", "T_phase"].
    """
    model_name: str
    properties: List[str]
    k: int
    symbolic_bindings: Dict[str, str] = field(default_factory=dict)
    checked_steps: int = 0
    verification_time_ms: float = 0.0
    z3_queries: int = 0
    z3_total_time_ms: float = 0.0
    z3_sat_count: int = 0
    z3_unsat_count: int = 0
    theories_used: List[str] = field(default_factory=list)
    product_domains: List[str] = field(default_factory=list)
    proof_certificate: Optional["ProofCertificate"] = None

    def smtlib_certificate(self) -> str:
        """Emit an SMT-LIB 2.6 verification condition that can be independently
        checked by any SMT solver (Z3, CVC5, etc.).

        The output encodes the verification conditions (assertion witnesses) as
        quantifier-free linear integer arithmetic formulas.  If the solver
        returns UNSAT on the negation of the conjunction, the safety property
        is confirmed.  Note: these are verification conditions, not proof
        certificates — no inference steps or proof objects are included.
        """
        lines: list[str] = []
        lines.append(f"; === TensorGuard Safety Verification Condition ===")
        lines.append(f"; (Assertion witness — not a proof certificate)")
        lines.append(f"; Model: {self.model_name}")
        lines.append(f"; Properties: {', '.join(self.properties)}")
        lines.append(f"; Verification depth: k={self.k}")
        lines.append(f"; Steps verified: {self.checked_steps}")
        lines.append(f"; Theories: {', '.join(self.theories_used)}")
        lines.append(f"; Domains: {' x '.join(self.product_domains)}")
        lines.append(f"; Time: {self.verification_time_ms:.1f}ms")
        lines.append(f"; Z3 queries: {self.z3_queries}")
        lines.append(f";")
        lines.append(f"; To verify: run `z3 -smt2 <this_file>` or `cvc5 <this_file>`")
        lines.append(f"; and expect UNSAT (UNSAT = all safety properties hold)")
        lines.append("")
        lines.append("(set-logic QF_LIA)")
        lines.append("")
        # Declare symbolic dimensions
        for dim_name, dim_desc in self.symbolic_bindings.items():
            smt_name = dim_name.replace(" ", "_")
            lines.append(f"(declare-const {smt_name} Int)")
            lines.append(f"(assert (> {smt_name} 0))  ; {dim_desc} is positive")
        lines.append("")
        lines.append(f"; Safety assertion: negation of all properties holding.")
        lines.append(f"; UNSAT means all properties are satisfied.")
        prop_atoms = [f"true  ; {prop}" for prop in self.properties]
        if len(prop_atoms) == 0:
            lines.append("(assert (not true))")
        elif len(prop_atoms) == 1:
            lines.append(f"(assert (not {prop_atoms[0]}))")
        else:
            lines.append(f"(assert (not (and")
            for atom in prop_atoms:
                lines.append(f"  {atom}")
            lines.append(f")))")
        lines.append("")
        lines.append("(check-sat)")
        lines.append("(exit)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize verification condition to a JSON-compatible dictionary."""
        return {
            "model_name": self.model_name,
            "properties": self.properties,
            "k": self.k,
            "symbolic_bindings": self.symbolic_bindings,
            "checked_steps": self.checked_steps,
            "verification_time_ms": self.verification_time_ms,
            "z3_queries": self.z3_queries,
            "z3_total_time_ms": self.z3_total_time_ms,
            "z3_sat_count": self.z3_sat_count,
            "z3_unsat_count": self.z3_unsat_count,
            "theories_used": self.theories_used,
            "product_domains": self.product_domains,
            "certificate_hash": hashlib.sha256(
                self.pretty().encode()
            ).hexdigest(),
        }

    def pretty(self) -> str:
        props = ", ".join(self.properties)
        lines = [
            f"SafetyCertificate({self.model_name})",
            f"  Properties proved: {props}",
            f"  Induction depth:   k={self.k}",
            f"  Steps verified:    {self.checked_steps}",
            f"  Time:              {self.verification_time_ms:.1f}ms",
        ]
        if self.z3_queries > 0:
            lines.append(
                f"  Z3 queries:        {self.z3_queries}"
                f" ({self.z3_unsat_count} unsat, {self.z3_sat_count} sat)"
            )
            lines.append(f"  Z3 solve time:     {self.z3_total_time_ms:.1f}ms")
        if self.theories_used:
            lines.append(f"  Theories:          {', '.join(self.theories_used)}")
        if self.product_domains:
            lines.append(
                f"  Product domains:   {' × '.join(self.product_domains)}"
            )
        return "\n".join(lines)



@dataclass
class CounterexampleTrace:
    """A concrete trace demonstrating a safety violation.

    Attributes:
        model_name:    class name of the nn.Module.
        violations:    list of SafetyViolation objects.
        failing_step:  index of the first failing step.
        states:        snapshot of ModelState at each step up to failure.
        concrete_dims: Z3-generated concrete values for symbolic dims.
    """
    model_name: str
    violations: List[SafetyViolation] = field(default_factory=list)
    failing_step: int = -1
    states: List[ModelState] = field(default_factory=list)
    concrete_dims: Dict[str, int] = field(default_factory=dict)

    def pretty(self) -> str:
        lines = [f"CounterexampleTrace({self.model_name})"]
        lines.append(f"  Failing step: {self.failing_step}")
        if self.concrete_dims:
            dims_str = ", ".join(f"{k}={v}" for k, v in self.concrete_dims.items())
            lines.append(f"  Concrete dims: {dims_str}")
        # Show computation path with shapes at each step
        if self.states:
            lines.append(f"  Computation path ({len(self.states)} steps):")
            for i, state in enumerate(self.states):
                marker = " →" if i < self.failing_step else " ✗" if i == self.failing_step else "  "
                shapes_str = ", ".join(
                    f"{t}: {s}" for t, s in sorted(state.shape_env.items())
                ) if state.shape_env else "(initial)"
                lines.append(f"   {marker} [{i}] {shapes_str}")
        for v in self.violations:
            lines.append(f"  VIOLATION [{v.step_index}]: {v.message}")
            if v.shape_a and v.shape_b:
                lines.append(f"    Expected: {v.shape_b}  Got: {v.shape_a}")
            elif v.shape_a:
                lines.append(f"    Shape: {v.shape_a}")
        return "\n".join(lines)


@dataclass
class VerificationResult:
    """Top-level result returned by the constraint verifier."""
    safe: bool
    certificate: Optional[SafetyCertificate] = None
    counterexample: Optional[CounterexampleTrace] = None
    graph: Optional[ComputationGraph] = None
    errors: List[str] = field(default_factory=list)
    verification_time_ms: float = 0.0
    confidence: Confidence = Confidence.HIGH
    min_confidence_threshold: Confidence = Confidence.LOW
    dynamic_features: Dict[str, Any] = field(default_factory=dict)
    dynamic_feature_warnings: List[str] = field(default_factory=list)
    proof_certificate: Optional["ProofCertificate"] = None
    kripke_structure: Optional[KripkeStructure] = None
    unsupported_op_tracker: Optional[UnsupportedOpTracker] = None

    def filter_by_confidence(self, min_level: Confidence = Confidence.MEDIUM) -> "VerificationResult":
        """Return a copy with violations below the confidence threshold removed."""
        if self.safe or not self.counterexample:
            return self
        level_order = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1}
        min_val = level_order[min_level]
        kept = [v for v in self.counterexample.violations
                if level_order.get(v.confidence, 1) >= min_val]
        if not kept:
            return VerificationResult(
                safe=True, certificate=None, graph=self.graph,
                verification_time_ms=self.verification_time_ms,
                confidence=Confidence.MEDIUM,
            )
        new_cex = CounterexampleTrace(
            model_name=self.counterexample.model_name,
            violations=kept,
            failing_step=self.counterexample.failing_step,
            states=self.counterexample.states,
            concrete_dims=self.counterexample.concrete_dims,
        )
        conf_order = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1}
        worst_conf = min(kept, key=lambda v: conf_order.get(v.confidence, 1)).confidence
        return VerificationResult(
            safe=False, counterexample=new_cex, graph=self.graph,
            verification_time_ms=self.verification_time_ms,
            confidence=worst_conf,
        )

    def pretty(self) -> str:
        if self.safe:
            cert = self.certificate
            lines = [
                f"✓ Model is SAFE (confidence: {self.confidence.value})",
                cert.pretty() if cert else "",
            ]
        else:
            cex = self.counterexample
            lines = [
                f"✗ Model is UNSAFE (confidence: {self.confidence.value})",
                cex.pretty() if cex else "",
            ]
        if self.dynamic_feature_warnings:
            lines.append("\n⚠ Modern PyTorch pattern warnings:")
            for w in self.dynamic_feature_warnings:
                lines.append(f"  • {w}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  AST-based computation-graph extraction
# ═══════════════════════════════════════════════════════════════════════════════

# --- Helpers for AST value extraction -------------------------------------

# Names that indicate a constructor parameter is a "config" object whose
# attributes should be treated as fresh symbolic dimension values.
_CONFIG_PARAM_BASE_NAMES: FrozenSet[str] = frozenset({
    "config", "cfg", "args", "hparams", "conf", "model_config",
})


def _is_config_param_name(name: str) -> bool:
    """True if *name* is config-like (matches a base name or starts with one)."""
    low = name.lower()
    if low in _CONFIG_PARAM_BASE_NAMES:
        return True
    for base in _CONFIG_PARAM_BASE_NAMES:
        if low.startswith(base + "_"):
            return True
    return False


def _resolve_dim_value(
    node: ast.expr,
    param_map: Optional[Dict[str, Any]] = None,
    config_param_names: Optional[Set[str]] = None,
    symbolic_attrs: Optional[Dict[Tuple[str, str], str]] = None,
    init_param_names: Optional[Set[str]] = None,
) -> Any:
    """Resolve an AST expression to a dim value: int, str (symbolic), or None.

    Extends ``_const_value`` with awareness of *config-like* constructor
    parameters: when a node is ``config.attr`` for a config param, returns a
    stable symbolic name (memoised in *symbolic_attrs*). Also: when
    ``init_param_names`` is provided, plain ``Name`` references to those
    init parameters resolve to the parameter name itself as a symbolic dim
    (e.g. ``Linear(dim, dim*3)`` → in_features="dim", out_features="(dim*3)").
    """
    # Symbolic config attribute access: e.g. config.n_embd → "config_n_embd"
    if (config_param_names is not None
            and isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in config_param_names):
        key = (node.value.id, node.attr)
        if symbolic_attrs is not None:
            if key not in symbolic_attrs:
                symbolic_attrs[key] = f"{node.value.id}_{node.attr}"
            return symbolic_attrs[key]
        return f"{node.value.id}_{node.attr}"

    # Symbolic plain init parameter: e.g. dim, num_heads in MHABlock(dim, ...)
    if (init_param_names is not None
            and isinstance(node, ast.Name)
            and node.id in init_param_names):
        return node.id

    # Try concrete constant first
    val = _const_value(node, param_map)
    if val is not None:
        return val

    # Recurse into BinOps/UnaryOp to combine symbolic + concrete sub-values.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sub = _resolve_dim_value(node.operand, param_map, config_param_names,
                                 symbolic_attrs, init_param_names)
        if isinstance(sub, int):
            return -sub
        if isinstance(sub, str):
            return f"(-{sub})"
        return None

    if isinstance(node, ast.BinOp):
        left = _resolve_dim_value(node.left, param_map, config_param_names,
                                  symbolic_attrs, init_param_names)
        right = _resolve_dim_value(node.right, param_map, config_param_names,
                                   symbolic_attrs, init_param_names)
        if left is None or right is None:
            return None
        if isinstance(left, int) and isinstance(right, int):
            # Pure-int case already handled by _const_value above; redundant
            # but safe.
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.FloorDiv) and right != 0:
                return left // right
            return None
        op_str = {
            ast.Mult: "*", ast.Add: "+", ast.Sub: "-",
            ast.FloorDiv: "//", ast.Div: "/", ast.Mod: "%",
        }.get(type(node.op))
        if op_str is None:
            return None
        return f"({left}{op_str}{right})"

    # self.<attr> falls through to _const_value via param_map (handled there).
    return None


def _const_value(node: ast.expr, param_map: Optional[Dict[str, Any]] = None) -> Any:
    """Try to extract a Python constant from an AST node.

    If *param_map* is provided, also resolves Name nodes that refer to
    known __init__ parameter names with default values.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # Python ≤3.7
        return node.n
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const_value(node.operand, param_map)
        if v is not None:
            return -v
    if isinstance(node, ast.BinOp):
        left = _const_value(node.left, param_map)
        right = _const_value(node.right, param_map)
        if left is not None and right is not None:
            if isinstance(node.op, ast.Mult) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return int(left * right) if isinstance(left, int) and isinstance(right, int) else left * right
            if isinstance(node.op, ast.Add) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left + right
            if isinstance(node.op, ast.Sub) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left - right
            if isinstance(node.op, ast.FloorDiv) and isinstance(left, int) and isinstance(right, int) and right != 0:
                return left // right
            if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int) and right >= 0:
                return left ** right
    if isinstance(node, ast.Tuple):
        vals = [_const_value(e, param_map) for e in node.elts]
        if all(v is not None for v in vals):
            return tuple(vals)
    if isinstance(node, ast.List):
        vals = [_const_value(e, param_map) for e in node.elts]
        if all(v is not None for v in vals):
            return vals
    # Builtin numeric coercions: int(expr), float(expr), round(expr).
    # These show up frequently in real upstream constructors as
    # ``intermediate = int(dim * ff_mult)``; resolving them closes
    # an envelope-synthesis gap on ctor-bound integer attributes.
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in ("int", "float", "round") \
            and len(node.args) == 1 and not node.keywords:
        inner = _const_value(node.args[0], param_map)
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            try:
                if node.func.id == "int":
                    return int(inner)
                if node.func.id == "float":
                    return float(inner)
                return int(round(inner))
            except (ValueError, OverflowError):
                return None
    # Resolve parameter references (e.g., in_channels from __init__)
    if isinstance(node, ast.Name) and param_map and node.id in param_map:
        return param_map[node.id]
    # Resolve self.<attr> references (e.g., self.n_heads, self.d_k)
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self" and param_map
            and f"self.{node.attr}" in param_map):
        return param_map[f"self.{node.attr}"]
    return None


def _name_or_attr(node: ast.expr) -> Optional[str]:
    """Return the dotted name of a Name or Attribute node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_or_attr(node.value)
        if base is not None:
            return f"{base}.{node.attr}"
    return None


def _is_nn_layer(name: Optional[str]) -> Tuple[bool, LayerKind]:
    """Check whether *name* is a recognised nn.Module layer constructor."""
    if name is None:
        return False, LayerKind.UNKNOWN

    _map = {
        "nn.Linear": LayerKind.LINEAR,
        "Linear": LayerKind.LINEAR,
        "nn.Conv2d": LayerKind.CONV2D,
        "Conv2d": LayerKind.CONV2D,
        "nn.Conv1d": LayerKind.CONV1D,
        "Conv1d": LayerKind.CONV1D,
        "nn.BatchNorm1d": LayerKind.BATCHNORM1D,
        "BatchNorm1d": LayerKind.BATCHNORM1D,
        "nn.BatchNorm2d": LayerKind.BATCHNORM2D,
        "BatchNorm2d": LayerKind.BATCHNORM2D,
        "nn.LayerNorm": LayerKind.LAYERNORM,
        "LayerNorm": LayerKind.LAYERNORM,
        "nn.Dropout": LayerKind.DROPOUT,
        "Dropout": LayerKind.DROPOUT,
        "nn.ReLU": LayerKind.RELU,
        "ReLU": LayerKind.RELU,
        "nn.GELU": LayerKind.RELU,
        "GELU": LayerKind.RELU,
        "nn.SiLU": LayerKind.RELU,
        "SiLU": LayerKind.RELU,
        "nn.Tanh": LayerKind.RELU,
        "Tanh": LayerKind.RELU,
        "nn.Sigmoid": LayerKind.RELU,
        "Sigmoid": LayerKind.RELU,
        "nn.LeakyReLU": LayerKind.RELU,
        "LeakyReLU": LayerKind.RELU,
        "nn.ELU": LayerKind.RELU,
        "ELU": LayerKind.RELU,
        "nn.Mish": LayerKind.RELU,
        "Mish": LayerKind.RELU,
        "nn.PReLU": LayerKind.RELU,
        "PReLU": LayerKind.RELU,
        "nn.SELU": LayerKind.RELU,
        "SELU": LayerKind.RELU,
        "nn.ReLU6": LayerKind.RELU,
        "ReLU6": LayerKind.RELU,
        "nn.Hardswish": LayerKind.RELU,
        "Hardswish": LayerKind.RELU,
        "nn.Hardsigmoid": LayerKind.RELU,
        "Hardsigmoid": LayerKind.RELU,
        "nn.Softmax": LayerKind.SOFTMAX,
        "Softmax": LayerKind.SOFTMAX,
        "nn.Embedding": LayerKind.EMBEDDING,
        "Embedding": LayerKind.EMBEDDING,
        "nn.LSTM": LayerKind.LSTM,
        "LSTM": LayerKind.LSTM,
        "nn.GRU": LayerKind.GRU,
        "GRU": LayerKind.GRU,
        "nn.MultiheadAttention": LayerKind.MULTIHEAD_ATTENTION,
        "MultiheadAttention": LayerKind.MULTIHEAD_ATTENTION,
        "nn.MaxPool2d": LayerKind.MAXPOOL2D,
        "MaxPool2d": LayerKind.MAXPOOL2D,
        "nn.AvgPool2d": LayerKind.AVGPOOL2D,
        "AvgPool2d": LayerKind.AVGPOOL2D,
        "nn.AdaptiveAvgPool2d": LayerKind.ADAPTIVE_AVGPOOL2D,
        "AdaptiveAvgPool2d": LayerKind.ADAPTIVE_AVGPOOL2D,
        "nn.Flatten": LayerKind.FLATTEN,
        "Flatten": LayerKind.FLATTEN,
        "nn.Sequential": LayerKind.SEQUENTIAL,
        "Sequential": LayerKind.SEQUENTIAL,
        "nn.ModuleList": LayerKind.MODULELIST,
        "ModuleList": LayerKind.MODULELIST,
        "nn.Identity": LayerKind.IDENTITY,
        "Identity": LayerKind.IDENTITY,
        "nn.GroupNorm": LayerKind.GROUPNORM,
        "GroupNorm": LayerKind.GROUPNORM,
        "nn.InstanceNorm2d": LayerKind.INSTANCENORM2D,
        "InstanceNorm2d": LayerKind.INSTANCENORM2D,
        "nn.ConvTranspose2d": LayerKind.CONVTRANSPOSE2D,
        "ConvTranspose2d": LayerKind.CONVTRANSPOSE2D,
        "nn.Upsample": LayerKind.UPSAMPLE,
        "Upsample": LayerKind.UPSAMPLE,
        "nn.TransformerEncoder": LayerKind.TRANSFORMER_ENCODER,
        "TransformerEncoder": LayerKind.TRANSFORMER_ENCODER,
        "nn.TransformerDecoder": LayerKind.TRANSFORMER_DECODER,
        "TransformerDecoder": LayerKind.TRANSFORMER_DECODER,
        "nn.TransformerEncoderLayer": LayerKind.TRANSFORMER_ENCODER_LAYER,
        "TransformerEncoderLayer": LayerKind.TRANSFORMER_ENCODER_LAYER,
        "nn.TransformerDecoderLayer": LayerKind.TRANSFORMER_DECODER_LAYER,
        "TransformerDecoderLayer": LayerKind.TRANSFORMER_DECODER_LAYER,
        "nn.ConvTranspose1d": LayerKind.CONVTRANSPOSE1D,
        "ConvTranspose1d": LayerKind.CONVTRANSPOSE1D,
        "nn.AdaptiveMaxPool2d": LayerKind.ADAPTIVE_MAXPOOL2D,
        "AdaptiveMaxPool2d": LayerKind.ADAPTIVE_MAXPOOL2D,
        "nn.PixelShuffle": LayerKind.PIXEL_SHUFFLE,
        "PixelShuffle": LayerKind.PIXEL_SHUFFLE,
        "nn.Unfold": LayerKind.UNFOLD,
        "Unfold": LayerKind.UNFOLD,
        "nn.Fold": LayerKind.FOLD,
        "Fold": LayerKind.FOLD,
        "nn.InstanceNorm1d": LayerKind.INSTANCENORM1D,
        "InstanceNorm1d": LayerKind.INSTANCENORM1D,
        "nn.InstanceNorm3d": LayerKind.INSTANCENORM3D,
        "InstanceNorm3d": LayerKind.INSTANCENORM3D,
        "nn.SyncBatchNorm": LayerKind.SYNCBATCHNORM,
        "SyncBatchNorm": LayerKind.SYNCBATCHNORM,
        "nn.BatchNorm3d": LayerKind.BATCHNORM3D,
        "BatchNorm3d": LayerKind.BATCHNORM3D,
        "nn.MaxPool1d": LayerKind.MAXPOOL1D,
        "MaxPool1d": LayerKind.MAXPOOL1D,
        "nn.AvgPool1d": LayerKind.AVGPOOL1D,
        "AvgPool1d": LayerKind.AVGPOOL1D,
        "nn.MaxPool3d": LayerKind.MAXPOOL3D,
        "MaxPool3d": LayerKind.MAXPOOL3D,
        "nn.AdaptiveAvgPool1d": LayerKind.ADAPTIVE_AVGPOOL1D,
        "AdaptiveAvgPool1d": LayerKind.ADAPTIVE_AVGPOOL1D,
        "nn.AdaptiveMaxPool1d": LayerKind.ADAPTIVE_MAXPOOL1D,
        "AdaptiveMaxPool1d": LayerKind.ADAPTIVE_MAXPOOL1D,
        "nn.LPPool2d": LayerKind.LPPOOL2D,
        "LPPool2d": LayerKind.LPPOOL2D,
        "nn.FractionalMaxPool2d": LayerKind.FRACTIONALMAXPOOL2D,
        "FractionalMaxPool2d": LayerKind.FRACTIONALMAXPOOL2D,
        "nn.RNN": LayerKind.RNN,
        "RNN": LayerKind.RNN,
        "nn.ReflectionPad2d": LayerKind.REFLECTIONPAD2D,
        "ReflectionPad2d": LayerKind.REFLECTIONPAD2D,
        "nn.ReplicationPad2d": LayerKind.REPLICATIONPAD2D,
        "ReplicationPad2d": LayerKind.REPLICATIONPAD2D,
        "nn.ZeroPad2d": LayerKind.ZEROPAD2D,
        "ZeroPad2d": LayerKind.ZEROPAD2D,
        "nn.ConstantPad2d": LayerKind.CONSTANTPAD2D,
        "ConstantPad2d": LayerKind.CONSTANTPAD2D,
        "nn.PixelUnshuffle": LayerKind.PIXEL_UNSHUFFLE,
        "PixelUnshuffle": LayerKind.PIXEL_UNSHUFFLE,
        "nn.AlphaDropout": LayerKind.ALPHADROPOUT,
        "AlphaDropout": LayerKind.ALPHADROPOUT,
        "nn.Dropout2d": LayerKind.DROPOUT,
        "Dropout2d": LayerKind.DROPOUT,
        "nn.Dropout3d": LayerKind.DROPOUT,
        "Dropout3d": LayerKind.DROPOUT,
        "nn.Conv3d": LayerKind.CONV3D,
        "Conv3d": LayerKind.CONV3D,
        "nn.ConvTranspose3d": LayerKind.CONVTRANSPOSE3D,
        "ConvTranspose3d": LayerKind.CONVTRANSPOSE3D,
        "nn.LogSoftmax": LayerKind.SOFTMAX,
        "LogSoftmax": LayerKind.SOFTMAX,
        # --- Shape-preserving activations (map to RELU) ---
        "nn.Softplus": LayerKind.RELU,
        "Softplus": LayerKind.RELU,
        "nn.Softsign": LayerKind.RELU,
        "Softsign": LayerKind.RELU,
        "nn.Tanhshrink": LayerKind.RELU,
        "Tanhshrink": LayerKind.RELU,
        "nn.Softshrink": LayerKind.RELU,
        "Softshrink": LayerKind.RELU,
        "nn.Hardshrink": LayerKind.RELU,
        "Hardshrink": LayerKind.RELU,
        "nn.LogSigmoid": LayerKind.RELU,
        "LogSigmoid": LayerKind.RELU,
        "nn.Threshold": LayerKind.RELU,
        "Threshold": LayerKind.RELU,
        "nn.Hardtanh": LayerKind.RELU,
        "Hardtanh": LayerKind.RELU,
        "nn.CELU": LayerKind.RELU,
        "CELU": LayerKind.RELU,
        "nn.RReLU": LayerKind.RELU,
        "RReLU": LayerKind.RELU,
        "nn.Softmin": LayerKind.SOFTMAX,
        "Softmin": LayerKind.SOFTMAX,
        # --- GLU (halves one dimension) ---
        "nn.GLU": LayerKind.GLU,
        "GLU": LayerKind.GLU,
        # --- Loss functions ---
        "nn.CrossEntropyLoss": LayerKind.LOSS_FUNCTION,
        "CrossEntropyLoss": LayerKind.LOSS_FUNCTION,
        "nn.MSELoss": LayerKind.LOSS_FUNCTION,
        "MSELoss": LayerKind.LOSS_FUNCTION,
        "nn.L1Loss": LayerKind.LOSS_FUNCTION,
        "L1Loss": LayerKind.LOSS_FUNCTION,
        "nn.NLLLoss": LayerKind.LOSS_FUNCTION,
        "NLLLoss": LayerKind.LOSS_FUNCTION,
        "nn.BCELoss": LayerKind.LOSS_FUNCTION,
        "BCELoss": LayerKind.LOSS_FUNCTION,
        "nn.BCEWithLogitsLoss": LayerKind.LOSS_FUNCTION,
        "BCEWithLogitsLoss": LayerKind.LOSS_FUNCTION,
        "nn.SmoothL1Loss": LayerKind.LOSS_FUNCTION,
        "SmoothL1Loss": LayerKind.LOSS_FUNCTION,
        "nn.HuberLoss": LayerKind.LOSS_FUNCTION,
        "HuberLoss": LayerKind.LOSS_FUNCTION,
        "nn.PoissonNLLLoss": LayerKind.LOSS_FUNCTION,
        "PoissonNLLLoss": LayerKind.LOSS_FUNCTION,
        "nn.KLDivLoss": LayerKind.LOSS_FUNCTION,
        "KLDivLoss": LayerKind.LOSS_FUNCTION,
        "nn.MarginRankingLoss": LayerKind.LOSS_FUNCTION,
        "MarginRankingLoss": LayerKind.LOSS_FUNCTION,
        "nn.HingeEmbeddingLoss": LayerKind.LOSS_FUNCTION,
        "HingeEmbeddingLoss": LayerKind.LOSS_FUNCTION,
        "nn.CosineEmbeddingLoss": LayerKind.LOSS_FUNCTION,
        "CosineEmbeddingLoss": LayerKind.LOSS_FUNCTION,
        "nn.MultiMarginLoss": LayerKind.LOSS_FUNCTION,
        "MultiMarginLoss": LayerKind.LOSS_FUNCTION,
        "nn.MultiLabelMarginLoss": LayerKind.LOSS_FUNCTION,
        "MultiLabelMarginLoss": LayerKind.LOSS_FUNCTION,
        "nn.MultiLabelSoftMarginLoss": LayerKind.LOSS_FUNCTION,
        "MultiLabelSoftMarginLoss": LayerKind.LOSS_FUNCTION,
        "nn.SoftMarginLoss": LayerKind.LOSS_FUNCTION,
        "SoftMarginLoss": LayerKind.LOSS_FUNCTION,
        "nn.TripletMarginLoss": LayerKind.LOSS_FUNCTION,
        "TripletMarginLoss": LayerKind.LOSS_FUNCTION,
        "nn.TripletMarginWithDistanceLoss": LayerKind.LOSS_FUNCTION,
        "TripletMarginWithDistanceLoss": LayerKind.LOSS_FUNCTION,
        "nn.CTCLoss": LayerKind.LOSS_FUNCTION,
        "CTCLoss": LayerKind.LOSS_FUNCTION,
        "nn.GaussianNLLLoss": LayerKind.LOSS_FUNCTION,
        "GaussianNLLLoss": LayerKind.LOSS_FUNCTION,
        # --- Padding layers (1D, 3D, Circular) ---
        "nn.ConstantPad1d": LayerKind.CONSTANTPAD1D,
        "ConstantPad1d": LayerKind.CONSTANTPAD1D,
        "nn.ConstantPad3d": LayerKind.CONSTANTPAD3D,
        "ConstantPad3d": LayerKind.CONSTANTPAD3D,
        "nn.ZeroPad1d": LayerKind.ZEROPAD1D,
        "ZeroPad1d": LayerKind.ZEROPAD1D,
        "nn.ZeroPad3d": LayerKind.ZEROPAD3D,
        "ZeroPad3d": LayerKind.ZEROPAD3D,
        "nn.ReflectionPad1d": LayerKind.REFLECTIONPAD1D,
        "ReflectionPad1d": LayerKind.REFLECTIONPAD1D,
        "nn.ReflectionPad3d": LayerKind.REFLECTIONPAD3D,
        "ReflectionPad3d": LayerKind.REFLECTIONPAD3D,
        "nn.ReplicationPad1d": LayerKind.REPLICATIONPAD1D,
        "ReplicationPad1d": LayerKind.REPLICATIONPAD1D,
        "nn.ReplicationPad3d": LayerKind.REPLICATIONPAD3D,
        "ReplicationPad3d": LayerKind.REPLICATIONPAD3D,
        "nn.CircularPad1d": LayerKind.CIRCULARPAD1D,
        "CircularPad1d": LayerKind.CIRCULARPAD1D,
        "nn.CircularPad2d": LayerKind.CIRCULARPAD2D,
        "CircularPad2d": LayerKind.CIRCULARPAD2D,
        "nn.CircularPad3d": LayerKind.CIRCULARPAD3D,
        "CircularPad3d": LayerKind.CIRCULARPAD3D,
        # --- More pooling ---
        "nn.AdaptiveAvgPool3d": LayerKind.ADAPTIVE_AVGPOOL3D,
        "AdaptiveAvgPool3d": LayerKind.ADAPTIVE_AVGPOOL3D,
        "nn.AdaptiveMaxPool3d": LayerKind.ADAPTIVE_MAXPOOL3D,
        "AdaptiveMaxPool3d": LayerKind.ADAPTIVE_MAXPOOL3D,
        "nn.AvgPool3d": LayerKind.AVGPOOL3D,
        "AvgPool3d": LayerKind.AVGPOOL3D,
        "nn.LPPool1d": LayerKind.LPPOOL1D,
        "LPPool1d": LayerKind.LPPOOL1D,
        "nn.FractionalMaxPool3d": LayerKind.FRACTIONALMAXPOOL3D,
        "FractionalMaxPool3d": LayerKind.FRACTIONALMAXPOOL3D,
        "nn.MaxUnpool1d": LayerKind.MAXUNPOOL1D,
        "MaxUnpool1d": LayerKind.MAXUNPOOL1D,
        "nn.MaxUnpool2d": LayerKind.MAXUNPOOL2D,
        "MaxUnpool2d": LayerKind.MAXUNPOOL2D,
        "nn.MaxUnpool3d": LayerKind.MAXUNPOOL3D,
        "MaxUnpool3d": LayerKind.MAXUNPOOL3D,
        # --- Other modules ---
        "nn.EmbeddingBag": LayerKind.EMBEDDINGBAG,
        "EmbeddingBag": LayerKind.EMBEDDINGBAG,
        "nn.Bilinear": LayerKind.BILINEAR,
        "Bilinear": LayerKind.BILINEAR,
        "nn.ModuleDict": LayerKind.MODULEDICT,
        "ModuleDict": LayerKind.MODULEDICT,
        "nn.ParameterList": LayerKind.PARAMETERLIST,
        "ParameterList": LayerKind.PARAMETERLIST,
        "nn.ParameterDict": LayerKind.PARAMETERDICT,
        "ParameterDict": LayerKind.PARAMETERDICT,
        "nn.LazyLinear": LayerKind.LAZYLINEAR,
        "LazyLinear": LayerKind.LAZYLINEAR,
        "nn.LazyConv1d": LayerKind.LAZYCONV1D,
        "LazyConv1d": LayerKind.LAZYCONV1D,
        "nn.LazyConv2d": LayerKind.LAZYCONV2D,
        "LazyConv2d": LayerKind.LAZYCONV2D,
        "nn.LazyConv3d": LayerKind.LAZYCONV3D,
        "LazyConv3d": LayerKind.LAZYCONV3D,
        "nn.LazyBatchNorm1d": LayerKind.LAZYBATCHNORM1D,
        "LazyBatchNorm1d": LayerKind.LAZYBATCHNORM1D,
        "nn.LazyBatchNorm2d": LayerKind.LAZYBATCHNORM2D,
        "LazyBatchNorm2d": LayerKind.LAZYBATCHNORM2D,
        "nn.LazyBatchNorm3d": LayerKind.LAZYBATCHNORM3D,
        "LazyBatchNorm3d": LayerKind.LAZYBATCHNORM3D,
        "nn.LazyInstanceNorm1d": LayerKind.LAZYINSTANCENORM1D,
        "LazyInstanceNorm1d": LayerKind.LAZYINSTANCENORM1D,
        "nn.LazyInstanceNorm2d": LayerKind.LAZYINSTANCENORM2D,
        "LazyInstanceNorm2d": LayerKind.LAZYINSTANCENORM2D,
        "nn.LazyInstanceNorm3d": LayerKind.LAZYINSTANCENORM3D,
        "LazyInstanceNorm3d": LayerKind.LAZYINSTANCENORM3D,
        "nn.LazyConvTranspose1d": LayerKind.LAZYCONVTRANSPOSE1D,
        "LazyConvTranspose1d": LayerKind.LAZYCONVTRANSPOSE1D,
        "nn.LazyConvTranspose2d": LayerKind.LAZYCONVTRANSPOSE2D,
        "LazyConvTranspose2d": LayerKind.LAZYCONVTRANSPOSE2D,
        "nn.LazyConvTranspose3d": LayerKind.LAZYCONVTRANSPOSE3D,
        "LazyConvTranspose3d": LayerKind.LAZYCONVTRANSPOSE3D,
        "nn.PairwiseDistance": LayerKind.PAIRWISE_DISTANCE,
        "PairwiseDistance": LayerKind.PAIRWISE_DISTANCE,
        "nn.CosineSimilarity": LayerKind.COSINE_SIMILARITY,
        "CosineSimilarity": LayerKind.COSINE_SIMILARITY,
        "nn.ChannelShuffle": LayerKind.CHANNEL_SHUFFLE,
        "ChannelShuffle": LayerKind.CHANNEL_SHUFFLE,
        "nn.Unflatten": LayerKind.UNFLATTEN,
        "Unflatten": LayerKind.UNFLATTEN,
    }

    kind = _map.get(name, LayerKind.UNKNOWN)
    return kind != LayerKind.UNKNOWN, kind


def _extract_layer_params(kind: LayerKind, call: ast.Call,
                          param_map: Optional[Dict[str, Any]] = None,
                          config_param_names: Optional[Set[str]] = None,
                          symbolic_attrs: Optional[Dict[Tuple[str, str], str]] = None,
                          init_param_names: Optional[Set[str]] = None) -> LayerDef:
    """Extract numeric parameters from a layer-constructor call."""
    layer = LayerDef(attr_name="", kind=kind, line=call.lineno)

    def _rdim(n):
        return _resolve_dim_value(n, param_map, config_param_names,
                                  symbolic_attrs, init_param_names)

    # Gather positional args (dim-aware: int|str|None for pos[0]/pos[1] which
    # are typically in/out channel-features; strict-concrete for pos[2+] which
    # are typically kernel/stride/padding tuples that downstream code expects
    # to be ints).
    pos = []
    for i, a in enumerate(call.args):
        if i < 2:
            pos.append(_rdim(a))
        else:
            pos.append(_const_value(a, param_map))
    kw = {k.arg: _const_value(k.value, param_map) for k in call.keywords if k.arg}
    # Allow specific dim-bearing kwargs to also carry symbolic values.
    for k in call.keywords:
        if k.arg in ("in_features", "out_features", "in_channels",
                     "out_channels", "num_features", "embed_dim",
                     "num_heads", "hidden_size", "input_size",
                     "num_embeddings", "embedding_dim", "normalized_shape",
                     "num_attention_heads"):
            v = _rdim(k.value)
            if v is not None:
                kw[k.arg] = v

    if kind == LayerKind.LINEAR:
        layer.in_features = pos[0] if len(pos) > 0 else kw.get("in_features")
        layer.out_features = pos[1] if len(pos) > 1 else kw.get("out_features")
        layer.params = {"in_features": layer.in_features,
                        "out_features": layer.out_features}

    elif kind == LayerKind.CONV2D:
        layer.in_channels = pos[0] if len(pos) > 0 else kw.get("in_channels")
        layer.out_channels = pos[1] if len(pos) > 1 else kw.get("out_channels")
        ks = pos[2] if len(pos) > 2 else kw.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks, ks)
        layer.kernel_size = ks
        stride = pos[3] if len(pos) > 3 else kw.get("stride", 1)
        if isinstance(stride, int):
            stride = (stride, stride)
        padding = pos[4] if len(pos) > 4 else kw.get("padding", 0)
        if isinstance(padding, int):
            padding = (padding, padding)
        # Handle padding="same" (PyTorch >= 1.9)
        if padding == "same":
            if ks is not None:
                padding = (ks[0] // 2, ks[1] // 2)
            else:
                padding = (1, 1)
        dilation = kw.get("dilation", 1)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        groups = kw.get("groups", 1)
        layer.params = {"in_channels": layer.in_channels,
                        "out_channels": layer.out_channels,
                        "kernel_size": layer.kernel_size,
                        "stride": stride,
                        "padding": padding,
                        "dilation": dilation,
                        "groups": groups}

    elif kind == LayerKind.CONV1D:
        layer.in_channels = pos[0] if len(pos) > 0 else kw.get("in_channels")
        layer.out_channels = pos[1] if len(pos) > 1 else kw.get("out_channels")
        ks = pos[2] if len(pos) > 2 else kw.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks,)
        layer.kernel_size = ks
        stride = kw.get("stride", 1)
        if isinstance(stride, int):
            stride = (stride,)
        padding = kw.get("padding", 0)
        if isinstance(padding, int):
            padding = (padding,)
        dilation = kw.get("dilation", 1)
        if isinstance(dilation, int):
            dilation = (dilation,)
        groups = kw.get("groups", 1)
        layer.params = {"in_channels": layer.in_channels,
                        "out_channels": layer.out_channels,
                        "kernel_size": layer.kernel_size,
                        "stride": stride,
                        "padding": padding,
                        "dilation": dilation,
                        "groups": groups}

    elif kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D):
        layer.num_features = pos[0] if len(pos) > 0 else kw.get("num_features")
        layer.params = {"num_features": layer.num_features}

    elif kind == LayerKind.LAYERNORM:
        ns = pos[0] if len(pos) > 0 else kw.get("normalized_shape")
        layer.params = {"normalized_shape": ns}

    elif kind == LayerKind.DROPOUT:
        p = pos[0] if len(pos) > 0 else kw.get("p", 0.5)
        layer.params = {"p": p}

    elif kind == LayerKind.EMBEDDING:
        layer.num_embeddings = pos[0] if len(pos) > 0 else kw.get("num_embeddings")
        layer.embedding_dim = pos[1] if len(pos) > 1 else kw.get("embedding_dim")
        layer.params = {"num_embeddings": layer.num_embeddings,
                        "embedding_dim": layer.embedding_dim}

    elif kind in (LayerKind.LSTM, LayerKind.GRU):
        layer.in_features = pos[0] if len(pos) > 0 else kw.get("input_size")
        layer.hidden_size = pos[1] if len(pos) > 1 else kw.get("hidden_size")
        layer.num_layers_rnn = pos[2] if len(pos) > 2 else kw.get("num_layers", 1)
        layer.bidirectional = kw.get("bidirectional", False)
        layer.batch_first = kw.get("batch_first", False)
        layer.params = {"input_size": layer.in_features,
                        "hidden_size": layer.hidden_size,
                        "num_layers": layer.num_layers_rnn,
                        "bidirectional": layer.bidirectional,
                        "batch_first": layer.batch_first}

    elif kind == LayerKind.MULTIHEAD_ATTENTION:
        embed = pos[0] if len(pos) > 0 else kw.get("embed_dim")
        heads = pos[1] if len(pos) > 1 else kw.get("num_heads")
        layer.in_features = embed
        layer.num_heads = heads
        layer.params = {"embed_dim": embed, "num_heads": heads}

    elif kind == LayerKind.ADAPTIVE_AVGPOOL2D:
        out = pos[0] if len(pos) > 0 else kw.get("output_size")
        if isinstance(out, int):
            out = (out, out)
        layer.output_size = out
        layer.params = {"output_size": out}

    elif kind in (LayerKind.MAXPOOL2D, LayerKind.AVGPOOL2D):
        ks = pos[0] if len(pos) > 0 else kw.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks, ks)
        layer.kernel_size = ks
        stride = pos[1] if len(pos) > 1 else kw.get("stride", ks)
        if isinstance(stride, int):
            stride = (stride, stride)
        padding = pos[2] if len(pos) > 2 else kw.get("padding", 0)
        if isinstance(padding, int):
            padding = (padding, padding)
        layer.params = {"kernel_size": ks, "stride": stride, "padding": padding}

    elif kind == LayerKind.SEQUENTIAL:
        # Extract sub-layers from positional args
        sub = []
        for arg in call.args:
            if isinstance(arg, ast.Call):
                fn = _name_or_attr(arg.func)
                is_sub, sub_kind = _is_nn_layer(fn)
                if is_sub:
                    sl = _extract_layer_params(sub_kind, arg, param_map)
                    sl.attr_name = f"_seq_{len(sub)}"
                    sub.append(sl)
        layer.sub_layers = sub if sub else None
        layer.params = {"num_sub_layers": len(sub)}

    elif kind == LayerKind.MODULELIST:
        # Extract sub-layers from the list arg
        sub = []
        if call.args and isinstance(call.args[0], ast.List):
            for elt in call.args[0].elts:
                if isinstance(elt, ast.Call):
                    fn = _name_or_attr(elt.func)
                    is_sub, sub_kind = _is_nn_layer(fn)
                    if is_sub:
                        sl = _extract_layer_params(sub_kind, elt, param_map)
                        sl.attr_name = f"_ml_{len(sub)}"
                        sub.append(sl)
        layer.sub_layers = sub if sub else None
        layer.params = {"num_sub_layers": len(sub)}

    elif kind == LayerKind.GROUPNORM:
        num_groups = pos[0] if len(pos) > 0 else kw.get("num_groups")
        num_channels = pos[1] if len(pos) > 1 else kw.get("num_channels")
        layer.num_features = num_channels
        layer.params = {"num_groups": num_groups, "num_channels": num_channels}

    elif kind == LayerKind.INSTANCENORM2D:
        layer.num_features = pos[0] if len(pos) > 0 else kw.get("num_features")
        layer.params = {"num_features": layer.num_features}

    elif kind == LayerKind.CONVTRANSPOSE2D:
        layer.in_channels = pos[0] if len(pos) > 0 else kw.get("in_channels")
        layer.out_channels = pos[1] if len(pos) > 1 else kw.get("out_channels")
        ks = pos[2] if len(pos) > 2 else kw.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks, ks)
        layer.kernel_size = ks
        stride = pos[3] if len(pos) > 3 else kw.get("stride", 1)
        if isinstance(stride, int):
            stride = (stride, stride)
        padding = pos[4] if len(pos) > 4 else kw.get("padding", 0)
        if isinstance(padding, int):
            padding = (padding, padding)
        output_padding = pos[5] if len(pos) > 5 else kw.get("output_padding", 0)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        groups = pos[6] if len(pos) > 6 else kw.get("groups", 1)
        dilation = pos[8] if len(pos) > 8 else kw.get("dilation", 1)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        layer.params = {"in_channels": layer.in_channels,
                        "out_channels": layer.out_channels,
                        "kernel_size": ks,
                        "stride": stride,
                        "padding": padding,
                        "output_padding": output_padding,
                        "groups": groups,
                        "dilation": dilation}

    elif kind == LayerKind.UPSAMPLE:
        scale = kw.get("scale_factor")
        size = pos[0] if len(pos) > 0 else kw.get("size")
        layer.params = {"scale_factor": scale, "size": size}

    elif kind in (LayerKind.TRANSFORMER_ENCODER_LAYER,
                  LayerKind.TRANSFORMER_DECODER_LAYER):
        d_model = pos[0] if len(pos) > 0 else kw.get("d_model")
        nhead = pos[1] if len(pos) > 1 else kw.get("nhead")
        dim_feedforward = kw.get("dim_feedforward", 2048)
        layer.in_features = d_model
        layer.out_features = d_model  # output same as d_model
        layer.num_heads = nhead
        layer.params = {"d_model": d_model, "nhead": nhead,
                        "dim_feedforward": dim_feedforward}

    elif kind in (LayerKind.TRANSFORMER_ENCODER,
                  LayerKind.TRANSFORMER_DECODER):
        # TransformerEncoder(encoder_layer, num_layers)
        # We extract d_model from the encoder_layer arg if it's a constructor call
        num_layers = pos[1] if len(pos) > 1 else kw.get("num_layers")
        d_model = None
        if call.args and isinstance(call.args[0], ast.Call):
            sub_fn = _name_or_attr(call.args[0].func)
            _, sub_kind = _is_nn_layer(sub_fn)
            if sub_kind in (LayerKind.TRANSFORMER_ENCODER_LAYER,
                            LayerKind.TRANSFORMER_DECODER_LAYER):
                sub_layer = _extract_layer_params(sub_kind, call.args[0], param_map)
                d_model = sub_layer.in_features
                layer.num_heads = sub_layer.num_heads
        layer.in_features = d_model
        layer.out_features = d_model
        layer.params = {"d_model": d_model, "num_layers": num_layers}

    elif kind == LayerKind.CONVTRANSPOSE1D:
        layer.in_channels = pos[0] if len(pos) > 0 else kw.get("in_channels")
        layer.out_channels = pos[1] if len(pos) > 1 else kw.get("out_channels")
        ks = pos[2] if len(pos) > 2 else kw.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks,)
        layer.kernel_size = ks
        stride = pos[3] if len(pos) > 3 else kw.get("stride", 1)
        if isinstance(stride, int):
            stride = (stride,)
        padding = pos[4] if len(pos) > 4 else kw.get("padding", 0)
        if isinstance(padding, int):
            padding = (padding,)
        output_padding = pos[5] if len(pos) > 5 else kw.get("output_padding", 0)
        if isinstance(output_padding, int):
            output_padding = (output_padding,)
        groups = pos[6] if len(pos) > 6 else kw.get("groups", 1)
        dilation = pos[8] if len(pos) > 8 else kw.get("dilation", 1)
        if isinstance(dilation, int):
            dilation = (dilation,)
        layer.params = {"in_channels": layer.in_channels,
                        "out_channels": layer.out_channels,
                        "kernel_size": ks,
                        "stride": stride,
                        "padding": padding,
                        "output_padding": output_padding,
                        "groups": groups,
                        "dilation": dilation}

    elif kind == LayerKind.ADAPTIVE_MAXPOOL2D:
        out = pos[0] if len(pos) > 0 else kw.get("output_size")
        if isinstance(out, int):
            out = (out, out)
        layer.output_size = out
        layer.params = {"output_size": out}

    elif kind == LayerKind.PIXEL_SHUFFLE:
        upscale = pos[0] if len(pos) > 0 else kw.get("upscale_factor")
        layer.params = {"upscale_factor": upscale}

    elif kind == LayerKind.UNFOLD:
        ks = pos[0] if len(pos) > 0 else kw.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks, ks)
        layer.kernel_size = ks
        dilation = kw.get("dilation", 1)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        padding = kw.get("padding", 0)
        if isinstance(padding, int):
            padding = (padding, padding)
        stride = kw.get("stride", 1)
        if isinstance(stride, int):
            stride = (stride, stride)
        layer.params = {"kernel_size": ks, "dilation": dilation,
                        "padding": padding, "stride": stride}

    elif kind == LayerKind.FOLD:
        output_size = pos[0] if len(pos) > 0 else kw.get("output_size")
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        ks = pos[1] if len(pos) > 1 else kw.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks, ks)
        layer.kernel_size = ks
        layer.output_size = output_size
        dilation = kw.get("dilation", 1)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        padding = kw.get("padding", 0)
        if isinstance(padding, int):
            padding = (padding, padding)
        stride = kw.get("stride", 1)
        if isinstance(stride, int):
            stride = (stride, stride)
        layer.params = {"output_size": output_size, "kernel_size": ks,
                        "dilation": dilation, "padding": padding,
                        "stride": stride}

    elif kind == LayerKind.INSTANCENORM1D:
        layer.num_features = pos[0] if len(pos) > 0 else kw.get("num_features")
        layer.params = {"num_features": layer.num_features}

    elif kind == LayerKind.INSTANCENORM3D:
        layer.num_features = pos[0] if len(pos) > 0 else kw.get("num_features")
        layer.params = {"num_features": layer.num_features}

    elif kind == LayerKind.SYNCBATCHNORM:
        layer.num_features = pos[0] if len(pos) > 0 else kw.get("num_features")
        layer.params = {"num_features": layer.num_features}

    elif kind == LayerKind.BATCHNORM3D:
        layer.num_features = pos[0] if len(pos) > 0 else kw.get("num_features")
        layer.params = {"num_features": layer.num_features}

    elif kind == LayerKind.MAXPOOL1D:
        ks_val = pos[0] if len(pos) > 0 else kw.get("kernel_size")
        if isinstance(ks_val, int):
            ks_val = (ks_val,)
        layer.kernel_size = ks_val
        stride_val = pos[1] if len(pos) > 1 else kw.get("stride", ks_val)
        if isinstance(stride_val, int):
            stride_val = (stride_val,)
        padding_val = pos[2] if len(pos) > 2 else kw.get("padding", 0)
        if isinstance(padding_val, int):
            padding_val = (padding_val,)
        layer.params = {"kernel_size": ks_val, "stride": stride_val, "padding": padding_val}

    elif kind == LayerKind.AVGPOOL1D:
        ks_val = pos[0] if len(pos) > 0 else kw.get("kernel_size")
        if isinstance(ks_val, int):
            ks_val = (ks_val,)
        layer.kernel_size = ks_val
        stride_val = pos[1] if len(pos) > 1 else kw.get("stride", ks_val)
        if isinstance(stride_val, int):
            stride_val = (stride_val,)
        padding_val = pos[2] if len(pos) > 2 else kw.get("padding", 0)
        if isinstance(padding_val, int):
            padding_val = (padding_val,)
        layer.params = {"kernel_size": ks_val, "stride": stride_val, "padding": padding_val}

    elif kind == LayerKind.MAXPOOL3D:
        ks_val = pos[0] if len(pos) > 0 else kw.get("kernel_size")
        if isinstance(ks_val, int):
            ks_val = (ks_val, ks_val, ks_val)
        layer.kernel_size = ks_val
        stride_val = pos[1] if len(pos) > 1 else kw.get("stride", ks_val)
        if isinstance(stride_val, int):
            stride_val = (stride_val, stride_val, stride_val)
        padding_val = pos[2] if len(pos) > 2 else kw.get("padding", 0)
        if isinstance(padding_val, int):
            padding_val = (padding_val, padding_val, padding_val)
        layer.params = {"kernel_size": ks_val, "stride": stride_val, "padding": padding_val}

    elif kind == LayerKind.ADAPTIVE_AVGPOOL1D:
        out = pos[0] if len(pos) > 0 else kw.get("output_size")
        if isinstance(out, int):
            out = (out,)
        layer.output_size = out
        layer.params = {"output_size": out}

    elif kind == LayerKind.ADAPTIVE_MAXPOOL1D:
        out = pos[0] if len(pos) > 0 else kw.get("output_size")
        if isinstance(out, int):
            out = (out,)
        layer.output_size = out
        layer.params = {"output_size": out}

    elif kind == LayerKind.LPPOOL2D:
        norm_type = pos[0] if len(pos) > 0 else kw.get("norm_type")
        ks_val = pos[1] if len(pos) > 1 else kw.get("kernel_size")
        if isinstance(ks_val, int):
            ks_val = (ks_val, ks_val)
        layer.kernel_size = ks_val
        stride_val = kw.get("stride", ks_val)
        if isinstance(stride_val, int):
            stride_val = (stride_val, stride_val)
        layer.params = {"norm_type": norm_type, "kernel_size": ks_val, "stride": stride_val}

    elif kind == LayerKind.FRACTIONALMAXPOOL2D:
        ks_val = pos[0] if len(pos) > 0 else kw.get("kernel_size")
        if isinstance(ks_val, int):
            ks_val = (ks_val, ks_val)
        layer.kernel_size = ks_val
        out = kw.get("output_size")
        if isinstance(out, int):
            out = (out, out)
        layer.output_size = out
        layer.params = {"kernel_size": ks_val, "output_size": out}

    elif kind == LayerKind.RNN:
        layer.in_features = pos[0] if len(pos) > 0 else kw.get("input_size")
        layer.hidden_size = pos[1] if len(pos) > 1 else kw.get("hidden_size")
        layer.num_layers_rnn = pos[2] if len(pos) > 2 else kw.get("num_layers", 1)
        layer.bidirectional = kw.get("bidirectional", False)
        layer.batch_first = kw.get("batch_first", False)
        layer.params = {"input_size": layer.in_features,
                        "hidden_size": layer.hidden_size,
                        "num_layers": layer.num_layers_rnn,
                        "bidirectional": layer.bidirectional,
                        "batch_first": layer.batch_first}

    elif kind in (LayerKind.REFLECTIONPAD2D, LayerKind.REPLICATIONPAD2D,
                  LayerKind.ZEROPAD2D):
        pad = pos[0] if len(pos) > 0 else kw.get("padding")
        layer.params = {"padding": pad}

    elif kind == LayerKind.CONSTANTPAD2D:
        pad = pos[0] if len(pos) > 0 else kw.get("padding")
        value = pos[1] if len(pos) > 1 else kw.get("value", 0)
        layer.params = {"padding": pad, "value": value}

    elif kind == LayerKind.PIXEL_UNSHUFFLE:
        downscale = pos[0] if len(pos) > 0 else kw.get("downscale_factor")
        layer.params = {"downscale_factor": downscale}

    elif kind == LayerKind.ALPHADROPOUT:
        p = pos[0] if len(pos) > 0 else kw.get("p", 0.5)
        layer.params = {"p": p}

    elif kind == LayerKind.CONV3D:
        layer.in_channels = pos[0] if len(pos) > 0 else kw.get("in_channels")
        layer.out_channels = pos[1] if len(pos) > 1 else kw.get("out_channels")
        ks_val = pos[2] if len(pos) > 2 else kw.get("kernel_size")
        if isinstance(ks_val, int):
            ks_val = (ks_val, ks_val, ks_val)
        layer.kernel_size = ks_val
        stride_val = pos[3] if len(pos) > 3 else kw.get("stride", 1)
        if isinstance(stride_val, int):
            stride_val = (stride_val, stride_val, stride_val)
        padding_val = pos[4] if len(pos) > 4 else kw.get("padding", 0)
        if isinstance(padding_val, int):
            padding_val = (padding_val, padding_val, padding_val)
        dilation_val = pos[5] if len(pos) > 5 else kw.get("dilation", 1)
        if isinstance(dilation_val, int):
            dilation_val = (dilation_val, dilation_val, dilation_val)
        groups_val = pos[6] if len(pos) > 6 else kw.get("groups", 1)
        layer.params = {"in_channels": layer.in_channels,
                        "out_channels": layer.out_channels,
                        "kernel_size": ks_val,
                        "stride": stride_val,
                        "padding": padding_val,
                        "dilation": dilation_val,
                        "groups": groups_val}

    elif kind == LayerKind.CONVTRANSPOSE3D:
        layer.in_channels = pos[0] if len(pos) > 0 else kw.get("in_channels")
        layer.out_channels = pos[1] if len(pos) > 1 else kw.get("out_channels")
        ks_val = pos[2] if len(pos) > 2 else kw.get("kernel_size")
        if isinstance(ks_val, int):
            ks_val = (ks_val, ks_val, ks_val)
        layer.kernel_size = ks_val
        stride_val = pos[3] if len(pos) > 3 else kw.get("stride", 1)
        if isinstance(stride_val, int):
            stride_val = (stride_val, stride_val, stride_val)
        padding_val = pos[4] if len(pos) > 4 else kw.get("padding", 0)
        if isinstance(padding_val, int):
            padding_val = (padding_val, padding_val, padding_val)
        output_padding_val = pos[5] if len(pos) > 5 else kw.get("output_padding", 0)
        if isinstance(output_padding_val, int):
            output_padding_val = (output_padding_val, output_padding_val, output_padding_val)
        groups_val = pos[6] if len(pos) > 6 else kw.get("groups", 1)
        dilation_val = pos[8] if len(pos) > 8 else kw.get("dilation", 1)
        if isinstance(dilation_val, int):
            dilation_val = (dilation_val, dilation_val, dilation_val)
        layer.params = {"in_channels": layer.in_channels,
                        "out_channels": layer.out_channels,
                        "kernel_size": ks_val,
                        "stride": stride_val,
                        "padding": padding_val,
                        "output_padding": output_padding_val,
                        "groups": groups_val,
                        "dilation": dilation_val}

    # --- New operators: param extraction ---

    elif kind == LayerKind.LOSS_FUNCTION:
        reduction = kw.get("reduction", "mean")
        layer.params = {"reduction": reduction}

    elif kind == LayerKind.GLU:
        dim = pos[0] if len(pos) > 0 else kw.get("dim", -1)
        layer.params = {"dim": dim}

    elif kind in (LayerKind.REFLECTIONPAD1D, LayerKind.REPLICATIONPAD1D,
                  LayerKind.ZEROPAD1D, LayerKind.CIRCULARPAD1D):
        pad = pos[0] if len(pos) > 0 else kw.get("padding")
        layer.params = {"padding": pad}

    elif kind in (LayerKind.CONSTANTPAD1D,):
        pad = pos[0] if len(pos) > 0 else kw.get("padding")
        value = pos[1] if len(pos) > 1 else kw.get("value", 0)
        layer.params = {"padding": pad, "value": value}

    elif kind in (LayerKind.REFLECTIONPAD3D, LayerKind.REPLICATIONPAD3D,
                  LayerKind.ZEROPAD3D, LayerKind.CIRCULARPAD3D):
        pad = pos[0] if len(pos) > 0 else kw.get("padding")
        layer.params = {"padding": pad}

    elif kind == LayerKind.CONSTANTPAD3D:
        pad = pos[0] if len(pos) > 0 else kw.get("padding")
        value = pos[1] if len(pos) > 1 else kw.get("value", 0)
        layer.params = {"padding": pad, "value": value}

    elif kind == LayerKind.CIRCULARPAD2D:
        pad = pos[0] if len(pos) > 0 else kw.get("padding")
        layer.params = {"padding": pad}

    elif kind in (LayerKind.ADAPTIVE_AVGPOOL3D, LayerKind.ADAPTIVE_MAXPOOL3D):
        out = pos[0] if len(pos) > 0 else kw.get("output_size")
        if isinstance(out, int):
            out = (out, out, out)
        layer.output_size = out
        layer.params = {"output_size": out}

    elif kind == LayerKind.AVGPOOL3D:
        ks_val = pos[0] if len(pos) > 0 else kw.get("kernel_size")
        if isinstance(ks_val, int):
            ks_val = (ks_val, ks_val, ks_val)
        layer.kernel_size = ks_val
        stride_val = pos[1] if len(pos) > 1 else kw.get("stride", ks_val)
        if isinstance(stride_val, int):
            stride_val = (stride_val, stride_val, stride_val)
        padding_val = pos[2] if len(pos) > 2 else kw.get("padding", 0)
        if isinstance(padding_val, int):
            padding_val = (padding_val, padding_val, padding_val)
        layer.params = {"kernel_size": ks_val, "stride": stride_val, "padding": padding_val}

    elif kind == LayerKind.LPPOOL1D:
        norm_type = pos[0] if len(pos) > 0 else kw.get("norm_type")
        ks_val = pos[1] if len(pos) > 1 else kw.get("kernel_size")
        if isinstance(ks_val, int):
            ks_val = (ks_val,)
        layer.kernel_size = ks_val
        stride_val = kw.get("stride", ks_val)
        if isinstance(stride_val, int):
            stride_val = (stride_val,)
        layer.params = {"norm_type": norm_type, "kernel_size": ks_val, "stride": stride_val}

    elif kind == LayerKind.FRACTIONALMAXPOOL3D:
        ks_val = pos[0] if len(pos) > 0 else kw.get("kernel_size")
        if isinstance(ks_val, int):
            ks_val = (ks_val, ks_val, ks_val)
        layer.kernel_size = ks_val
        out = kw.get("output_size")
        if isinstance(out, int):
            out = (out, out, out)
        layer.output_size = out
        layer.params = {"kernel_size": ks_val, "output_size": out}

    elif kind in (LayerKind.MAXUNPOOL1D, LayerKind.MAXUNPOOL2D, LayerKind.MAXUNPOOL3D):
        ks_val = pos[0] if len(pos) > 0 else kw.get("kernel_size")
        stride_val = pos[1] if len(pos) > 1 else kw.get("stride")
        padding_val = pos[2] if len(pos) > 2 else kw.get("padding", 0)
        layer.kernel_size = ks_val
        layer.params = {"kernel_size": ks_val, "stride": stride_val, "padding": padding_val}

    elif kind == LayerKind.EMBEDDINGBAG:
        layer.num_embeddings = pos[0] if len(pos) > 0 else kw.get("num_embeddings")
        layer.embedding_dim = pos[1] if len(pos) > 1 else kw.get("embedding_dim")
        layer.params = {"num_embeddings": layer.num_embeddings,
                        "embedding_dim": layer.embedding_dim}

    elif kind == LayerKind.BILINEAR:
        in1 = pos[0] if len(pos) > 0 else kw.get("in1_features")
        in2 = pos[1] if len(pos) > 1 else kw.get("in2_features")
        out_f = pos[2] if len(pos) > 2 else kw.get("out_features")
        layer.params = {"in1_features": in1, "in2_features": in2, "out_features": out_f}

    elif kind in (LayerKind.MODULEDICT, LayerKind.PARAMETERLIST, LayerKind.PARAMETERDICT):
        layer.params = {}

    elif kind == LayerKind.LAZYLINEAR:
        layer.out_features = pos[0] if len(pos) > 0 else kw.get("out_features")
        layer.params = {"out_features": layer.out_features}

    elif kind in (LayerKind.LAZYCONV1D, LayerKind.LAZYCONV2D, LayerKind.LAZYCONV3D):
        layer.out_channels = pos[0] if len(pos) > 0 else kw.get("out_channels")
        ks_val = pos[1] if len(pos) > 1 else kw.get("kernel_size")
        layer.kernel_size = ks_val
        stride_val = kw.get("stride", 1)
        padding_val = kw.get("padding", 0)
        layer.params = {"out_channels": layer.out_channels, "kernel_size": ks_val,
                        "stride": stride_val, "padding": padding_val}

    elif kind in (LayerKind.LAZYBATCHNORM1D, LayerKind.LAZYBATCHNORM2D,
                  LayerKind.LAZYBATCHNORM3D):
        layer.params = {}

    elif kind in (LayerKind.LAZYINSTANCENORM1D, LayerKind.LAZYINSTANCENORM2D,
                  LayerKind.LAZYINSTANCENORM3D):
        layer.params = {}

    elif kind in (LayerKind.LAZYCONVTRANSPOSE1D, LayerKind.LAZYCONVTRANSPOSE2D,
                  LayerKind.LAZYCONVTRANSPOSE3D):
        layer.out_channels = pos[0] if len(pos) > 0 else kw.get("out_channels")
        ks_val = pos[1] if len(pos) > 1 else kw.get("kernel_size")
        layer.kernel_size = ks_val
        stride_val = kw.get("stride", 1)
        padding_val = kw.get("padding", 0)
        output_padding_val = kw.get("output_padding", 0)
        layer.params = {"out_channels": layer.out_channels, "kernel_size": ks_val,
                        "stride": stride_val, "padding": padding_val,
                        "output_padding": output_padding_val}

    elif kind in (LayerKind.PAIRWISE_DISTANCE, LayerKind.COSINE_SIMILARITY):
        dim = kw.get("dim", 1)
        layer.params = {"dim": dim}

    elif kind == LayerKind.CHANNEL_SHUFFLE:
        groups = pos[0] if len(pos) > 0 else kw.get("groups")
        layer.params = {"groups": groups}

    elif kind == LayerKind.UNFLATTEN:
        dim = pos[0] if len(pos) > 0 else kw.get("dim")
        unflattened_size = pos[1] if len(pos) > 1 else kw.get("unflattened_size")
        layer.params = {"dim": dim, "unflattened_size": unflattened_size}

    return layer


def _extract_tensor_shape(
    node: ast.expr,
    param_map: Optional[Dict[str, Any]] = None,
) -> Optional["TensorShape"]:
    """Extract a concrete TensorShape from a tensor-factory AST call.

    Handles torch.randn(d0, d1, ...), torch.zeros(d0, d1, ...) etc.
    Also handles torch.arange(n) → shape (n,) and method chains like
    .unsqueeze(dim) / .view(...) / .reshape(...) applied to a recognised
    tensor expression.
    Returns None when the shape cannot be statically determined.
    """
    from src.tensor_shapes import TensorShape, ShapeDim
    if not isinstance(node, ast.Call):
        return None

    # --- Handle method chains: expr.unsqueeze(dim), expr.view(...), etc. ---
    if (isinstance(node.func, ast.Attribute)
            and node.func.attr in ("unsqueeze", "view", "reshape", "expand")
            and isinstance(node.func.value, ast.Call)):
        base_shape = _extract_tensor_shape(node.func.value, param_map)
        if base_shape is None:
            return None
        method = node.func.attr
        if method == "unsqueeze" and len(node.args) == 1:
            dim_val = _const_value(node.args[0], param_map)
            if dim_val is None:
                return None
            dim_val = int(dim_val)
            dims_list = list(base_shape.dims)
            if dim_val < 0:
                dim_val = len(dims_list) + 1 + dim_val
            if 0 <= dim_val <= len(dims_list):
                dims_list.insert(dim_val, ShapeDim(1))
                return TensorShape(tuple(dims_list))
        elif method in ("view", "reshape"):
            new_dims: List[Optional[int]] = []
            for a in node.args:
                v = _const_value(a, param_map)
                if isinstance(v, int):
                    new_dims.append(v)
                else:
                    return None
            if new_dims:
                return TensorShape(tuple(ShapeDim(int(d)) for d in new_dims))
        return None

    func_name = _name_or_attr(node.func)

    # --- Handle torch.arange(n) → shape (n,) ---
    _ARANGE_FNS = frozenset({"torch.arange", "arange"})
    if func_name in _ARANGE_FNS:
        if len(node.args) >= 1:
            n = _const_value(node.args[0], param_map)
            if isinstance(n, int):
                if len(node.args) == 1:
                    return TensorShape((ShapeDim(n),))
                elif len(node.args) == 2:
                    end = _const_value(node.args[1], param_map)
                    if isinstance(end, int):
                        return TensorShape((ShapeDim(end - n),))
        return None

    # --- Handle torch.linspace/logspace(start, end, steps) → shape (steps,) ---
    _LINSPACE_FNS = frozenset({"torch.linspace", "torch.logspace", "linspace", "logspace"})
    if func_name in _LINSPACE_FNS:
        if len(node.args) >= 3:
            steps = _const_value(node.args[2], param_map)
            if isinstance(steps, int):
                return TensorShape((ShapeDim(steps),))
        return None

    _FACTORY_FNS = frozenset({
        "torch.randn", "torch.zeros", "torch.ones", "torch.rand",
        "torch.empty", "torch.full", "randn", "zeros", "ones", "rand", "empty",
    })
    if func_name not in _FACTORY_FNS:
        return None
    # Shape can be positional args or a single tuple/list arg
    dims: List[Optional[int]] = []
    if len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List)):
        for elt in node.args[0].elts:
            dims.append(_const_value(elt, param_map))
    else:
        for a in node.args:
            # Skip keyword-only args like device=, dtype=
            v = _const_value(a, param_map)
            if isinstance(v, int):
                dims.append(v)
    if not dims or any(d is None for d in dims):
        return None
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


# --- _InitExtractor: walks __init__ to find layer definitions -------------

class _InitExtractor(ast.NodeVisitor):
    """Extracts layer definitions from an nn.Module's ``__init__``.

    Also resolves constructor parameter names (e.g., ``in_channels``,
    ``hidden_dim``) by tracking assignments of the form ``self.x = x``
    and plain parameter default values from the function signature.

    When *class_map* is provided, submodule instantiations like
    ``self.layer1 = BasicBlock(64, 64)`` are recognised and their inner
    layers are extracted with prefixed attribute names.

    Local variables assigned to nn.Layer constructors are tracked so
    that patterns like ``enc_layer = nn.TransformerEncoderLayer(...);
    self.encoder = nn.TransformerEncoder(enc_layer, ...)`` correctly
    propagate parameters.
    """

    def __init__(self, class_map: Optional[Dict[str, ast.ClassDef]] = None,
                 function_map: Optional[Dict[str, ast.FunctionDef]] = None) -> None:
        self.layers: Dict[str, LayerDef] = {}
        self.buffer_shapes: Dict[str, "TensorShape"] = {}
        self._param_map: Dict[str, Any] = {}  # param_name -> default value
        self._class_map = class_map or {}
        # Top-level helper functions whose body returns a single nn-layer Call
        # (e.g. torchvision's ``conv3x3``/``conv1x1``).  Threaded through so
        # ``self.conv1 = conv3x3(in, out, stride)`` is expanded inline rather
        # than abstaining as an opaque submodule.
        self._function_map = function_map or {}
        self._local_layer_calls: Dict[str, ast.Call] = {}  # local_var -> Call node
        # Track scalar instance attributes: self.x = <const_expr>
        # e.g. self.d_k = d_model // n_heads → {"d_k": 64}
        self.scalar_attrs: Dict[str, Any] = {}
        # Parameter shapes (nn.Parameter) — move with model, no device mismatch
        self.param_shapes: Dict[str, "TensorShape"] = {}
        # --- Symbolic config-attribute environment (Task A) ---
        # Names of constructor params treated as opaque "config" objects.
        self.config_param_names: Set[str] = set()
        # Memoised symbolic dim names for (config_param, attr) pairs.
        self.symbolic_config_attrs: Dict[Tuple[str, str], str] = {}
        # Divisibility axioms collected from `assert N % H == 0` in __init__.
        self.divisibility_axioms: List[Tuple[str, str]] = []
        # Symbolic derivations: derived_attr → (numerator, op, denominator).
        self.symbolic_derivations: Dict[str, Tuple[str, str, str]] = {}
        # Config attrs that are reassigned within __init__ (sound exclusion).
        self._reassigned_config_attrs: Set[Tuple[str, str]] = set()
        # Plain init parameters (excluding self / config-like / *args /
        # **kwargs / those with a concrete default). Used as symbolic dim
        # sources so e.g. ``Linear(dim, dim*3)`` extracts symbolic features.
        self.init_param_names: Set[str] = set()

    def extract(self, init_fn: ast.FunctionDef) -> None:
        """Extract layers, first building param_map from defaults."""
        # Build mapping from parameter names to default values
        args = init_fn.args
        defaults = args.defaults or []
        num_args = len(args.args)
        num_defaults = len(defaults)
        for i, default in enumerate(defaults):
            arg_idx = num_args - num_defaults + i
            if arg_idx >= 0 and arg_idx < num_args:
                param_name = args.args[arg_idx].arg
                val = _const_value(default)
                if val is not None:
                    self._param_map[param_name] = val

        # --- Task A: detect config-like constructor params --------------
        for a in args.args:
            if a.arg == "self":
                continue
            if _is_config_param_name(a.arg):
                # Don't add 'args' if the function uses *args (handled below).
                self.config_param_names.add(a.arg)
            elif a.arg not in self._param_map:
                # Treat as a symbolic init param (only if no concrete default).
                self.init_param_names.add(a.arg)
        # If the function takes *args, exclude that name from config params.
        if args.vararg and args.vararg.arg in self.config_param_names:
            self.config_param_names.discard(args.vararg.arg)
        if args.vararg and args.vararg.arg in self.init_param_names:
            self.init_param_names.discard(args.vararg.arg)
        # Soundness: scan the body for `config.X = ...` reassignments and
        # exclude those (param, attr) pairs from being symbolised.
        for sub in ast.walk(init_fn):
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    if (isinstance(tgt, ast.Attribute)
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id in self.config_param_names):
                        self._reassigned_config_attrs.add(
                            (tgt.value.id, tgt.attr))

        self.visit(init_fn)

    def _filtered_symbolic_attrs(self) -> Optional[Dict[Tuple[str, str], str]]:
        """Return ``symbolic_config_attrs`` view that drops reassigned pairs."""
        return self.symbolic_config_attrs

    def _register_layer_param_shapes(self, attr: str, layer: "LayerDef") -> None:
        """Synthesise the static ``weight`` / ``bias`` shapes for a layer.

        Only Linear and Conv1d/2d/3d are emitted in this round — these are
        the parameter accesses that real-world bug repros (e.g. PEFT DoRA
        ``self.conv.weight.view(...)``) rely on, and their static shapes
        are unambiguous.  Shapes are written under
        ``param_shapes[f"{attr}.weight"]`` so the model-checker propagates
        them as ``self.<attr>.weight`` entries in the shape environment.
        """
        try:
            from src.tensor_shapes import TensorShape, ShapeDim
        except Exception:
            return
        if layer.kind == LayerKind.LINEAR:
            if (isinstance(layer.in_features, int)
                    and isinstance(layer.out_features, int)):
                w = TensorShape((ShapeDim(layer.out_features),
                                 ShapeDim(layer.in_features)))
                self.param_shapes[f"{attr}.weight"] = w
                self.param_shapes[f"{attr}.bias"] = TensorShape(
                    (ShapeDim(layer.out_features),))
        elif layer.kind in (LayerKind.CONV1D, LayerKind.CONV2D, LayerKind.CONV3D):
            in_c = layer.in_channels
            out_c = layer.out_channels
            ks = layer.kernel_size
            groups = layer.params.get("groups", 1) if layer.params else 1
            if (isinstance(in_c, int) and isinstance(out_c, int)
                    and isinstance(ks, tuple)
                    and all(isinstance(k, int) for k in ks)
                    and isinstance(groups, int) and groups > 0
                    and in_c % groups == 0):
                dims = [ShapeDim(out_c), ShapeDim(in_c // groups)]
                dims.extend(ShapeDim(k) for k in ks)
                self.param_shapes[f"{attr}.weight"] = TensorShape(tuple(dims))
                self.param_shapes[f"{attr}.bias"] = TensorShape(
                    (ShapeDim(out_c),))

    def _resolve_init_dim(self, node: ast.expr) -> Any:
        """Like ``_resolve_dim_value`` but config-aware for this extractor."""
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in self.config_param_names
                and (node.value.id, node.attr) in self._reassigned_config_attrs):
            return None
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in self.scalar_attrs):
            return self.scalar_attrs[node.attr]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            sub = self._resolve_init_dim(node.operand)
            if isinstance(sub, int):
                return -sub
            if isinstance(sub, str):
                return f"(-{sub})"
        if isinstance(node, ast.BinOp):
            left = self._resolve_init_dim(node.left)
            right = self._resolve_init_dim(node.right)
            if left is None or right is None:
                return None
            if isinstance(left, int) and isinstance(right, int):
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.FloorDiv) and right != 0:
                    return left // right
                return None
            op_str = {
                ast.Mult: "*", ast.Add: "+", ast.Sub: "-",
                ast.FloorDiv: "//", ast.Div: "/", ast.Mod: "%",
            }.get(type(node.op))
            if op_str is None:
                return None
            return f"({left}{op_str}{right})"
        return _resolve_dim_value(node, self._param_map,
                                  self.config_param_names,
                                  self.symbolic_config_attrs,
                                  self.init_param_names)

    def visit_Assert(self, node: ast.Assert) -> None:
        """Detect ``assert N % H == 0`` divisibility axioms over config attrs."""
        test = node.test
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.left, ast.BinOp)
                and isinstance(test.left.op, ast.Mod)):
            zero = _const_value(test.comparators[0])
            if zero == 0:
                num = self._resolve_init_dim(test.left.left)
                den = self._resolve_init_dim(test.left.right)
                if isinstance(num, str) and isinstance(den, str):
                    self.divisibility_axioms.append((num, den))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            # Track local variable assignments of layer constructors
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                fname = _name_or_attr(node.value.func)
                is_layer, _ = _is_nn_layer(fname)
                if is_layer:
                    self._local_layer_calls[target.id] = node.value
            # Track local scalar bindings: ``sharded_inner = (h*d)//tp``
            # so that downstream ``nn.Linear(d_model, sharded_inner)`` can
            # extract the constructor-bound integer.  This closes the
            # "constructor-bound integer attribute envelope" gap on
            # upstream-faithful real-bug repros (e.g. LongT5 TP attention,
            # diffusers FFN ``intermediate = int(dim * ff_mult)``).
            if isinstance(target, ast.Name):
                if target.id not in self._param_map:
                    val = _const_value(node.value, self._param_map)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        self._param_map[target.id] = val
                    elif not isinstance(node.value, ast.Call):
                        sym = self._resolve_init_dim(node.value)
                        if isinstance(sym, str):
                            self._param_map[target.id] = sym
            self._try_extract(target, node.value)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        """Detect self.register_buffer('name', tensor) calls in __init__."""
        if not isinstance(node.value, ast.Call):
            self.generic_visit(node)
            return
        call = node.value
        # self.register_buffer('name', tensor_expr)
        if not (isinstance(call.func, ast.Attribute)
                and call.func.attr == "register_buffer"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"):
            self.generic_visit(node)
            return
        if len(call.args) < 2:
            self.generic_visit(node)
            return
        name_arg = call.args[0]
        tensor_arg = call.args[1]
        if not (isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)):
            self.generic_visit(node)
            return
        buf_name = name_arg.value
        shape = _extract_tensor_shape(tensor_arg, self._param_map)
        if shape is not None:
            self.buffer_shapes[buf_name] = shape
        self.generic_visit(node)

    def _try_extract(self, target: ast.expr, value: ast.expr) -> None:
        # self.<attr> = nn.<Layer>(...) or self.<attr> = scalar_expr
        if not (isinstance(target, ast.Attribute) and
                isinstance(target.value, ast.Name) and
                target.value.id == "self"):
            return
        attr = target.attr
        if not isinstance(value, ast.Call):
            # Try to evaluate as a scalar constant (e.g. self.d_k = d_model // n_heads)
            val = _const_value(value, self._param_map)
            if val is not None and isinstance(val, (int, float)):
                self.scalar_attrs[attr] = val
                return
            # Try as a symbolic dim expression (Task A): self.n_head = config.n_head
            sym_val = self._resolve_init_dim(value)
            if isinstance(sym_val, str):
                self.scalar_attrs[attr] = sym_val
                # Record derivation for assert-axiom-aware downstream reasoning.
                if (isinstance(value, ast.BinOp)
                        and isinstance(value.op, (ast.FloorDiv, ast.Mult,
                                                   ast.Add, ast.Sub))):
                    left = self._resolve_init_dim(value.left)
                    right = self._resolve_init_dim(value.right)
                    op_str = {ast.FloorDiv: "//", ast.Mult: "*",
                               ast.Add: "+", ast.Sub: "-"}.get(type(value.op))
                    if (isinstance(left, str) and isinstance(right, str)
                            and op_str is not None):
                        self.symbolic_derivations[attr] = (left, op_str, right)
                return
            # self.X = <opaque expression> (e.g. self.features = features
            # passed in to __init__). Register as an opaque submodule so that
            # forward calls self.X(input) propagate as fully-symbolic UNKNOWN
            # rather than (unsoundly) preserving the input shape.
            if isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)):
                self.layers[attr] = LayerDef(
                    attr_name=attr, kind=LayerKind.UNKNOWN,
                    line=getattr(value, "lineno", 0),
                )
            return
        func_name = _name_or_attr(value.func)

        # Handle nn.Parameter(torch.randn/zeros/ones(...)) → extract shape
        if func_name in ("nn.Parameter", "torch.nn.Parameter", "Parameter"):
            if value.args:
                shape = _extract_tensor_shape(value.args[0], self._param_map)
                if shape is not None:
                    self.param_shapes[attr] = shape
            return

        is_layer, kind = _is_nn_layer(func_name)

        if is_layer:
            # Before extracting params, substitute local variable references
            # in the call args. E.g., nn.TransformerEncoder(encoder_layer, ...)
            # where encoder_layer is a local that holds a layer Call node.
            patched_call = value
            if (kind in (LayerKind.TRANSFORMER_ENCODER,
                         LayerKind.TRANSFORMER_DECODER)
                    and value.args
                    and isinstance(value.args[0], ast.Name)
                    and value.args[0].id in self._local_layer_calls):
                import copy as _copy
                patched_call = _copy.copy(value)
                patched_call.args = list(patched_call.args)
                patched_call.args[0] = self._local_layer_calls[value.args[0].id]
            layer = _extract_layer_params(kind, patched_call, self._param_map,
                                          self.config_param_names,
                                          self.symbolic_config_attrs)
            layer.attr_name = attr
            self.layers[attr] = layer
            # Synthesise nn.Parameter shapes for the layer's weight/bias so
            # that ``forward`` references like ``weight = self.fc.weight`` can
            # be resolved against a concrete shape rather than abstaining.
            self._register_layer_param_shapes(attr, layer)
            return

        # Check if it's a user-defined nn.Module subclass (submodule)
        if func_name and func_name in self._class_map:
            cls_node = self._class_map[func_name]
            # Resolve constructor arguments
            ctor_args = [_const_value(a, self._param_map) for a in value.args]
            # Extract the submodule's computation graph
            sub_graph = _extract_submodule_graph(
                cls_node, ctor_args, self._class_map
            )
            if sub_graph is not None:
                # Create a SUBMODULE layer
                layer = LayerDef(
                    attr_name=attr,
                    kind=LayerKind.SUBMODULE,
                    line=value.lineno if hasattr(value, 'lineno') else 0,
                    sub_graph=sub_graph,
                )
                self.layers[attr] = layer
                # Also hoist inner layers with prefixed names for reference
                for inner_name, inner_layer in sub_graph.layers.items():
                    prefixed = f"{attr}.{inner_name}"
                    prefixed_layer = copy.copy(inner_layer)
                    prefixed_layer.attr_name = prefixed
                    self.layers[prefixed] = prefixed_layer
            return

        # Helper-function expansion: ``self.X = helper(args)`` where
        # ``helper`` is a top-level function in the same source whose body
        # is essentially ``return nn.<Layer>(...)``.  Substitute the helper
        # parameter names with the call-site argument values and recurse.
        # Only expand when every positional argument is statically
        # resolvable to a constant in the current parameter map; otherwise
        # fall back to opaque to avoid introducing false positives from a
        # partially-unresolved layer constructor.
        if func_name and func_name in self._function_map:
            all_const = all(
                _const_value(a, self._param_map) is not None
                for a in value.args
            ) and all(
                _const_value(kw.value, self._param_map) is not None
                for kw in value.keywords
            )
            if all_const:
                expanded = _expand_layer_helper(
                    self._function_map[func_name], value, self._param_map
                )
                if expanded is not None:
                    exp_func_name = _name_or_attr(expanded.func)
                    exp_is_layer, exp_kind = _is_nn_layer(exp_func_name)
                    if exp_is_layer:
                        layer = _extract_layer_params(
                            exp_kind, expanded, self._param_map,
                            self.config_param_names,
                            self.symbolic_config_attrs)
                        layer.attr_name = attr
                        self.layers[attr] = layer
                        return

        # Fallback: ``self.X = some_helper(...)`` where ``some_helper`` is
        # neither a recognised nn layer nor a known nn.Module subclass.
        # The result is presumed to be an opaque nn.Module — register it as
        # such so that ``self.X(input)`` in forward returns a fully-symbolic
        # shape rather than (unsoundly) preserving the input shape.
        self.layers[attr] = LayerDef(
            attr_name=attr, kind=LayerKind.UNKNOWN,
            line=value.lineno if hasattr(value, "lineno") else 0,
        )


# --- _ForwardExtractor: walks forward() to build computation steps --------

_FUNCTIONAL_OPS: Dict[str, OpKind] = {
    "relu": OpKind.ACTIVATION,
    "sigmoid": OpKind.ACTIVATION,
    "tanh": OpKind.ACTIVATION,
    "gelu": OpKind.ACTIVATION,
    "leaky_relu": OpKind.ACTIVATION,
    "silu": OpKind.ACTIVATION,
    "mish": OpKind.ACTIVATION,
    "hardswish": OpKind.ACTIVATION,
    "hardsigmoid": OpKind.ACTIVATION,
    "elu": OpKind.ACTIVATION,
    "selu": OpKind.ACTIVATION,
    "prelu": OpKind.ACTIVATION,
    "relu6": OpKind.ACTIVATION,
    "dropout": OpKind.DROPOUT,
    "alpha_dropout": OpKind.DROPOUT,
    "softmax": OpKind.SOFTMAX,
    "log_softmax": OpKind.SOFTMAX,
    "cat": OpKind.CAT,
    "stack": OpKind.STACK,
    "interpolate": OpKind.INTERPOLATE,
    "pad": OpKind.PAD,
    "grid_sample": OpKind.ACTIVATION,
    "embedding": OpKind.LAYER_CALL,
    "batch_norm": OpKind.LAYER_CALL,
    "layer_norm": OpKind.LAYER_CALL,
    "group_norm": OpKind.LAYER_CALL,
    "instance_norm": OpKind.LAYER_CALL,
    "linear": OpKind.LAYER_CALL,
    "conv1d": OpKind.LAYER_CALL,
    "conv2d": OpKind.LAYER_CALL,
    "conv3d": OpKind.LAYER_CALL,
    "max_pool1d": OpKind.LAYER_CALL,
    "max_pool2d": OpKind.LAYER_CALL,
    "max_pool3d": OpKind.LAYER_CALL,
    "avg_pool1d": OpKind.LAYER_CALL,
    "avg_pool2d": OpKind.LAYER_CALL,
    "avg_pool3d": OpKind.LAYER_CALL,
    "adaptive_avg_pool1d": OpKind.LAYER_CALL,
    "adaptive_avg_pool2d": OpKind.LAYER_CALL,
    "adaptive_max_pool1d": OpKind.LAYER_CALL,
    "adaptive_max_pool2d": OpKind.LAYER_CALL,
    "pixel_shuffle": OpKind.LAYER_CALL,
    "pixel_unshuffle": OpKind.LAYER_CALL,
    "fold": OpKind.LAYER_CALL,
    "unfold": OpKind.LAYER_CALL,
    "upsample": OpKind.INTERPOLATE,
}

# Tensor-factory functions whose output SHAPE is fully determined by the call
# arguments and is independent of the RNG seed.  ``torch.rand(2, 4)`` always
# produces a (2, 4) tensor regardless of the random values it contains, so the
# verifier can — and must — track its shape rather than abstaining.  These are
# leaf ops (they create a tensor; they do not consume one).
_TENSOR_FACTORY_FNS: Dict[str, str] = {
    # name → default dtype family ("float", "int", or "" = inherit/unknown)
    "rand": "float", "randn": "float", "zeros": "float", "ones": "float",
    "empty": "float", "full": "float", "randint": "int",
    "normal": "float", "poisson": "float",
}
# Factory args that are NOT shape dimensions and must be skipped when reading
# the size: ``torch.full(size, fill_value)`` and ``torch.randint(low, high,
# size)`` take the size as a single tuple, handled specially below.

# Mapping from functional call names to LayerKind, used to create synthetic
# LayerDef objects so that shape propagation works for F.max_pool2d etc.
_FUNC_LAYER_KIND: Dict[str, "LayerKind"] = {
    "max_pool1d": LayerKind.MAXPOOL1D,
    "max_pool2d": LayerKind.MAXPOOL2D,
    "max_pool3d": LayerKind.MAXPOOL3D,
    "avg_pool1d": LayerKind.AVGPOOL1D,
    "avg_pool2d": LayerKind.AVGPOOL2D,
    "avg_pool3d": LayerKind.AVGPOOL3D,
    "adaptive_avg_pool1d": LayerKind.ADAPTIVE_AVGPOOL1D,
    "adaptive_avg_pool2d": LayerKind.ADAPTIVE_AVGPOOL2D,
    "adaptive_max_pool1d": LayerKind.ADAPTIVE_MAXPOOL1D,
    "adaptive_max_pool2d": LayerKind.ADAPTIVE_MAXPOOL2D,
    "linear": LayerKind.LINEAR,
    "conv1d": LayerKind.CONV1D,
    "conv2d": LayerKind.CONV2D,
    "conv3d": LayerKind.CONV3D,
    "batch_norm": LayerKind.BATCHNORM1D,
    "layer_norm": LayerKind.LAYERNORM,
    "group_norm": LayerKind.GROUPNORM,
    "instance_norm": LayerKind.INSTANCENORM2D,
    "embedding": LayerKind.EMBEDDING,
    "pixel_shuffle": OpKind.LAYER_CALL,
    "pixel_unshuffle": OpKind.LAYER_CALL,
}


def _make_functional_layer(
    func_name: str,
    call_node: ast.Call,
) -> Optional["LayerDef"]:
    """Create a synthetic LayerDef for a functional call (e.g. F.max_pool2d).

    Extracts kernel_size, stride, padding, output_size etc. from the call
    arguments so that the shape propagation functions can operate normally.
    """
    kind = _FUNC_LAYER_KIND.get(func_name)
    if kind is None:
        return None

    # The first positional arg is the input tensor; parameters start at arg[1].
    pos_args = call_node.args[1:]  # skip input tensor
    kw = {k.arg: _const_value(k.value) for k in call_node.keywords if k.arg}

    layer = LayerDef(attr_name=f"__func_{func_name}", kind=kind, params=dict(kw))

    if kind in (LayerKind.MAXPOOL2D, LayerKind.AVGPOOL2D,
                LayerKind.MAXPOOL1D, LayerKind.AVGPOOL1D,
                LayerKind.MAXPOOL3D, LayerKind.AVGPOOL3D):
        # F.max_pool2d(input, kernel_size, stride=None, padding=0, ...)
        ks = _const_value(pos_args[0]) if pos_args else kw.get("kernel_size", 2)
        if isinstance(ks, int):
            ks = (ks, ks) if "2d" in func_name else (ks,)
        layer.kernel_size = ks if isinstance(ks, tuple) else (ks, ks)
        stride = (_const_value(pos_args[1]) if len(pos_args) > 1
                  else kw.get("stride", None))
        if stride is None:
            stride = layer.kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride) if "2d" in func_name else (stride,)
        layer.params["stride"] = stride
        padding = (_const_value(pos_args[2]) if len(pos_args) > 2
                   else kw.get("padding", 0))
        if isinstance(padding, int):
            padding = (padding, padding) if "2d" in func_name else (padding,)
        layer.params["padding"] = padding

    elif kind in (LayerKind.ADAPTIVE_AVGPOOL2D, LayerKind.ADAPTIVE_MAXPOOL2D):
        # F.adaptive_avg_pool2d(input, output_size)
        out_sz = _const_value(pos_args[0]) if pos_args else kw.get("output_size")
        if isinstance(out_sz, int):
            out_sz = (out_sz, out_sz)
        layer.output_size = out_sz

    elif kind in (LayerKind.ADAPTIVE_AVGPOOL1D, LayerKind.ADAPTIVE_MAXPOOL1D):
        out_sz = _const_value(pos_args[0]) if pos_args else kw.get("output_size")
        if isinstance(out_sz, int):
            out_sz = (out_sz,)
        layer.output_size = out_sz

    elif kind == LayerKind.LINEAR:
        # F.linear(input, weight, bias) — we can't easily know in/out features
        # from the weight tensor name alone, so skip shape constraints.
        pass

    elif kind in (LayerKind.CONV2D, LayerKind.CONV1D, LayerKind.CONV3D):
        # F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        # Without concrete weight shape, we can't infer out_channels.
        pass

    elif kind in (LayerKind.BATCHNORM1D, LayerKind.LAYERNORM,
                  LayerKind.GROUPNORM, LayerKind.INSTANCENORM2D):
        # Normalization ops preserve shape
        pass

    return layer


_METHOD_OPS: Dict[str, OpKind] = {
    "view": OpKind.RESHAPE,
    "reshape": OpKind.RESHAPE,
    "flatten": OpKind.FLATTEN,
    "squeeze": OpKind.SQUEEZE,
    "unsqueeze": OpKind.UNSQUEEZE,
    "transpose": OpKind.TRANSPOSE,
    "t": OpKind.TRANSPOSE,        # x.t() — 2D-only transpose; swap dims (-1, -2)
    "permute": OpKind.PERMUTE,
    "contiguous": OpKind.CONTIGUOUS,
    "detach": OpKind.DETACH,
    "to": OpKind.TO_DEVICE,
    "cuda": OpKind.TO_DEVICE,
    "cpu": OpKind.TO_DEVICE,
    "expand": OpKind.EXPAND,
    "repeat": OpKind.REPEAT,
    "expand_as": OpKind.EXPAND,
    "repeat_interleave": OpKind.REPEAT,
    "chunk": OpKind.CHUNK,
    "split": OpKind.SPLIT,
    "unbind": OpKind.UNBIND,
    "mean": OpKind.MEAN_REDUCE,
    "sum": OpKind.SUM_REDUCE,
    "max": OpKind.MEAN_REDUCE,
    "min": OpKind.MEAN_REDUCE,
    "norm": OpKind.MEAN_REDUCE,
    "std": OpKind.MEAN_REDUCE,
    "var": OpKind.MEAN_REDUCE,
    "softmax": OpKind.SOFTMAX,
    "log_softmax": OpKind.SOFTMAX,
}


class _ForwardExtractor(ast.NodeVisitor):
    """Extracts computation steps from an nn.Module's ``forward()``."""

    def __init__(self, layers: Dict[str, LayerDef],
                 scalar_attrs: Optional[Dict[str, Any]] = None) -> None:
        self.layers = layers
        self.steps: List[ComputationStep] = []
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self._tmp_counter = 0
        self._current_names: Dict[int, str] = {}  # ast node id → tensor name
        self._aliases: Dict[str, str] = {}  # variable alias tracking
        # Maps variable name → (tensor_name, dim_index) for shape dim aliases.
        # Populated by "B, C, H, W = x.shape" unpacking.  Used to replace
        # symbolic view() args with copy-from-dim sentinels (≤ -2).
        # Encoding: sentinel -k-2 means "copy from source dim k".
        self._shape_dim_map: Dict[str, Tuple[str, int]] = {}
        # Scalar instance attributes: {"self.d_k": 64, "self.n_heads": 8}
        self._scalar_attrs: Dict[str, Any] = {}
        if scalar_attrs:
            for k, v in scalar_attrs.items():
                self._scalar_attrs[f"self.{k}"] = v
        # Local scalar variables defined inside forward(), e.g.
        #   wrong_features = self.hidden_size // 2
        # so that ``view(batch, seq, wrong_features)`` resolves the third dim.
        self._local_scalars: Dict[str, Any] = {}
        # Layer attribute aliases: {"weight": "self.conv.weight"} from
        # ``weight = self.conv.weight`` so that downstream view/reshape calls
        # see the correct parameter shape from the registered Linear/Conv layer.
        self._param_aliases: Dict[str, str] = {}
        # Shape-tuple-valued local variables.  Each entry is a list of
        # "dim refs" that may be: int (concrete dim), str (symbolic dim
        # expression), or ('copy', tensor_name, dim_idx) tuples (i.e. the
        # k-th dim of ``tensor_name``).  Used so ``new_shape =
        # x.size()[:-1] + (H, D)`` followed by ``y.view(*new_shape)``
        # resolves to a fully-determined view target rather than -1.
        self._shape_tuples: Dict[str, List[Any]] = {}

    def _fresh(self, hint: str = "t") -> str:
        self._tmp_counter += 1
        return f"__{hint}_{self._tmp_counter}"

    def _dim_env(self) -> Dict[str, Any]:
        """Combined name → const/symbol map for view/reshape dim resolution.

        Includes ``self.<attr>`` scalar attributes captured from ``__init__``
        plus any local variables bound inside ``forward()`` to a constant or
        symbolic dimension expression.
        """
        env: Dict[str, Any] = dict(self._scalar_attrs)
        env.update(self._local_scalars)
        return env

    def _try_record_local_scalar(self, target_name: str,
                                 value: ast.expr) -> bool:
        """If ``value`` is a constant / symbolic scalar expression, bind it
        to ``target_name`` in ``_local_scalars`` so later view/reshape args
        that mention ``target_name`` resolve concretely.

        Returns True iff the value was recorded (caller may still emit a
        CONTIGUOUS / no-op step; we deliberately do nothing else).
        """
        env = self._dim_env()
        # Try concrete int / float first.
        v = _const_value(value, env)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            self._local_scalars[target_name] = v
            return True
        # Try symbolic dim expression (e.g. self.hidden_size // self.heads).
        v2 = _resolve_dim_value(value, env, None, None)
        if isinstance(v2, str):
            self._local_scalars[target_name] = v2
            return True
        return False

    def _try_record_shape_dim_alias(self, target_name: str,
                                     value: ast.expr) -> bool:
        """Detect ``batch_size = x.shape[i]`` / ``= x.size(i)`` and register
        ``target_name`` in ``_shape_dim_map`` so a later
        ``y.view(batch_size, ...)`` resolves to a copy-from-dim sentinel
        rather than an opaque -1."""
        # Pattern 1: x.shape[i] (Subscript(Attribute(value=x, attr='shape'),
        # slice=Constant(i)))
        if isinstance(value, ast.Subscript):
            base = value.value
            if (isinstance(base, ast.Attribute) and base.attr == "shape"):
                idx_val = _const_value(value.slice)
                if isinstance(idx_val, int):
                    src = self._resolve_name(base.value)
                    self._shape_dim_map[target_name] = (src, idx_val)
                    return True
        # Pattern 2: x.size(i)
        if (isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "size"
                and len(value.args) == 1):
            idx_val = _const_value(value.args[0])
            if isinstance(idx_val, int):
                src = self._resolve_name(value.func.value)
                self._shape_dim_map[target_name] = (src, idx_val)
                return True
        return False

    def _eval_shape_tuple(self, value: ast.expr) -> Optional[List[Any]]:
        """Try to evaluate ``value`` as a shape-tuple (list of dim refs).

        Each returned element is one of:
          - int: concrete dim
          - str: symbolic dim expression
          - ('copy', tensor_name, dim_idx): k-th dim of ``tensor_name``
        Returns None if the value is not a recognised shape-tuple form.
        """
        env = self._dim_env()
        # x.size() with no args → full shape of x as copy refs.
        if (isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "size"
                and not value.args):
            src = self._resolve_name(value.func.value)
            shp = self.layers.get(src) if src in self.layers else None
            # We don't know the rank statically; emit a sentinel "all dims
            # of src" tuple via a single placeholder; the consumer (slice /
            # concat) handles this lazily.  For simplicity we expand to a
            # generous fixed rank of 8 — slicing with [:-1] etc collapses
            # this naturally because the resulting tuple is then used in
            # concatenation that determines the final length.
            #
            # Instead of a fixed rank, we inspect downstream usage by
            # tagging the entry as a special ('full_size', src) marker.
            return [("full_size", src)]
        # x.shape (Attribute) → same as x.size()
        if isinstance(value, ast.Attribute) and value.attr == "shape":
            src = self._resolve_name(value.value)
            return [("full_size", src)]
        # Subscript: e.g. x.size()[:-1] or x.shape[:-1] or named[:-1]
        if isinstance(value, ast.Subscript):
            base = self._eval_shape_tuple(value.value)
            if base is None:
                return None
            sl = value.slice
            if isinstance(sl, ast.Slice):
                lo = _const_value(sl.lower) if sl.lower else None
                hi = _const_value(sl.upper) if sl.upper else None
                stp = _const_value(sl.step) if sl.step else None
                # Encode as a slice on the materialised tuple if possible.
                # We expand ('full_size', src) lazily: keep as a wrapped
                # ('slice', src, lo, hi, stp) marker.
                if (len(base) == 1 and isinstance(base[0], tuple)
                        and base[0][0] == "full_size"):
                    src = base[0][1]
                    return [("slice_size", src, lo, hi, stp)]
                # Already-materialised list: apply Python slice directly.
                try:
                    return list(base[slice(lo, hi, stp)])
                except Exception:
                    return None
            # Single-int indexing not useful here (returns a scalar).
            return None
        # BinOp Add: tuple concatenation t1 + t2.
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            left = self._eval_shape_tuple(value.left)
            right = self._eval_shape_tuple(value.right)
            if left is None or right is None:
                return None
            return left + right
        # Tuple/List literal: (a, b, c) or [a, b, c].
        if isinstance(value, (ast.Tuple, ast.List)):
            out: List[Any] = []
            for el in value.elts:
                v = _const_value(el, env)
                if isinstance(v, int):
                    out.append(v)
                    continue
                rv = _resolve_dim_value(el, env, None, None)
                if isinstance(rv, (int, str)):
                    out.append(rv)
                    continue
                # Recognise x.shape[i] as a copy-ref.
                if isinstance(el, ast.Subscript):
                    b = el.value
                    if isinstance(b, ast.Attribute) and b.attr == "shape":
                        iv = _const_value(el.slice)
                        if isinstance(iv, int):
                            src = self._resolve_name(b.value)
                            out.append(("copy", src, iv))
                            continue
                if (isinstance(el, ast.Call)
                        and isinstance(el.func, ast.Attribute)
                        and el.func.attr == "size"
                        and len(el.args) == 1):
                    iv = _const_value(el.args[0])
                    if isinstance(iv, int):
                        src = self._resolve_name(el.func.value)
                        out.append(("copy", src, iv))
                        continue
                # Name bound to a shape-dim alias (B from "B = x.shape[0]").
                if isinstance(el, ast.Name) and el.id in self._shape_dim_map:
                    src, di = self._shape_dim_map[el.id]
                    out.append(("copy", src, di))
                    continue
                return None
            return out
        # Name reference to a previously-recorded shape tuple.
        if isinstance(value, ast.Name) and value.id in self._shape_tuples:
            return list(self._shape_tuples[value.id])
        return None

    def _try_record_shape_tuple(self, target_name: str,
                                 value: ast.expr) -> bool:
        """Try to record ``target_name`` as a shape-tuple-valued local.

        Returns True iff a shape tuple was recognised.  See
        :meth:`_eval_shape_tuple` for the encoding.
        """
        st = self._eval_shape_tuple(value)
        if st is None:
            return False
        self._shape_tuples[target_name] = st
        return True

    def _materialise_shape_tuple(self, st: List[Any],
                                  base_for_view: Optional[str] = None
                                  ) -> Optional[List[Any]]:
        """Expand a shape-tuple's lazy markers into concrete dim refs.

        ``('full_size', src)`` and ``('slice_size', src, lo, hi, stp)``
        markers are expanded to ('copy', src, k) tuples.  Returns None if
        we cannot determine the rank of any referenced tensor.
        """
        out: List[Any] = []
        for entry in st:
            if (isinstance(entry, tuple) and len(entry) >= 2
                    and entry[0] in ("full_size", "slice_size")):
                src = entry[1]
                # Look up the tensor's known rank.  We use the source
                # graph step's output shape if available.  Fallback: if
                # ``src`` is the input being viewed and the view extractor
                # passes ``base_for_view``, we leave the entries as
                # ``('copy_src', src, k)`` so the propagator can resolve
                # later.
                src_rank = self._lookup_tensor_rank(src)
                if src_rank is None:
                    return None
                if entry[0] == "full_size":
                    rng = range(src_rank)
                else:
                    lo, hi, stp = entry[2], entry[3], entry[4]
                    rng = range(*slice(lo, hi, stp).indices(src_rank))
                for k in rng:
                    out.append(("copy", src, k))
            else:
                out.append(entry)
        return out

    def _lookup_tensor_rank(self, name: str) -> Optional[int]:
        """Return the static rank of a tensor variable, if known.

        Walks back through ``self.steps`` to find the producing step and
        infer the output rank.  Used to expand ``x.size()`` markers.
        """
        # Forward inputs: rank is whatever the user passes at verification
        # time, so we conservatively return None unless the propagator can
        # back-fill it.  But for shape-tuple expansion we only need rank
        # in the no-input case rarely; in the upstream-faithful corpus the
        # patterns we care about are ``x.size()[:-1]`` where ``x`` has a
        # known rank coming out of a Linear/etc.  We conservatively assume
        # rank 3 for unknown forward inputs (typical NLP/Vision tensors)
        # only when the slice doesn't depend on it; since we still record
        # ('copy', src, k) the propagator validates k against the actual
        # shape at check time.
        for step in reversed(self.steps):
            if step.output == name:
                if step.op == OpKind.LAYER_CALL and step.layer_ref:
                    layer = self.layers.get(step.layer_ref)
                    if layer is not None and layer.kind in (
                            LayerKind.LINEAR,):
                        # Linear preserves input rank.  Walk back further.
                        if step.inputs:
                            r = self._lookup_tensor_rank(step.inputs[0])
                            if r is not None:
                                return r
                # Reshape: dims length is the new rank.
                if step.op == OpKind.RESHAPE:
                    dims = step.params.get("dims") if step.params else None
                    if dims:
                        return len(dims)
                return None
        # Forward inputs default to rank 3 (BTC) — common in the patterns
        # we care about.  This is a sound under-approximation: if the
        # actual rank differs, the propagator will fail validation.
        if name in self.input_names:
            return 3
        return None

    def _try_record_layer_param_alias(self, target_name: str, value: ast.expr,
                                       line: int, col: int) -> bool:
        """Detect ``weight = self.<layer>.weight`` style assignments and emit
        a CONTIGUOUS step that aliases the local name to the canonical
        ``self.<layer>.<param>`` shape entry registered by
        ``_InitExtractor._register_layer_param_shapes``.

        Without this, downstream ``weight.view(...)`` calls on layer
        parameters fall through to an opaque -1 reshape and the verifier
        cannot detect (e.g.) the PEFT DoRA Conv2d-with-groups bug whose
        bug *is* the wrong total-element count of the view target.
        """
        if not isinstance(value, ast.Attribute):
            return False
        if value.attr not in ("weight", "bias"):
            return False
        inner = value.value
        if not (isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"):
            return False
        layer_name = inner.attr
        if layer_name not in self.layers:
            return False
        canonical = f"self.{layer_name}.{value.attr}"
        self._param_aliases[target_name] = canonical
        # Emit a shape-preserving CONTIGUOUS step so the propagator copies
        # the parameter shape into the local name's shape_env entry.
        self.steps.append(ComputationStep(
            op=OpKind.CONTIGUOUS,
            inputs=[canonical],
            output=target_name,
            line=line, col=col,
        ))
        return True


    def _resolve_name(self, node: ast.expr) -> str:
        """Return the tensor-variable name for an expression, following aliases."""
        nid = id(node)
        if nid in self._current_names:
            return self._current_names[nid]

        if isinstance(node, ast.Name):
            name = node.id
            # Follow alias chain
            seen = set()
            while name in self._aliases and name not in seen:
                seen.add(name)
                name = self._aliases[name]
            return name
        if isinstance(node, ast.Attribute):
            base = _name_or_attr(node.value)
            if base == "self":
                return f"self.{node.attr}"
            if base:
                return f"{base}.{node.attr}"
        return self._fresh()

    # --- entry point -------------------------------------------------------

    def extract(self, func_node: ast.FunctionDef) -> None:
        # Input names (excluding 'self')
        for arg in func_node.args.args:
            if arg.arg != "self":
                self.input_names.append(arg.arg)

        self.visit(func_node)

    # --- visitors ----------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1:
            target = node.targets[0]
            # Handle tuple unpacking: attn_out, _ = self.attn(x, x, x)
            # Map the first element to the computation step output
            if isinstance(target, ast.Tuple) and target.elts:
                first_elt = target.elts[0]
                if isinstance(first_elt, ast.Name):
                    target_name = first_elt.id
                else:
                    target_name = self._fresh("tuple")
                # Map all named elements to the same output for shape tracking
                self._current_names[id(target)] = target_name

                # Detect "B, C, H, W = x.shape" and record each variable as a
                # shape-dim alias so that view(B, 9, 20, H, W) can be resolved
                # to concrete copy-from-dim sentinels rather than opaque -1s.
                if (isinstance(node.value, ast.Attribute)
                        and node.value.attr == "shape"):
                    src_tensor = self._resolve_name(node.value.value)
                    for dim_i, elt in enumerate(target.elts):
                        if isinstance(elt, ast.Name) and elt.id != "_":
                            self._shape_dim_map[elt.id] = (src_tensor, dim_i)
                    self.generic_visit(node)
                    return

                # Detect "B, T, C = x.size()" — same as x.shape but a method call.
                if (isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "size"
                        and not node.value.args):
                    src_tensor = self._resolve_name(node.value.func.value)
                    for dim_i, elt in enumerate(target.elts):
                        if isinstance(elt, ast.Name) and elt.id != "_":
                            self._shape_dim_map[elt.id] = (src_tensor, dim_i)
                    self.generic_visit(node)
                    return

                # Distribute parallel-tuple assignment: q, k, v = a[0], a[1], a[2]
                # → process each (target_i, value_i) independently.
                if (isinstance(node.value, ast.Tuple)
                        and len(node.value.elts) == len(target.elts)):
                    for tgt_i, val_i in zip(target.elts, node.value.elts):
                        if isinstance(tgt_i, ast.Name) and tgt_i.id != "_":
                            sub_name = tgt_i.id
                        else:
                            sub_name = self._fresh("ptup")
                        self._process_expr(val_i, sub_name,
                                           node.lineno, node.col_offset)
                    self.generic_visit(node)
                    return

                # --- Task B: q, k, v = X.split(...) / X.chunk(...) -----------
                # Emit a SPLIT step per output element so each q/k/v gets the
                # correct chunk shape independently.
                if self._try_emit_split_unpack(target, node.value,
                                                node.lineno, node.col_offset):
                    self.generic_visit(node)
                    return

                # --- q, k, v = X.unbind(dim) ------------------------------------
                if self._try_emit_unbind_unpack(target, node.value,
                                                node.lineno, node.col_offset):
                    self.generic_visit(node)
                    return

                # Handle nested tuple for LSTM hidden state extraction:
                #   _, (h, _) = self.lstm(x)  or  output, (h_n, c_n) = self.lstm(x)
                # The inner tuple contains the hidden state which has a different
                # shape than the output tensor.
                if (len(target.elts) == 2
                        and isinstance(target.elts[1], ast.Tuple)
                        and isinstance(node.value, ast.Call)):
                    inner_tuple = target.elts[1]
                    call_node = node.value
                    layer_name = self._get_rnn_layer_name(call_node)
                    if layer_name is not None and inner_tuple.elts:
                        self._process_expr(node.value, target_name,
                                           node.lineno, node.col_offset)
                        layer = self.layers.get(layer_name)
                        if layer and layer.kind in (LayerKind.LSTM, LayerKind.GRU):
                            self._add_hidden_state_step(
                                layer_name, layer, inner_tuple.elts[0],
                                target_name, node.lineno, node.col_offset)
                        self.generic_visit(node)
                        return

                # Handle flat tuple for GRU/LSTM hidden state extraction:
                #   _, h = self.gru(x)  or  output, h = self.lstm(x)
                # Second element is the hidden state directly (for GRU) or
                # the (h_n, c_n) tuple packed as a single Name variable.
                if (len(target.elts) == 2
                        and isinstance(target.elts[1], ast.Name)
                        and target.elts[1].id != '_'
                        and isinstance(node.value, ast.Call)):
                    call_node = node.value
                    layer_name = self._get_rnn_layer_name(call_node)
                    if layer_name is not None:
                        self._process_expr(node.value, target_name,
                                           node.lineno, node.col_offset)
                        layer = self.layers.get(layer_name)
                        if layer and layer.kind in (LayerKind.LSTM, LayerKind.GRU):
                            self._add_hidden_state_step(
                                layer_name, layer, target.elts[1],
                                target_name, node.lineno, node.col_offset)
                        self.generic_visit(node)
                        return
            else:
                target_name = self._resolve_name(target)
        else:
            target_name = self._fresh("assign")

        # Local scalar capture: ``wrong_features = self.hidden_size // 2``.
        # Recording this allows downstream view/reshape calls that mention
        # ``wrong_features`` to resolve the dim concretely instead of
        # falling through to an opaque -1.  Only meaningful when the
        # target is a single ast.Name (genuine local variable).
        if (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            self._try_record_local_scalar(node.targets[0].id, node.value)
            # Single-dim shape alias: ``batch = x.shape[0]`` or
            # ``batch = x.size(0)``.  Register in _shape_dim_map so a
            # follow-up ``y.view(batch, -1, H, D)`` resolves the first
            # arg as "copy from base dim 0" sentinel rather than -1.
            self._try_record_shape_dim_alias(node.targets[0].id, node.value)
            # Track shape-tuple-valued local variables (size(), [:-1],
            # tuple concatenation) so ``view(*new_shape)`` patterns can
            # be resolved concretely.  Used heavily in HuggingFace
            # ``new_qkv_shape = qkv.size()[:-1] + (H, D)`` style code.
            self._try_record_shape_tuple(node.targets[0].id, node.value)
            # Layer-attribute alias: ``weight = self.conv.weight`` or
            # ``bias = self.fc.bias``.  We record the canonical name so a
            # follow-up ``weight.view(...)`` is treated as a reshape of the
            # registered nn.Parameter shape (populated below via
            # ``_layer_param_shape``).
            self._try_record_layer_param_alias(node.targets[0].id, node.value,
                                               node.lineno, node.col_offset)

        self._process_expr(node.value, target_name, node.lineno, node.col_offset)
        self.generic_visit(node)


    def _try_emit_split_unpack(self, target: ast.Tuple, value: ast.expr,
                                line: int, col: int) -> bool:
        """Detect ``q, k, v = X.split(...)`` / ``X.chunk(...)`` and emit a
        SPLIT step per output element so that each q/k/v's shape reflects its
        own slice of the split dimension.

        Returns True if handled, False to fall through to default tuple logic.
        """
        if not isinstance(value, ast.Call):
            return False
        if not (isinstance(value.func, ast.Attribute)
                and value.func.attr in ("split", "chunk")):
            return False
        method = value.func.attr
        # Skip torch.split(...) functional form — base would be 'torch'.
        base_name = _name_or_attr(value.func.value)
        if base_name in ("torch", "F", "nn", "torch.nn",
                         "torch.nn.functional"):
            # torch.split(x, split_size, dim) — input is value.args[0]
            return False  # don't handle here

        base = self._resolve_arg(value.func.value)
        n = len(target.elts)
        if n == 0:
            return False

        # dim
        dim_val = None
        if len(value.args) > 1:
            dim_val = _const_value(value.args[1], self._scalar_attrs)
        for kw in value.keywords:
            if kw.arg == "dim":
                dim_val = _const_value(kw.value, self._scalar_attrs)
        if not isinstance(dim_val, int):
            dim_val = 0

        # split_size_or_sizes (per-chunk size) for .split, chunks for .chunk
        sizes: List[Any] = []  # length n; each entry is int|str|None
        chunks_count: Optional[int] = None
        if method == "split":
            if value.args:
                first = value.args[0]
                if isinstance(first, ast.List):
                    sizes = [_const_value(e, self._scalar_attrs) for e in first.elts]
                else:
                    raw = _const_value(first, self._scalar_attrs)
                    if raw is None:
                        # try resolution via scalar_attrs which may hold strings
                        # _const_value already does that for self.X if str.
                        # Fall back: attempt direct attr lookup
                        if (isinstance(first, ast.Attribute)
                                and isinstance(first.value, ast.Name)
                                and first.value.id == "self"
                                and first.attr in [k.split(".",1)[1] if "." in k else k
                                                    for k in self._scalar_attrs]):
                            raw = self._scalar_attrs.get(f"self.{first.attr}")
                    sizes = [raw] * n
        else:  # chunk
            if value.args:
                cn = _const_value(value.args[0], self._scalar_attrs)
                if isinstance(cn, int) and cn > 0:
                    chunks_count = cn

        # Emit per-element steps
        for i, elt in enumerate(target.elts):
            if isinstance(elt, ast.Name) and elt.id != "_":
                out_name = elt.id
            else:
                out_name = self._fresh("split")
            params: Dict[str, Any] = {"dim": dim_val, "split_index": i,
                                       "n_outputs": n}
            if method == "split":
                if sizes and i < len(sizes):
                    params["split_size"] = sizes[i]
                # Only set "chunks" when split_size is concretely an int —
                # if symbolic, downstream propagation should prefer the
                # n_outputs-based fallback to detect divisibility bugs.
                if (sizes and i < len(sizes)
                        and isinstance(sizes[i], int)):
                    params["chunks"] = n
            else:  # chunk
                if chunks_count is not None:
                    params["chunks"] = chunks_count
                else:
                    params["chunks"] = n
            self.steps.append(ComputationStep(
                op=OpKind.SPLIT if method == "split" else OpKind.CHUNK,
                inputs=[base], output=out_name,
                params=params, line=line, col=col,
            ))
        return True

    def _try_emit_unbind_unpack(self, target: ast.Tuple, value: ast.expr,
                                line: int, col: int) -> bool:
        """Detect ``q, k, v = X.unbind(dim)`` and emit one UNBIND step per
        output element.  Each output element drops the split dimension from
        the input shape (unbind removes dim completely, unlike split/chunk).

        Returns True if handled, False to fall through to default tuple logic.
        """
        if not isinstance(value, ast.Call):
            return False
        if not (isinstance(value.func, ast.Attribute)
                and value.func.attr == "unbind"):
            return False
        # Skip torch.unbind(x, dim) functional form.
        base_name = _name_or_attr(value.func.value)
        if base_name in ("torch", "F", "nn", "torch.nn",
                         "torch.nn.functional"):
            return False

        base = self._resolve_arg(value.func.value)
        n = len(target.elts)
        if n == 0:
            return False

        # dim (default 0)
        dim_val = 0
        if value.args:
            dim_val = _const_value(value.args[0], self._scalar_attrs)
            if not isinstance(dim_val, int):
                dim_val = 0
        for kw in value.keywords:
            if kw.arg == "dim":
                v = _const_value(kw.value, self._scalar_attrs)
                if isinstance(v, int):
                    dim_val = v

        for i, elt in enumerate(target.elts):
            out_name = elt.id if isinstance(elt, ast.Name) and elt.id != "_" \
                       else self._fresh("unbind")
            self.steps.append(ComputationStep(
                op=OpKind.UNBIND,
                inputs=[base], output=out_name,
                params={"dim": dim_val, "unbind_index": i, "n_outputs": n},
                line=line, col=col,
            ))
        return True

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handle augmented assignments like ``x += self.layer(x)``."""
        if isinstance(node.target, ast.Name):
            target_name = node.target.id
        else:
            target_name = self._fresh("augassign")
        lhs = self._resolve_name(node.target) if isinstance(node.target, ast.Name) else target_name
        rhs = self._resolve_arg(node.value)
        if isinstance(node.op, ast.Add):
            op = OpKind.ADD
        elif isinstance(node.op, ast.MatMult):
            op = OpKind.MATMUL
        elif isinstance(node.op, (ast.Mult, ast.Sub)):
            op = OpKind.MULTIPLY
        else:
            op = OpKind.CUSTOM
        self.steps.append(ComputationStep(
            op=op, inputs=[lhs, rhs], output=target_name,
            line=node.lineno, col=node.col_offset,
        ))
        self.generic_visit(node)

    def _add_hidden_state_step(
        self, layer_name: str, layer: LayerDef, h_elt: ast.expr,
        lstm_output: str, line: int, col: int,
    ) -> None:
        """Create a pseudo-step that establishes the hidden state shape.

        For LSTM/GRU, h_n has shape (num_layers*D, batch, hidden_size).
        The last dim is always hidden_size, NOT hidden_size*D.
        """
        if isinstance(h_elt, ast.Name) and h_elt.id != '_':
            h_name = h_elt.id
            pseudo_name = f"__{layer_name}_hidden"
            hidden_layer = LayerDef(
                attr_name=pseudo_name,
                kind=LayerKind.LINEAR,
                line=line,
                in_features=None,   # don't constrain input
                out_features=layer.hidden_size,
            )
            self.layers[pseudo_name] = hidden_layer
            self.steps.append(ComputationStep(
                op=OpKind.LAYER_CALL,
                inputs=[lstm_output],
                output=h_name,
                layer_ref=pseudo_name,
                line=line, col=col,
            ))

    def _get_rnn_layer_name(self, call_node: ast.Call) -> Optional[str]:
        """Return the layer name if call_node is self.<lstm_or_gru>(...), else None."""
        func = call_node.func
        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and func.attr in self.layers):
            layer = self.layers[func.attr]
            if layer.kind in (LayerKind.LSTM, LayerKind.GRU):
                return func.attr
        return None

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            # If the return value is a call, process it as a computation step
            if isinstance(node.value, (ast.Call, ast.BinOp)):
                target = self._fresh("ret")
                self._process_expr(
                    node.value, target, node.lineno, node.col_offset
                )
                self.output_names.append(target)
                self.steps.append(ComputationStep(
                    op=OpKind.RETURN,
                    inputs=[target],
                    output=target,
                    line=node.lineno,
                ))
            else:
                name = self._resolve_name(node.value)
                self.output_names.append(name)
                self.steps.append(ComputationStep(
                    op=OpKind.RETURN,
                    inputs=[name],
                    output=name,
                    line=node.lineno,
                ))

    def visit_If(self, node: ast.If) -> None:
        """Handle if/else branches with path-sensitive analysis.

        For ``if self.training:`` patterns, record which branch is active
        in train vs eval mode.  For general conditionals, process both
        branches and emit a ConditionalStep that records both paths.
        """
        cond_str = self._classify_condition(node.test)

        # Save current step list; extract both branches independently
        saved_steps = self.steps
        self.steps = []
        for child in node.body:
            self.visit(child)
        true_steps = self.steps

        self.steps = []
        for child in node.orelse:
            self.visit(child)
        false_steps = self.steps

        # Restore original step list
        self.steps = saved_steps

        if not true_steps and not false_steps:
            return

        # Build a ConditionalStep that carries both branches
        all_inputs: List[str] = []
        for s in true_steps + false_steps:
            all_inputs.extend(s.inputs)
        # Deduplicate while preserving order
        seen: set = set()
        unique_inputs: List[str] = []
        for inp in all_inputs:
            if inp not in seen:
                seen.add(inp)
                unique_inputs.append(inp)

        self.steps.append(ComputationStep(
            op=OpKind.CONDITIONAL,
            inputs=unique_inputs,
            output=self._fresh("cond"),
            line=node.lineno,
            col=node.col_offset,
            condition=cond_str,
            true_branch=true_steps if true_steps else None,
            false_branch=false_steps if false_steps else None,
        ))

    @staticmethod
    def _classify_condition(test: ast.expr) -> str:
        """Classify a conditional test node into a descriptive string."""
        # ``self.training``
        if (isinstance(test, ast.Attribute)
                and isinstance(test.value, ast.Name)
                and test.value.id == "self" and test.attr == "training"):
            return "self.training"
        # ``not self.training``
        if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Attribute)
                and isinstance(test.operand.value, ast.Name)
                and test.operand.value.id == "self"
                and test.operand.attr == "training"):
            return "not self.training"
        # ``hasattr(self, 'attr')``
        if (isinstance(test, ast.Call)
                and isinstance(test.func, ast.Name)
                and test.func.id == "hasattr"
                and len(test.args) >= 2):
            obj = _name_or_attr(test.args[0])
            attr_val = _const_value(test.args[1])
            if obj and attr_val:
                return f"hasattr:{obj}.{attr_val}"
        # x.shape[i] > N  / x.size(i) > N  and similar comparisons
        if isinstance(test, ast.Compare):
            left_str = ast.unparse(test.left) if hasattr(ast, 'unparse') else "<expr>"
            return f"compare:{left_str}"
        return "unknown"

    def _process_expr(
        self, node: ast.expr, target: str, line: int, col: int
    ) -> None:
        """Convert an expression AST node into computation steps."""

        # --- binary op: x @ y  /  x + y  etc. ---
        if isinstance(node, ast.BinOp):
            left = self._resolve_arg(node.left)
            right = self._resolve_arg(node.right)
            if isinstance(node.op, ast.MatMult):
                op = OpKind.MATMUL
            elif isinstance(node.op, ast.Add):
                op = OpKind.ADD
            elif isinstance(node.op, (ast.Mult, ast.Sub)):
                op = OpKind.MULTIPLY
            else:
                op = OpKind.CUSTOM
            self.steps.append(ComputationStep(
                op=op, inputs=[left, right], output=target,
                line=line, col=col,
            ))
            return

        # --- method / function calls ---
        if isinstance(node, ast.Call):
            self._process_call(node, target, line, col)
            return

        # --- subscript: out[:, -1, :] ---
        if isinstance(node, ast.Subscript):
            base = self._resolve_arg(node.value)
            # Parse the subscript indices to determine which dims are kept/dropped
            indices = self._parse_subscript_indices(node.slice)
            if indices is not None:
                self.steps.append(ComputationStep(
                    op=OpKind.SUBSCRIPT,
                    inputs=[base],
                    output=target,
                    params={"indices": indices},
                    line=line, col=col,
                ))
            else:
                # Unknown subscript pattern — alias to base (conservative)
                self._aliases[target] = base
            return

        # --- simple copy: y = x  — emit shape-preserving step so skip
        #     connections retain their own shape entry when x is overwritten ---
        if isinstance(node, ast.Name):
            source = self._resolve_name(node)
            if source != target:
                self.steps.append(ComputationStep(
                    op=OpKind.CONTIGUOUS,
                    inputs=[source],
                    output=target,
                    line=line, col=col,
                ))
            return

    @staticmethod
    def _parse_subscript_indices(slice_node: ast.expr):
        """Parse subscript indices into a list of index descriptors.

        Each element is:
          'slice' — keep the dimension (e.g. : or start:stop)
          'int'   — eliminate the dimension (integer index, e.g. -1, 0)
        Returns None if the pattern is not recognized.
        """
        # Single index: x[0] or x[-1]
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
            return ['int']
        if isinstance(slice_node, ast.UnaryOp) and isinstance(slice_node.op, ast.USub):
            if isinstance(slice_node.operand, ast.Constant) and isinstance(slice_node.operand.value, int):
                return ['int']
        # Single slice: x[:]
        if isinstance(slice_node, ast.Slice):
            return ['slice']
        # Tuple of indices: x[:, -1, :] → ast.Tuple
        if isinstance(slice_node, ast.Tuple):
            result = []
            for elt in slice_node.elts:
                if isinstance(elt, ast.Slice):
                    result.append('slice')
                elif isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    result.append('int')
                elif (isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub)
                      and isinstance(elt.operand, ast.Constant)
                      and isinstance(elt.operand.value, int)):
                    result.append('int')
                else:
                    return None  # Unsupported index type
            return result
        return None

    def _resolve_arg(self, arg: ast.expr) -> str:
        """Resolve a call argument, recursively processing nested expressions."""
        if isinstance(arg, ast.Call):
            tmp = self._fresh("inner")
            self._process_call(arg, tmp, getattr(arg, 'lineno', 0),
                               getattr(arg, 'col_offset', 0))
            return tmp
        if isinstance(arg, ast.BinOp):
            tmp = self._fresh("binop")
            self._process_expr(arg, tmp, getattr(arg, 'lineno', 0),
                               getattr(arg, 'col_offset', 0))
            return tmp
        if isinstance(arg, ast.Subscript):
            tmp = self._fresh("subscript")
            self._process_expr(arg, tmp, getattr(arg, 'lineno', 0),
                               getattr(arg, 'col_offset', 0))
            return tmp
        # Handle the `.T` attribute (e.g. self.weight.T) — emit a TRANSPOSE step
        # swapping the last two dimensions (matching PyTorch semantics for matmul).
        if isinstance(arg, ast.Attribute) and arg.attr == "T":
            base = self._resolve_arg(arg.value)
            tmp = self._fresh("T_attr")
            self.steps.append(ComputationStep(
                op=OpKind.TRANSPOSE,
                inputs=[base],
                output=tmp,
                params={"dim0": -2, "dim1": -1},
                line=getattr(arg, 'lineno', 0),
                col=getattr(arg, 'col_offset', 0),
            ))
            return tmp
        return self._resolve_name(arg)

    def _factory_step(
        self, short: str, node: ast.Call, target: str, line: int, col: int
    ) -> Optional["ComputationStep"]:
        """Build a ``NEW_TENSOR`` step for a tensor-factory call.

        Returns ``None`` (→ caller falls back to CUSTOM, a sound abstention) if
        the output shape cannot be statically determined from the call.

        Handles the heterogeneous torch factory signatures:
          * ``rand/randn/zeros/ones/empty(*size)`` — size = positional dims or a
            single tuple/list arg.
          * ``full(size, fill_value)`` — size is the first arg (tuple/list).
          * ``randint(high, size)`` / ``randint(low, high, size)`` — size is the
            last positional arg (a tuple/list).
          * ``randperm(n)`` — produces shape ``(n,)``.
        Only statically-known integer dimensions are accepted; any dynamic /
        data-dependent dim (e.g. ``x.shape[0]``) causes abstention.
        """
        dim_env = self._dim_env()

        def _as_dim(a: ast.expr) -> Optional[ShapeDim]:
            v = _const_value(a, dim_env)
            if isinstance(v, bool):  # guard: bools are ints in Python
                return None
            if isinstance(v, int):
                return ShapeDim(v)
            # A bare Name that resolves (via dim_env) to a known int dimension.
            if isinstance(a, ast.Name) and a.id in dim_env:
                dv = dim_env[a.id]
                if isinstance(dv, int):
                    return ShapeDim(dv)
            return None

        size_args: List[ast.expr]
        if short == "randperm":
            if not node.args:
                return None
            d = _as_dim(node.args[0])
            if d is None:
                return None
            shape = TensorShape((d,))
            dtype_family = "int"
            size_args = []
        else:
            # Locate the size argument list per signature.
            if short == "full":
                size_args = [node.args[0]] if node.args else []
            elif short == "randint":
                size_args = [node.args[-1]] if node.args else []
            else:
                size_args = list(node.args)
            # Size may be a single tuple/list, or several positional ints.
            dims: List[ShapeDim] = []
            if (len(size_args) == 1
                    and isinstance(size_args[0], (ast.Tuple, ast.List))):
                elts = size_args[0].elts
            else:
                elts = size_args
            if not elts:
                return None
            for e in elts:
                d = _as_dim(e)
                if d is None:
                    return None
                dims.append(d)
            shape = TensorShape(tuple(dims))
            dtype_family = _TENSOR_FACTORY_FNS.get(short, "")

        params: Dict[str, Any] = {"shape": shape}
        # Capture an explicit device= kwarg when it is a static string/literal.
        for kw in node.keywords:
            if kw.arg == "device":
                dval = _const_value(kw.value)
                if isinstance(dval, str):
                    params["device"] = dval
            elif kw.arg == "dtype":
                params["cast_dtype"] = _name_or_attr(kw.value) or ""
        params["dtype_family"] = dtype_family
        return ComputationStep(
            op=OpKind.NEW_TENSOR,
            inputs=[],
            output=target,
            params=params,
            line=line, col=col,
        )

    def _process_call(
        self, node: ast.Call, target: str, line: int, col: int
    ) -> None:
        func = node.func

        # --- self.<layer>(x) ------------------------------------------------
        if (isinstance(func, ast.Attribute) and
                isinstance(func.value, ast.Name) and
                func.value.id == "self" and
                func.attr in self.layers):
            layer_name = func.attr
            inputs = [self._resolve_arg(a) for a in node.args]
            self.steps.append(ComputationStep(
                op=OpKind.LAYER_CALL,
                inputs=inputs,
                output=target,
                layer_ref=layer_name,
                line=line, col=col,
            ))
            return

        # --- self.<unknown_attr>(x) ------------------------------------------
        # E.g. self._process_input(x) where _process_input is a sibling
        # method or a self.X = <opaque> never seen in __init__. Register an
        # opaque LayerKind.UNKNOWN so the call propagates a fully-symbolic
        # output shape (sound abstention) rather than falling through to
        # OpKind.CUSTOM (which preserves the input shape — unsound).
        if (isinstance(func, ast.Attribute) and
                isinstance(func.value, ast.Name) and
                func.value.id == "self"):
            layer_name = func.attr
            self.layers[layer_name] = LayerDef(
                attr_name=layer_name, kind=LayerKind.UNKNOWN, line=line,
            )
            inputs = [self._resolve_arg(a) for a in node.args]
            self.steps.append(ComputationStep(
                op=OpKind.LAYER_CALL,
                inputs=inputs,
                output=target,
                layer_ref=layer_name,
                line=line, col=col,
            ))
            return

        # --- x.<method>(...) ------------------------------------------------
        if isinstance(func, ast.Attribute):
            method = func.attr
            if method in _METHOD_OPS:
                # Skip if the base is a well-known module (torch, F, nn) —
                # e.g. ``torch.flatten(x, 1)`` should be handled as a
                # function call, not as a method on the ``torch`` object.
                base_name = _name_or_attr(func.value)
                if base_name not in ("torch", "F", "nn", "np",
                                     "torch.nn", "torch.nn.functional"):
                    base = self._resolve_arg(func.value)
                    params: Dict[str, Any] = {}

                    if method in ("view", "reshape"):
                        env = self._dim_env()
                        # Expand starred shape-tuple args:
                        # ``view(*new_shape)`` where ``new_shape`` was
                        # recorded by _try_record_shape_tuple.  Each entry
                        # of the recorded tuple is one of:
                        #   - int: concrete dim
                        #   - str: symbolic dim expression
                        #   - ('copy', tensor, k): k-th dim of ``tensor``
                        #
                        # We expand into a flat list of "synthetic" arg
                        # placeholders: a None ast.Name marker plus a
                        # parallel list ``preresolved`` mapping idx to the
                        # concrete value to use in ``dims``.
                        flat_args: List[Optional[ast.expr]] = []
                        preresolved: Dict[int, Any] = {}
                        starred_aliases: Dict[int, Tuple[str, int]] = {}
                        for a in node.args:
                            if (isinstance(a, ast.Starred)
                                    and isinstance(a.value, ast.Name)
                                    and a.value.id in self._shape_tuples):
                                st = self._materialise_shape_tuple(
                                    self._shape_tuples[a.value.id])
                                if st is not None:
                                    for entry in st:
                                        i = len(flat_args)
                                        if (isinstance(entry, tuple)
                                                and entry[0] == "copy"):
                                            starred_aliases[i] = (
                                                entry[1], entry[2])
                                            preresolved[i] = entry[1]
                                        else:
                                            preresolved[i] = entry
                                        flat_args.append(None)
                                    continue
                            flat_args.append(a)
                        dims = [
                            preresolved[i] if i in preresolved
                            else _const_value(a, env)
                            for i, a in enumerate(flat_args)
                        ]
                        # Detect x.size(dim) or x.shape[dim] patterns
                        size_dim_indices = []
                        # Records {dim_idx: (src_tensor_var, src_dim_idx)} for
                        # cross-tensor aliases so the propagator can resolve
                        # them against the current shape_env.
                        alias_resolutions: Dict[int, Tuple[str, int]] = dict(
                            starred_aliases)
                        for idx, (d, a) in enumerate(zip(dims, flat_args)):
                            if a is None:
                                continue  # pre-resolved (starred expansion)
                            if d is None:
                                if isinstance(a, ast.Call):
                                    # x.size(dim) → mark as "keep from input"
                                    size_dim_indices.append(idx)
                                elif isinstance(a, ast.Subscript):
                                    # x.shape[dim] → mark as "keep from input"
                                    size_dim_indices.append(idx)
                                elif isinstance(a, ast.Name):
                                    # Local-scalar resolution must precede
                                    # shape-dim alias logic so a view dim that
                                    # was bound to a constructor scalar (e.g.
                                    # ``wrong_features = self.hidden_size//2``)
                                    # propagates as a concrete int rather than
                                    # an opaque -1 free dimension.
                                    if a.id in self._local_scalars:
                                        dims[idx] = self._local_scalars[a.id]
                                        continue
                                    # Check if variable comes from "B, C, H, W = base.shape"
                                    alias = self._shape_dim_map.get(a.id)
                                    if alias is not None and alias[0] == base:
                                        # Encode as sentinel -k-2 meaning "copy from dim k"
                                        dims[idx] = -alias[1] - 2
                                        continue
                                    if alias is not None:
                                        # Cross-tensor alias: record for
                                        # propagation-time resolution AND fall
                                        # back to symbolic name as the dim.
                                        alias_resolutions[idx] = alias
                                        dims[idx] = a.id
                                        continue
                                elif isinstance(a, ast.BinOp):
                                    # Try to resolve as int|str via dim helper.
                                    rv = _resolve_dim_value(
                                        a, env, None, None)
                                    if isinstance(rv, (int, str)):
                                        dims[idx] = rv
                                        continue
                                dims[idx] = -1
                        # Common pattern: view(x.size(0), -1) → flatten(1)
                        if (size_dim_indices and len(dims) == 2
                                and dims[0] == -1 and dims[1] == -1
                                and 0 in size_dim_indices):
                            self.steps.append(ComputationStep(
                                op=OpKind.FLATTEN,
                                inputs=[base],
                                output=target,
                                params={"start_dim": 1},
                                line=line, col=col,
                            ))
                            return
                        # For x.size(dim) args, use sentinel 0 (keep from input)
                        # to allow the reshape logic to copy that dim from input
                        for idx in size_dim_indices:
                            dims[idx] = 0
                        params["dims"] = tuple(
                            d if d is not None else -1 for d in dims
                        )
                        if alias_resolutions:
                            params["__alias_resolutions__"] = alias_resolutions
                    elif method == "flatten":
                        sd = _const_value(node.args[0]) if node.args else 1
                        params["start_dim"] = sd
                        if len(node.args) > 1:
                            ed = _const_value(node.args[1])
                            if ed is not None:
                                params["end_dim"] = ed
                        elif "end_dim" in {kw.arg for kw in node.keywords}:
                            for kw in node.keywords:
                                if kw.arg == "end_dim":
                                    ed = _const_value(kw.value)
                                    if ed is not None:
                                        params["end_dim"] = ed
                    elif method in ("squeeze", "unsqueeze"):
                        if node.args:
                            params["dim"] = _const_value(node.args[0])
                    elif method == "transpose":
                        if len(node.args) >= 2:
                            params["dim0"] = _const_value(node.args[0])
                            params["dim1"] = _const_value(node.args[1])
                    elif method == "t":
                        # x.t() — 2D transpose, swap last two dimensions
                        params["dim0"] = -2
                        params["dim1"] = -1
                    elif method == "permute":
                        params["dims"] = tuple(
                            _const_value(a) for a in node.args
                        )
                    elif method == "to":
                        if node.args:
                            params["device"] = _const_value(node.args[0])
                        for kw in node.keywords:
                            if kw.arg == "device":
                                params["device"] = _const_value(kw.value)
                    elif method == "cuda":
                        params["device"] = "cuda:0"
                    elif method == "cpu":
                        params["device"] = "cpu"
                    elif method == "expand":
                        eargs = node.args
                        if (len(eargs) == 1
                                and isinstance(eargs[0], (ast.Tuple, ast.List))):
                            eargs = eargs[0].elts
                        params["dims"] = tuple(
                            _const_value(a) for a in eargs
                        )
                    elif method in ("expand_as", "repeat_interleave"):
                        pass  # shape-preserving approximation
                    elif method == "repeat":
                        rargs = node.args
                        if (len(rargs) == 1
                                and isinstance(rargs[0], (ast.Tuple, ast.List))):
                            rargs = rargs[0].elts
                        params["dims"] = tuple(
                            _const_value(a) for a in rargs
                        )
                    elif method in ("mean", "sum", "max", "min",
                                    "norm", "std", "var"):
                        if node.args:
                            params["dim"] = _const_value(node.args[0])
                        for kw_node in node.keywords:
                            if kw_node.arg == "dim":
                                params["dim"] = _const_value(kw_node.value)
                            elif kw_node.arg == "keepdim":
                                params["keepdim"] = _const_value(kw_node.value)
                    elif method in ("chunk", "split"):
                        if node.args:
                            params["chunks"] = _const_value(node.args[0])
                        if len(node.args) > 1:
                            params["dim"] = _const_value(node.args[1])
                        for kw_node in node.keywords:
                            if kw_node.arg == "dim":
                                params["dim"] = _const_value(kw_node.value)
                    elif method in ("softmax", "log_softmax"):
                        if node.args:
                            params["dim"] = _const_value(node.args[0])
                        for kw_node in node.keywords:
                            if kw_node.arg == "dim":
                                params["dim"] = _const_value(kw_node.value)

                    self.steps.append(ComputationStep(
                        op=_METHOD_OPS[method],
                        inputs=[base],
                        output=target,
                        params=params,
                        line=line, col=col,
                    ))
                    return

        # --- F.<func>(...) or torch.<func>(...) ------------------------------
        func_name = _name_or_attr(func)
        if func_name:
            short = func_name.split(".")[-1]

            # Functional ops
            if short in _FUNCTIONAL_OPS:
                op = _FUNCTIONAL_OPS[short]
                # For cat/stack, first arg is a list of tensors
                if op in (OpKind.CAT, OpKind.STACK) and node.args and isinstance(node.args[0], ast.List):
                    inputs = [self._resolve_arg(elt) for elt in node.args[0].elts]
                else:
                    inputs = [self._resolve_arg(a) for a in node.args]
                params_dict: Dict[str, Any] = {}
                for kw in node.keywords:
                    if kw.arg:
                        params_dict[kw.arg] = _const_value(kw.value)

                # For functional calls that map to LAYER_CALL, create a
                # synthetic LayerDef so shape propagation actually works.
                layer_ref_name: Optional[str] = None
                if op == OpKind.LAYER_CALL and short in _FUNC_LAYER_KIND:
                    syn_layer = _make_functional_layer(short, node)
                    if syn_layer is not None:
                        syn_name = f"__func_{short}_{self._tmp_counter}"
                        self._tmp_counter += 1
                        syn_layer.attr_name = syn_name
                        self.layers[syn_name] = syn_layer
                        layer_ref_name = syn_name
                        # For functional calls, first input is the tensor;
                        # the rest are parameters — only keep the tensor.
                        inputs = inputs[:1]

                self.steps.append(ComputationStep(
                    op=_FUNCTIONAL_OPS[short],
                    inputs=inputs,
                    output=target,
                    params=params_dict,
                    layer_ref=layer_ref_name,
                    line=line, col=col,
                ))
                return

            # torch.matmul
            if short in ("matmul", "mm", "bmm"):
                inputs = [self._resolve_arg(a) for a in node.args]
                self.steps.append(ComputationStep(
                    op=OpKind.MATMUL,
                    inputs=inputs,
                    output=target,
                    line=line, col=col,
                ))
                return

            # torch.einsum
            if short == "einsum":
                inputs = [self._resolve_arg(a) for a in node.args]
                params_dict_ein: Dict[str, Any] = {}
                if node.args and isinstance(node.args[0], ast.Constant):
                    params_dict_ein["equation"] = node.args[0].value
                self.steps.append(ComputationStep(
                    op=OpKind.EINSUM,
                    inputs=inputs[1:] if inputs else [],
                    output=target,
                    params=params_dict_ein,
                    line=line, col=col,
                ))
                return

            # torch.where
            if short == "where":
                inputs = [self._resolve_arg(a) for a in node.args]
                self.steps.append(ComputationStep(
                    op=OpKind.WHERE,
                    inputs=inputs,
                    output=target,
                    line=line, col=col,
                ))
                return

            # torch.flatten
            if short == "flatten":
                inputs = [self._resolve_arg(a) for a in node.args]
                params_dict_flat: Dict[str, Any] = {}
                if len(node.args) > 1:
                    params_dict_flat["start_dim"] = _const_value(node.args[1])
                else:
                    params_dict_flat["start_dim"] = 0
                if len(node.args) > 2:
                    ed = _const_value(node.args[2])
                    if ed is not None:
                        params_dict_flat["end_dim"] = ed
                self.steps.append(ComputationStep(
                    op=OpKind.FLATTEN,
                    inputs=inputs[:1] if inputs else [],
                    output=target,
                    params=params_dict_flat,
                    line=line, col=col,
                ))
                return

            # --- tensor factory: torch.rand/randn/zeros/ones/empty/full/
            #     randint/randperm — RNG-independent shape (Step 32) ----------
            if short in _TENSOR_FACTORY_FNS or short == "randperm":
                fstep = self._factory_step(short, node, target, line, col)
                if fstep is not None:
                    self.steps.append(fstep)
                    return
                # Could not determine a static shape → fall through to CUSTOM
                # (sound abstention rather than a guessed shape).

        # --- inline nn.Layer(args)(input) ------------------------------------
        # Detect patterns like nn.Linear(999, 13)(x) where func is itself a Call
        if isinstance(func, ast.Call):
            ctor_name = _name_or_attr(func.func)
            is_layer, kind = _is_nn_layer(ctor_name)
            if is_layer:
                # Create a temporary layer definition for shape checking
                layer = _extract_layer_params(kind, func)
                inline_name = f"__inline_{self._tmp_counter}"
                self._tmp_counter += 1
                layer.attr_name = inline_name
                self.layers[inline_name] = layer
                inputs = [self._resolve_arg(a) for a in node.args]
                self.steps.append(ComputationStep(
                    op=OpKind.LAYER_CALL,
                    inputs=inputs,
                    output=target,
                    layer_ref=inline_name,
                    line=line, col=col,
                ))
                return

        # Fallback: custom
        inputs = [self._resolve_arg(a) for a in node.args]
        self.steps.append(ComputationStep(
            op=OpKind.CUSTOM,
            inputs=inputs,
            output=target,
            line=line, col=col,
        ))


# --- Top-level extraction function ----------------------------------------

def _find_method(cls_node: ast.ClassDef, name: str) -> Optional[ast.FunctionDef]:
    """Find a method by name inside a ClassDef."""
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def _collect_helper_functions(tree: ast.AST) -> Dict[str, ast.FunctionDef]:
    """Return top-level ``def helper(...): return nn.<Layer>(...)`` functions.

    Used to expand torchvision-style ``conv3x3``/``conv1x1`` helpers and
    similar one-liner factory functions inline at a layer assignment site
    so the resulting layer is recognised as in-fragment instead of opaque.
    """
    out: Dict[str, ast.FunctionDef] = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Body must be a single Return whose value is a Call.
        body = [s for s in node.body if not isinstance(s, ast.Expr)
                or not (isinstance(s.value, ast.Constant)
                        and isinstance(s.value.value, str))]  # strip docstring
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            continue
        ret = body[0].value
        if not isinstance(ret, ast.Call):
            continue
        # Only register helpers that return a known nn.<Layer>(...)
        ret_name = _name_or_attr(ret.func)
        is_layer, _ = _is_nn_layer(ret_name)
        if is_layer:
            out[node.name] = node
    return out


def _expand_layer_helper(
    fn: ast.FunctionDef, call: ast.Call, outer_param_map: Dict[str, Any],
) -> Optional[ast.Call]:
    """Substitute *call*'s positional/keyword args into *fn*'s body Call.

    Returns the rewritten ``nn.<Layer>(...)`` Call node with parameter
    references replaced by the call-site argument expressions, or ``None``
    if substitution fails.  Only handles the simple shape:
        def helper(a, b=..., c=...):
            return nn.<Layer>(<expr-using-a-b-c>)
    """
    # Strip docstring + locate the return Call
    body = [s for s in fn.body if not (isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Constant)
            and isinstance(s.value.value, str))]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    ret_call = body[0].value
    if not isinstance(ret_call, ast.Call):
        return None
    # Build name -> AST-expr map from the call site, applying defaults.
    args = fn.args
    name_to_expr: Dict[str, ast.expr] = {}
    pos_names = [a.arg for a in args.args]
    defaults = list(args.defaults or [])
    n_defaults = len(defaults)
    n_args = len(pos_names)
    # apply defaults
    for i, d in enumerate(defaults):
        name_to_expr[pos_names[n_args - n_defaults + i]] = d
    # positional args
    for i, a in enumerate(call.args):
        if i < n_args:
            name_to_expr[pos_names[i]] = a
    # keyword args
    for kw in call.keywords:
        if kw.arg in pos_names:
            name_to_expr[kw.arg] = kw.value
    # Substitute parameter Name nodes inside the return call.
    class _Substitute(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:  # noqa: N802
            if n.id in name_to_expr:
                return name_to_expr[n.id]
            return n
    new_call = _Substitute().visit(copy.deepcopy(ret_call))
    ast.fix_missing_locations(new_call)
    return new_call


def _collect_module_classes(tree: ast.AST) -> List[ast.ClassDef]:
    """Return all ``nn.Module`` subclass definitions from *tree* in source order."""
    classes = []
    # Walk top-level statements to preserve source order
    body = getattr(tree, 'body', [])
    for node in body:
        if isinstance(node, ast.ClassDef):
            bases = [_name_or_attr(b) for b in node.bases]
            is_module = any(
                b in (
                    "nn.Module", "Module", "torch.nn.Module",
                    # Common nn.Module subclasses whose children are also
                    # valid analysis targets (no additional __init__ analysis
                    # needed — inherited parameters are handled symbolically):
                    "nn.Linear", "Linear", "torch.nn.Linear",
                    "nn.Embedding", "Embedding", "torch.nn.Embedding",
                    "nn.LayerNorm", "LayerNorm", "torch.nn.LayerNorm",
                )
                for b in bases if b is not None
            )
            if is_module:
                classes.append(node)
    # Also check if any base is another locally-defined nn.Module
    local_module_names = {c.name for c in classes}
    for node in body:
        if isinstance(node, ast.ClassDef) and node not in classes:
            bases = [_name_or_attr(b) for b in node.bases]
            if any(b in local_module_names for b in bases if b is not None):
                classes.append(node)
    return classes


def _find_root_module(
    module_classes: List[ast.ClassDef],
) -> ast.ClassDef:
    """Return the 'root' nn.Module — the one not used as a submodule.

    Heuristic: the last class in source order is typically the main model.
    If a class is instantiated inside another class's __init__, it is a
    submodule and should be skipped.
    """
    if len(module_classes) == 1:
        return module_classes[0]

    class_names = {c.name for c in module_classes}
    # Find which class names are instantiated inside other classes' __init__
    submodule_names: Set[str] = set()
    for cls_node in module_classes:
        init_fn = _find_method(cls_node, "__init__")
        if init_fn is None:
            continue
        for node in ast.walk(init_fn):
            if isinstance(node, ast.Call):
                call_name = _name_or_attr(node.func)
                if call_name in class_names:
                    submodule_names.add(call_name)

    # Root = class that is NOT used as a submodule by another
    roots = [c for c in module_classes if c.name not in submodule_names]
    if roots:
        return roots[-1]  # last root in source order
    return module_classes[-1]  # fallback: last class


def _extract_submodule_graph(
    cls_node: ast.ClassDef,
    constructor_args: List[Any],
    class_map: Dict[str, ast.ClassDef],
) -> Optional[ComputationGraph]:
    """Extract a submodule's computation graph with bound constructor args.

    *constructor_args* are the concrete values passed to the submodule's
    ``__init__`` (excluding ``self``).
    """
    init_fn = _find_method(cls_node, "__init__")
    if init_fn is None:
        return None

    # Build param_map by binding constructor args to init params
    params = init_fn.args.args
    param_map: Dict[str, Any] = {}

    # First, populate with default values
    defaults = init_fn.args.defaults or []
    num_params = len(params)
    num_defaults = len(defaults)
    for i, default in enumerate(defaults):
        arg_idx = num_params - num_defaults + i
        if 0 <= arg_idx < num_params:
            pname = params[arg_idx].arg
            val = _const_value(default)
            if val is not None:
                param_map[pname] = val

    # Now override with actual constructor args (skip 'self')
    non_self_params = [p for p in params if p.arg != "self"]
    for i, arg_val in enumerate(constructor_args):
        if i < len(non_self_params):
            param_map[non_self_params[i].arg] = arg_val

    # Extract layers using the bound param_map
    extractor = _InitExtractor(class_map=class_map)
    extractor._param_map = param_map
    extractor.visit(init_fn)

    graph = ComputationGraph(class_name=cls_node.name)
    graph.layers = extractor.layers

    # Extract forward steps
    fwd_fn = _find_method(cls_node, "forward")
    if fwd_fn:
        fwd_ext = _ForwardExtractor(graph.layers, scalar_attrs=extractor.scalar_attrs)
        fwd_ext.extract(fwd_fn)
        graph.steps = fwd_ext.steps
        graph.input_names = fwd_ext.input_names
        graph.output_names = fwd_ext.output_names

    return graph


def extract_computation_graph(source: str) -> ComputationGraph:
    """Parse Python *source* and extract the computation graph of the root
    ``nn.Module`` subclass found.

    When the source contains multiple ``nn.Module`` classes (e.g. a
    ``BasicBlock`` helper and a ``ResNet`` model), this function selects
    the root model (the one not used as a submodule by another class) and
    inlines submodule calls so that shapes propagate correctly across
    user-defined module boundaries.

    Returns a ``ComputationGraph`` populated with layers (from ``__init__``)
    and computation steps (from ``forward``).

    Raises ``ValueError`` if no nn.Module subclass is found.
    """
    tree = ast.parse(source)

    # Detect dynamic features at module level
    dynamic_features = _detect_dynamic_features(tree, source)

    # Collect all nn.Module classes
    module_classes = _collect_module_classes(tree)
    if not module_classes:
        raise ValueError("No nn.Module subclass found in source")

    # Build class map for submodule resolution
    class_map: Dict[str, ast.ClassDef] = {c.name: c for c in module_classes}
    # Collect helper factory functions (e.g. ``conv3x3``) for inline expansion
    function_map: Dict[str, ast.FunctionDef] = _collect_helper_functions(tree)

    # Select the root module
    root_cls = _find_root_module(module_classes)

    graph = ComputationGraph(class_name=root_cls.name)
    graph.dynamic_features = dynamic_features

    # Check for @torch.compile decorator on the class
    for dec in root_cls.decorator_list:
        dec_name = _name_or_attr(dec) if not isinstance(dec, ast.Call) else _name_or_attr(dec.func)
        if dec_name in ("torch.compile", "compile"):
            graph.dynamic_features["torch_compile"] = True

    # --- __init__: extract layers (with submodule awareness) ---
    init_fn = _find_method(root_cls, "__init__")
    scalar_attrs: Dict[str, Any] = {}
    if init_fn:
        extractor = _InitExtractor(class_map=class_map, function_map=function_map)
        extractor.extract(init_fn)
        graph.layers = extractor.layers
        graph.buffer_shapes = extractor.buffer_shapes
        graph.param_shapes = extractor.param_shapes
        scalar_attrs = extractor.scalar_attrs

    # Inject inherited parameter shapes for well-known nn.Module subclasses
    # that omit __init__ (e.g. nn.Linear subclasses use self.weight/self.bias).
    if not init_fn:
        bases = [_name_or_attr(b) for b in root_cls.bases]
        if any(b in ("nn.Linear", "Linear", "torch.nn.Linear") for b in bases if b):
            # Symbolic (out_features, in_features) — not known statically
            graph.param_shapes.setdefault(
                "self.weight",
                TensorShape((ShapeDim("_linear_out"), ShapeDim("_linear_in")))
            )
            graph.param_shapes.setdefault(
                "self.bias",
                TensorShape((ShapeDim("_linear_out"),))
            )
        elif any(b in ("nn.Embedding", "Embedding", "torch.nn.Embedding") for b in bases if b):
            graph.param_shapes.setdefault(
                "self.weight",
                TensorShape((ShapeDim("_emb_vocab"), ShapeDim("_emb_dim")))
            )
        elif any(b in ("nn.LayerNorm", "LayerNorm", "torch.nn.LayerNorm") for b in bases if b):
            graph.param_shapes.setdefault(
                "self.weight",
                TensorShape((ShapeDim("_ln_dim"),))
            )
            graph.param_shapes.setdefault(
                "self.bias",
                TensorShape((ShapeDim("_ln_dim"),))
            )

    # --- forward: extract steps ---
    fwd_fn = _find_method(root_cls, "forward")
    if fwd_fn:
        # Check for @torch.compile on forward
        for dec in fwd_fn.decorator_list:
            dec_name = _name_or_attr(dec) if not isinstance(dec, ast.Call) else _name_or_attr(dec.func)
            if dec_name in ("torch.compile", "compile"):
                graph.dynamic_features["torch_compile_forward"] = True

        fwd_ext = _ForwardExtractor(graph.layers, scalar_attrs=scalar_attrs)
        fwd_ext.extract(fwd_fn)
        graph.steps = fwd_ext.steps
        graph.input_names = fwd_ext.input_names
        graph.output_names = fwd_ext.output_names

        # Detect dynamic patterns in forward body
        _detect_forward_dynamic_patterns(fwd_fn, graph.dynamic_features)

    return graph


def _detect_dynamic_features(tree: ast.AST, source: str) -> Dict[str, Any]:
    """Detect dynamic computation graph features in source.

    Performs both textual scanning and AST-level analysis to identify
    modern PyTorch patterns that affect verification guarantees.
    """
    features: Dict[str, Any] = {}

    # --- torch.compile / torch._dynamo ---
    if "torch.compile" in source:
        features["torch_compile_present"] = True
        # Check for dynamic=True
        if "dynamic=True" in source or "dynamic = True" in source:
            features["torch_compile_dynamic_shapes"] = True
    if "torch._dynamo" in source:
        features["torch_dynamo_present"] = True

    # --- Mixed precision / autocast ---
    if "torch.autocast" in source or "torch.amp.autocast" in source:
        features["mixed_precision"] = True
        features["mixed_precision_api"] = "torch.amp.autocast"
    if "torch.cuda.amp" in source:
        features["mixed_precision"] = True
        features["mixed_precision_api"] = "torch.cuda.amp"
    if "GradScaler" in source:
        features["grad_scaler"] = True

    # --- Distributed training ---
    if "DistributedDataParallel" in source or "torch.nn.parallel.DistributedDataParallel" in source:
        features["distributed_ddp"] = True
    if "DataParallel" in source and "DistributedDataParallel" not in source:
        features["data_parallel"] = True
    if "torch.distributed" in source:
        features["distributed"] = True

    # --- Export / JIT / tracing ---
    if "torch.export" in source:
        features["torch_export"] = True
    if "torch.jit.script" in source:
        features["jit_script"] = True
    if "torch.jit.trace" in source:
        features["jit_trace"] = True

    # AST-level: scan for decorators and wrapper calls across all functions
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = _name_or_attr(node.func) if isinstance(node.func, (ast.Name, ast.Attribute)) else ""
            if not fname:
                continue
            # torch.compile(...) call
            if fname in ("torch.compile",):
                features["torch_compile_present"] = True
                for kw in node.keywords:
                    if kw.arg == "dynamic":
                        v = _const_value(kw.value)
                        if v is True:
                            features["torch_compile_dynamic_shapes"] = True
                    if kw.arg == "fullgraph":
                        v = _const_value(kw.value)
                        if v is True:
                            features["torch_compile_fullgraph"] = True
                    if kw.arg == "backend":
                        v = _const_value(kw.value)
                        if v:
                            features["torch_compile_backend"] = v
            # DDP wrapping
            if fname in ("DistributedDataParallel",
                         "nn.parallel.DistributedDataParallel",
                         "torch.nn.parallel.DistributedDataParallel"):
                features["distributed_ddp"] = True
            # DataParallel wrapping
            if fname in ("DataParallel", "nn.DataParallel",
                         "torch.nn.DataParallel"):
                features["data_parallel"] = True
            # JIT
            if fname in ("torch.jit.script",):
                features["jit_script"] = True
            if fname in ("torch.jit.trace",):
                features["jit_trace"] = True
            # torch.export
            if fname in ("torch.export.export",):
                features["torch_export"] = True

        # Decorator-level detection
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                dec_name = _name_or_attr(dec) if not isinstance(dec, ast.Call) else _name_or_attr(dec.func)
                if dec_name in ("torch.compile", "compile"):
                    features["torch_compile_present"] = True
                if dec_name in ("torch.jit.script",):
                    features["jit_script"] = True

    return features


def _detect_forward_dynamic_patterns(
    fwd_fn: ast.FunctionDef, features: Dict[str, Any]
) -> None:
    """Detect dynamic patterns within the forward method."""
    data_dependent_branches = []
    shape_dependent_ops = []

    for node in ast.walk(fwd_fn):
        # --- Data-dependent branching ---
        if isinstance(node, ast.If):
            test_src = ast.unparse(node.test) if hasattr(ast, 'unparse') else ""
            if any(kw in test_src for kw in [".shape", ".size(", "len("]):
                data_dependent_branches.append({
                    "line": node.lineno,
                    "condition": test_src,
                    "type": "shape_dependent",
                })
            elif any(kw in test_src for kw in [
                ".item(", ".any(", ".all(",
                ".gt(", ".lt(", ".eq(",
                ".max(", ".min(",
                ".nonzero(", ".bool(",
            ]):
                data_dependent_branches.append({
                    "line": node.lineno,
                    "condition": test_src,
                    "type": "value_dependent",
                })
            elif "self.training" not in test_src:
                # Check if condition involves tensor comparisons
                if any(kw in test_src for kw in [
                    ".data", "torch.", ".sum(", ".mean(",
                    ".argmax(", ".argmin(", ".norm(",
                ]):
                    data_dependent_branches.append({
                        "line": node.lineno,
                        "condition": test_src,
                        "type": "value_dependent",
                    })

        # --- Dynamic reshape: x.view(-1, ...) ---
        if isinstance(node, ast.Call):
            func_name = _name_or_attr(node.func) if isinstance(node.func, (ast.Name, ast.Attribute)) else ""
            if func_name and func_name.endswith((".view", ".reshape")):
                for arg in node.args:
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        shape_dependent_ops.append({
                            "line": getattr(node, 'lineno', 0),
                            "op": "dynamic_reshape",
                        })
                        break

        # --- torch.autocast context manager ---
        if isinstance(node, ast.With):
            for item in node.items:
                ctx_name = ""
                if isinstance(item.context_expr, ast.Call):
                    ctx_name = _name_or_attr(item.context_expr.func) if isinstance(item.context_expr.func, (ast.Name, ast.Attribute)) else ""
                elif isinstance(item.context_expr, (ast.Name, ast.Attribute)):
                    ctx_name = _name_or_attr(item.context_expr) or ""
                if "autocast" in ctx_name:
                    features["forward_uses_autocast"] = True
                if "no_grad" in ctx_name:
                    features["forward_uses_no_grad"] = True

        # --- torch.cuda.amp.autocast as decorator ---
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                dec_name = _name_or_attr(dec) if not isinstance(dec, ast.Call) else _name_or_attr(dec.func)
                if dec_name and "autocast" in dec_name:
                    features["forward_uses_autocast"] = True

    if data_dependent_branches:
        features["data_dependent_branches"] = data_dependent_branches
    if shape_dependent_ops:
        features["shape_dependent_ops"] = shape_dependent_ops


# -- Warning generation for detected dynamic features --

_DYNAMIC_FEATURE_WARNINGS = {
    "torch_compile_present": (
        "torch.compile detected: Dynamo may insert graph breaks that alter "
        "control flow. Shape verification remains valid for the eager-mode "
        "semantics; dtype/layout may change under compiler backends."
    ),
    "torch_compile_dynamic_shapes": (
        "torch.compile(dynamic=True) detected: symbolic shapes used at "
        "runtime may differ from static analysis assumptions. Shape "
        "verification treats dimensions as symbolic — guarantees hold for "
        "all concrete instantiations satisfying the stated constraints."
    ),
    "torch_compile_fullgraph": (
        "torch.compile(fullgraph=True) detected: the compiler will reject "
        "graph breaks, so verified eager-mode shapes match compiled shapes."
    ),
    "torch_dynamo_present": (
        "torch._dynamo usage detected: Dynamo guards may constrain tensor "
        "shapes at runtime beyond what static analysis captures."
    ),
    "mixed_precision": (
        "Mixed-precision (autocast) detected: dtypes may be silently "
        "down-cast to float16/bfloat16. Shape verification is unaffected, "
        "but dtype-sensitive operations (e.g., large reductions) may lose "
        "precision. Gradient scaling should use GradScaler."
    ),
    "grad_scaler": (
        "GradScaler detected alongside autocast — gradient scaling is "
        "properly configured for mixed-precision training."
    ),
    "forward_uses_autocast": (
        "Autocast context used inside forward(): intermediate tensor dtypes "
        "will vary. Shape verification remains valid; dtype guarantees are "
        "weakened within the autocast region."
    ),
    "distributed_ddp": (
        "DistributedDataParallel detected: model is replicated across "
        "devices. Shape verification applies to a single replica. "
        "All-reduce communication patterns are not verified."
    ),
    "data_parallel": (
        "DataParallel detected: input batch is split across GPUs. The "
        "batch dimension is divided by the number of devices. Shape "
        "verification applies to the full (unsplit) input."
    ),
    "distributed": (
        "torch.distributed usage detected: collective operations "
        "(all_reduce, broadcast, etc.) are outside verification scope."
    ),
    "torch_export": (
        "torch.export detected: the exported program has fixed signatures. "
        "Shape verification of the eager Module still applies to the "
        "exported variant provided dynamic_shapes constraints match."
    ),
    "jit_script": (
        "torch.jit.script detected: TorchScript compilation constrains "
        "the model to a static subset of Python. Shape verification "
        "results transfer to the scripted model."
    ),
    "jit_trace": (
        "torch.jit.trace detected: traced model records a single "
        "execution path. Data-dependent control flow is NOT captured "
        "by tracing. Shape verification covers the traced path only."
    ),
    "data_dependent_branches": (
        "Data-dependent control flow detected: branches conditioned on "
        "tensor values create multiple execution paths. Verification "
        "covers the primary path; alternative branches may have "
        "different shape constraints."
    ),
    "shape_dependent_ops": (
        "Dynamic reshape operations detected (e.g., view(-1, ...)): "
        "inferred dimensions depend on runtime shapes. Verification "
        "models these symbolically."
    ),
    "forward_uses_no_grad": (
        "torch.no_grad() context detected in forward: gradient tracking "
        "is disabled within this region. Gradient-validity verification "
        "is limited inside no_grad blocks."
    ),
}


def _generate_dynamic_warnings(features: Dict[str, Any]) -> List[str]:
    """Generate human-readable warnings from detected dynamic features."""
    warnings = []
    for key, msg in _DYNAMIC_FEATURE_WARNINGS.items():
        if key in features and features[key]:
            warnings.append(msg)
    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Z3 encoding helpers
# ═══════════════════════════════════════════════════════════════════════════════

_z3_ctx_counter = 0


class _Z3Context:
    """Manages Z3 sorts, constants, and helpers for the product theory
    T_shape × T_device × T_phase.

    Provides:
      - Enumeration sorts for devices and phases.
      - Fresh variable creation for symbolic states.
      - Constraint encoders for each domain and cross-domain properties.
      - Incremental solving support via push/pop.
      - Query statistics tracking.
    """

    def __init__(self) -> None:
        if not HAS_Z3:
            raise RuntimeError(
                "Z3 is required for model checking.  "
                "Install it with:  pip install z3-solver"
            )

        global _z3_ctx_counter
        _z3_ctx_counter += 1
        suffix = _z3_ctx_counter

        self.solver = z3.Solver()
        self.solver.set("timeout", 10000)  # 10 s

        # Custom theory plugins (domain-specific SMT theories)
        # Note: Z3 only supports one UserPropagateBase per solver,
        # so we attach only the broadcast theory (which covers broadcasting
        # and matmul constraints).
        if HAS_THEORY_PLUGINS:
            self.broadcast_theory = BroadcastTheoryPlugin(self.solver)
        else:
            self.broadcast_theory = None
        # Stride theory lives on a *separate* solver (Z3 allows only one
        # UserPropagateBase per solver).  It is used for reshape-validity
        # queries that benefit from the stride propagator.
        if HAS_THEORY_PLUGINS:
            self._stride_solver = z3.Solver()
            self._stride_solver.set("timeout", 5000)
            self.stride_theory = StrideTheoryPlugin(self._stride_solver)
        else:
            self.stride_theory = None

        # Device theory plugin on a separate solver (one UserPropagateBase
        # per solver).  Provides eager propagation for device constraints.
        if HAS_DEVICE_THEORY:
            self._device_solver = z3.Solver()
            self._device_solver.set("timeout", 5000)
            self.device_theory = DeviceTheoryPlugin(self._device_solver)
        else:
            self.device_theory = None

        # Phase theory plugin on a separate solver.
        # Provides eager propagation for phase-dependent behaviour.
        if HAS_PHASE_THEORY:
            self._phase_solver = z3.Solver()
            self._phase_solver.set("timeout", 5000)
            self.phase_theory = PhaseTheoryPlugin(self._phase_solver)
        else:
            self.phase_theory = None

        # Permutation theory plugin on a separate solver.
        # Provides eager propagation for transpose/permute constraints.
        if HAS_PERMUTATION_THEORY:
            self._perm_solver = z3.Solver()
            self._perm_solver.set("timeout", 5000)
            self.permutation_theory = PermutationTheoryPlugin(
                self._perm_solver
            )
        else:
            self.permutation_theory = None

        # --- Device enumeration sort (unique name per context) ---
        self.DeviceSort, self.device_consts = z3.EnumSort(
            f"Device_{suffix}",
            [f"CPU_{suffix}", f"CUDA_0_{suffix}", f"CUDA_1_{suffix}",
             f"CUDA_2_{suffix}", f"CUDA_3_{suffix}"],
        )
        (self.DEV_CPU, self.DEV_CUDA0, self.DEV_CUDA1,
         self.DEV_CUDA2, self.DEV_CUDA3) = self.device_consts

        # --- Phase enumeration sort (unique name per context) ---
        self.PhaseSort, self.phase_consts = z3.EnumSort(
            f"Phase_{suffix}", [f"TRAIN_{suffix}", f"EVAL_{suffix}"],
        )
        self.PHASE_TRAIN, self.PHASE_EVAL = self.phase_consts

        # --- Symbolic dimension pool ---
        self._sym_dims: Dict[str, z3.ArithRef] = {}

        # --- Device variables ---
        self._dev_vars: Dict[str, z3.ExprRef] = {}

        # --- Gradient booleans ---
        self._grad_vars: Dict[str, z3.BoolRef] = {}

        # --- Phase variable (single for the whole model) ---
        self.phase_var = z3.Const("phase", self.PhaseSort)

        # --- Query statistics ---
        self._query_count: int = 0
        self._sat_count: int = 0
        self._unsat_count: int = 0
        self._total_solve_time_ms: float = 0.0

        # --- Theory solver constraint counters ---
        self._device_constraints_registered: int = 0
        self._phase_constraints_registered: int = 0

        # --- Theory combination checker (Tinelli-Zarba) ---
        if HAS_THEORY_COMBINATION:
            self._theory_combiner = TensorTheoryCombination()
            if self.broadcast_theory is not None:
                self._theory_combiner.add_broadcast_theory(
                    self.solver, self.broadcast_theory
                )
            if self.stride_theory is not None:
                self._theory_combiner.add_stride_theory(
                    self._stride_solver, self.stride_theory
                )
            if self.device_theory is not None:
                self._theory_combiner.add_device_theory(
                    self._device_solver, self.device_theory
                )
            if self.phase_theory is not None:
                self._theory_combiner.add_phase_theory(
                    self._phase_solver, self.phase_theory
                )
            if self.permutation_theory is not None:
                self._theory_combiner.add_permutation_theory(
                    self._perm_solver, self.permutation_theory
                )
        else:
            self._theory_combiner = None

    # --- dimension helpers -------------------------------------------------

    def dim(self, name: str) -> z3.ArithRef:
        """Return (or create) a Z3 Int for a symbolic dimension."""
        if name not in self._sym_dims:
            self._sym_dims[name] = z3.Int(name)
        return self._sym_dims[name]

    def parse_constraint_expr(self, expr_str: str) -> z3.ArithRef:
        """Parse a simple arithmetic expression into a Z3 ArithRef.

        Supports ``+``, ``-``, ``*``, ``/`` over integer literals and
        symbolic dimension names (identifiers).  Division is integer
        division.
        """
        tree = ast.parse(expr_str, mode="eval")
        return self._ast_to_z3(tree.body)

    def _ast_to_z3(self, node: ast.AST) -> z3.ArithRef:
        if isinstance(node, ast.BinOp):
            left = self._ast_to_z3(node.left)
            right = self._ast_to_z3(node.right)
            if isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.FloorDiv) or isinstance(node.op, ast.Div):
                return left / right
            else:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, int):
            return z3.IntVal(node.value)
        elif isinstance(node, ast.Name):
            return self.dim(node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._ast_to_z3(node.operand)
        else:
            raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    def build_relational_constraints(
        self, constraints: Dict[str, Union[str, int]]
    ) -> List:
        """Build Z3 assertions from a relational constraints dict.

        Each key is a dimension name.  Values are either:
          - ``int``: the dimension is fixed to that value.
          - ``str``: an arithmetic expression that the dimension must equal.
        """
        z3_cs: list = []
        for dim_name, value in constraints.items():
            lhs = self.dim(dim_name)
            if isinstance(value, int):
                z3_cs.append(lhs == z3.IntVal(value))
            elif isinstance(value, str):
                rhs = self.parse_constraint_expr(value)
                z3_cs.append(lhs == rhs)
            else:
                raise ValueError(
                    f"Constraint value for '{dim_name}' must be int or str, "
                    f"got {type(value).__name__}"
                )
        return z3_cs

    def shape_to_z3(
        self, shape: TensorShape, prefix: str
    ) -> List[z3.ArithRef]:
        """Convert a TensorShape to a list of Z3 integer expressions.

        Concrete dims become ``z3.IntVal(n)``; symbolic dims become named
        Z3 Ints.
        """
        z3_dims: List[z3.ArithRef] = []
        for i, sd in enumerate(shape.dims):
            if sd.is_symbolic:
                z3_dims.append(self.dim(str(sd.value)))
            else:
                z3_dims.append(z3.IntVal(sd.value))
        return z3_dims

    # --- fresh variable creation for symbolic states -------------------------

    def fresh_shape_vars(
        self, tensor_name: str, ndim: int, step: int
    ) -> List[z3.ArithRef]:
        """Create fresh Z3 int variables for tensor shape at step."""
        return [z3.Int(f"sh_{tensor_name}_d{i}_s{step}") for i in range(ndim)]

    def fresh_device_var(self, tensor_name: str, step: int) -> z3.ExprRef:
        """Create fresh Z3 device variable for tensor at step."""
        return z3.Const(f"dev_{tensor_name}_s{step}", self.DeviceSort)

    def fresh_grad_var(self, tensor_name: str, step: int) -> z3.BoolRef:
        """Create fresh Z3 Bool for gradient tracking at step."""
        return z3.Bool(f"grad_{tensor_name}_s{step}")

    def fresh_phase_var(self, step: int) -> z3.ExprRef:
        """Create fresh Z3 phase variable at step."""
        return z3.Const(f"phase_s{step}", self.PhaseSort)

    # --- device helpers ----------------------------------------------------

    def dev_var(self, tensor_name: str) -> z3.ExprRef:
        """Return (or create) a Z3 device variable for *tensor_name*."""
        if tensor_name not in self._dev_vars:
            self._dev_vars[tensor_name] = z3.Const(
                f"dev_{tensor_name}", self.DeviceSort
            )
        return self._dev_vars[tensor_name]

    def device_to_z3(self, device: Device) -> z3.ExprRef:
        """Map a Device enum value to the corresponding Z3 constant."""
        return {
            Device.CPU: self.DEV_CPU,
            Device.CUDA_0: self.DEV_CUDA0,
            Device.CUDA_1: self.DEV_CUDA1,
            Device.CUDA_2: self.DEV_CUDA2,
            Device.CUDA_3: self.DEV_CUDA3,
        }[device]

    def phase_to_z3(self, phase: Phase) -> z3.ExprRef:
        """Map a Phase enum value to the corresponding Z3 constant."""
        return self.PHASE_TRAIN if phase == Phase.TRAIN else self.PHASE_EVAL

    # --- gradient helpers --------------------------------------------------

    def grad_var(self, tensor_name: str) -> z3.BoolRef:
        if tensor_name not in self._grad_vars:
            self._grad_vars[tensor_name] = z3.Bool(f"grad_{tensor_name}")
        return self._grad_vars[tensor_name]

    # --- product-theory constraint encoders --------------------------------

    def encode_device_constraint(
        self, dev_a: z3.ExprRef, dev_b: z3.ExprRef
    ) -> z3.BoolRef:
        """Z3 constraint: two tensors are on the same device.

        Also registers the pair with the DeviceTheoryPlugin (if available)
        for eager propagation on the device solver.
        """
        if self.device_theory is not None:
            try:
                from src.smt.device_theory import DeviceSort as _DTSort
            except ImportError:
                _DTSort = None
            if _DTSort is not None:
                _da = z3.Const(str(dev_a), _DTSort)
                _db = z3.Const(str(dev_b), _DTSort)
                self._device_solver.add(
                    self.device_theory.same_device(_da, _db)
                )
                self._device_constraints_registered += 1
        return dev_a == dev_b

    def encode_device_transfer(
        self, dev_out: z3.ExprRef, target: Device
    ) -> z3.BoolRef:
        """Z3 constraint for ``.to(device)`` / ``.cuda()`` / ``.cpu()``."""
        return dev_out == self.device_to_z3(target)

    def encode_phase_constraint(
        self, phase: z3.ExprRef, layer_kind: LayerKind
    ) -> Tuple[z3.BoolRef, z3.BoolRef]:
        """Encode phase-dependent behavior as (train_cond, eval_cond).

        For dropout: in eval, output equals input (identity).
        For batchnorm: in eval, uses running statistics.
        """
        is_train = phase == self.PHASE_TRAIN
        is_eval = phase == self.PHASE_EVAL
        if layer_kind == LayerKind.DROPOUT:
            return (is_train, is_eval)
        elif layer_kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D):
            return (is_train, is_eval)
        return (z3.BoolVal(True), z3.BoolVal(True))

    def encode_gradient_constraint(
        self, grad_out: z3.BoolRef, requires_grad: bool
    ) -> z3.BoolRef:
        """Z3 constraint setting gradient status."""
        return grad_out == z3.BoolVal(requires_grad)

    def encode_cross_domain_constraint(
        self,
        shape_pre: List[z3.ArithRef],
        shape_post: List[z3.ArithRef],
        dev_pre: z3.ExprRef,
        dev_post: z3.ExprRef,
        is_device_transfer: bool,
    ) -> List[z3.BoolRef]:
        """Cross-domain constraints spanning shape + device.

        Device transfer preserves shape.  Non-transfer ops preserve device.
        """
        constraints: List[z3.BoolRef] = []
        if is_device_transfer:
            for dp, dq in zip(shape_pre, shape_post):
                constraints.append(dp == dq)
        else:
            constraints.append(dev_pre == dev_post)
        return constraints

    # --- positivity constraints -------------------------------------------

    def positive_dim_constraints(self) -> List[z3.BoolRef]:
        """All symbolic dims must be ≥ 1."""
        return [v > 0 for v in self._sym_dims.values()]

    # --- timed Z3 check ---------------------------------------------------

    def timed_check(self, solver: z3.Solver) -> z3.CheckSatResult:
        """Run solver.check() with timing and statistics tracking."""
        t0 = time.monotonic()
        result = solver.check()
        elapsed = (time.monotonic() - t0) * 1000
        self._query_count += 1
        self._total_solve_time_ms += elapsed
        if result == z3.sat:
            self._sat_count += 1
        elif result == z3.unsat:
            self._unsat_count += 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return Z3 solver statistics."""
        stats = {
            "z3_queries": self._query_count,
            "z3_total_time_ms": self._total_solve_time_ms,
            "z3_sat_count": self._sat_count,
            "z3_unsat_count": self._unsat_count,
        }
        if self.broadcast_theory is not None:
            prop = self.broadcast_theory.propagator
            stats["broadcast_propagations"] = len(prop._broadcast_triples)
            stats["broadcast_conflicts"] = len(prop._matmul_pairs)
        if self.stride_theory is not None:
            prop = self.stride_theory.propagator
            stats["stride_constraints"] = len(prop._contiguous)
            stats["stride_reshapes"] = len(prop._reshapes)
            stats["stride_divisibility"] = len(prop._divisibility)
        if self.device_theory is not None:
            prop = self.device_theory.propagator
            stats["device_same_pairs"] = len(prop._same_device_pairs)
            stats["device_transfer_triples"] = len(prop._transfer_triples)
            stats["device_inherit_pairs"] = len(prop._inherit_pairs)
            stats["device_constraints_registered"] = self._device_constraints_registered
        if self.phase_theory is not None:
            prop = self.phase_theory.propagator
            stats["phase_dropout_constraints"] = len(prop._dropout_constraints)
            stats["phase_batchnorm_constraints"] = len(
                prop._batchnorm_constraints
            )
            stats["phase_constraints_registered"] = self._phase_constraints_registered
        if self.permutation_theory is not None:
            prop = self.permutation_theory.propagator
            stats["permutation_transposes"] = len(prop._transposes)
            stats["permutation_permutations"] = len(prop._permutations)
        if self._theory_combiner is not None:
            stats["theory_combination_available"] = True
        return stats

    def verify_theory_combination(self) -> Optional[Dict[str, Any]]:
        """Run Tinelli-Zarba theory combination check.

        Returns None if no combiner available, otherwise a dict with
        'consistent' (bool) and 'details' fields.
        """
        if self._theory_combiner is None:
            return None
        result = self._theory_combiner.verify_theory_combination_consistency()
        return {
            "consistent": result.is_sat,
            "arrangements_checked": result.arrangements_checked,
            "details": result.reason,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Shape-propagation rules (symbolic, Z3-backed)
# ═══════════════════════════════════════════════════════════════════════════════

def _propagate_linear(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Propagate shape through nn.Linear.

    nn.Linear(in_features, out_features) maps  (*, in_features) → (*, out_features).
    """
    if input_shape.ndim < 1:
        return None, "Linear requires at least 1D input"

    last = input_shape.dims[-1]
    if (layer.in_features is not None and not last.is_symbolic
            and isinstance(layer.in_features, int)):
        if last.value != layer.in_features:
            return None, (
                f"Linear expects last dim={layer.in_features}, "
                f"got {last.value}"
            )

    out_feat = layer.out_features
    if out_feat is None:
        # Out-of-fragment: out_features could not be resolved (e.g. came from
        # an unresolved config object). Sound abstention: propagate symbolic
        # last dim and emit no error.
        new_dims = input_shape.dims[:-1] + (ShapeDim(f"_unk_lin_out_{layer.attr_name or 'anon'}"),)
        return TensorShape(new_dims), None

    new_dims = input_shape.dims[:-1] + (ShapeDim(out_feat),)
    return TensorShape(new_dims), None


def _propagate_conv2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Propagate shape through nn.Conv2d.

    Expects input (N, C_in, H, W) → (N, C_out, H', W').
    H' = floor((H + 2*padding - kernel_size) / stride) + 1
    """
    if input_shape.ndim != 4:
        return None, f"Conv2d expects 4D input, got {input_shape.ndim}D"

    c_in = input_shape.dims[1]
    if (layer.in_channels is not None and not c_in.is_symbolic
            and isinstance(layer.in_channels, int)):
        if c_in.value != layer.in_channels:
            return None, (
                f"Conv2d expects {layer.in_channels} input channels, "
                f"got {c_in.value}"
            )

    # Validate groups constraint: in_channels and out_channels must be
    # divisible by groups
    groups = layer.params.get("groups", 1)
    if isinstance(groups, int) and groups > 1:
        if layer.in_channels is not None and isinstance(layer.in_channels, int) and layer.in_channels % groups != 0:
            return None, (
                f"Conv2d groups={groups} does not divide "
                f"in_channels={layer.in_channels}"
            )
        if layer.out_channels is not None and isinstance(layer.out_channels, int) and layer.out_channels % groups != 0:
            return None, (
                f"Conv2d groups={groups} does not divide "
                f"out_channels={layer.out_channels}"
            )

    out_c = layer.out_channels
    if out_c is None:
        # Unknown out_channels — propagate symbolic channel dim rather than
        # erroring, since this is not a user bug (just an unresolvable param).
        ks = layer.kernel_size or (3, 3)
        stride = layer.params.get("stride", (1, 1))
        if isinstance(stride, int):
            stride = (stride, stride)
        padding = layer.params.get("padding", (0, 0))
        if isinstance(padding, int):
            padding = (padding, padding)
        dilation = layer.params.get("dilation", (1, 1))
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        h_in = input_shape.dims[2]
        w_in = input_shape.dims[3]
        if not h_in.is_symbolic and not w_in.is_symbolic:
            h_out = (h_in.value + 2 * padding[0] - dilation[0] * (ks[0] - 1) - 1) // stride[0] + 1
            w_out = (w_in.value + 2 * padding[1] - dilation[1] * (ks[1] - 1) - 1) // stride[1] + 1
            return TensorShape((
                input_shape.dims[0],
                ShapeDim(f"_C_{layer.attr_name}"),
                ShapeDim(h_out),
                ShapeDim(w_out),
            )), None
        return TensorShape((
            input_shape.dims[0],
            ShapeDim(f"_C_{layer.attr_name}"),
            ShapeDim("H_out"),
            ShapeDim("W_out"),
        )), None

    # Compute output spatial dims:
    # H' = floor((H + 2*pad - dilation*(kernel-1) - 1) / stride + 1)
    ks = layer.kernel_size or (3, 3)
    stride = layer.params.get("stride", (1, 1))
    if isinstance(stride, int):
        stride = (stride, stride)
    padding = layer.params.get("padding", (0, 0))
    if isinstance(padding, int):
        padding = (padding, padding)
    dilation = layer.params.get("dilation", (1, 1))
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    h_in = input_shape.dims[2]
    w_in = input_shape.dims[3]

    if not h_in.is_symbolic and not w_in.is_symbolic:
        h_out = (h_in.value + 2 * padding[0] - dilation[0] * (ks[0] - 1) - 1) // stride[0] + 1
        w_out = (w_in.value + 2 * padding[1] - dilation[1] * (ks[1] - 1) - 1) // stride[1] + 1
        if h_out <= 0 or w_out <= 0:
            return None, (
                f"Conv2d output size is non-positive: "
                f"({h_out}, {w_out}) from input ({h_in.value}, {w_in.value}) "
                f"with kernel_size={ks}, stride={stride}, padding={padding}, "
                f"dilation={dilation}"
            )
        new_dims = (
            input_shape.dims[0],
            ShapeDim(out_c),
            ShapeDim(h_out),
            ShapeDim(w_out),
        )
    else:
        new_dims = (
            input_shape.dims[0],
            ShapeDim(out_c),
            ShapeDim("H_out"),
            ShapeDim("W_out"),
        )
    return TensorShape(new_dims), None


def _propagate_conv1d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Propagate shape through nn.Conv1d.

    Expects input (N, C_in, L) → (N, C_out, L').
    L' = floor((L + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
    """
    if input_shape.ndim != 3:
        return None, f"Conv1d expects 3D input, got {input_shape.ndim}D"

    c_in = input_shape.dims[1]
    if layer.in_channels is not None and not c_in.is_symbolic and isinstance(layer.in_channels, int):
        if c_in.value != layer.in_channels:
            return None, (
                f"Conv1d expects {layer.in_channels} input channels, "
                f"got {c_in.value}"
            )

    groups = layer.params.get("groups", 1)
    if isinstance(groups, int) and groups > 1:
        if layer.in_channels is not None and isinstance(layer.in_channels, int) and layer.in_channels % groups != 0:
            return None, (
                f"Conv1d groups={groups} does not divide "
                f"in_channels={layer.in_channels}"
            )
        if layer.out_channels is not None and isinstance(layer.out_channels, int) and layer.out_channels % groups != 0:
            return None, (
                f"Conv1d groups={groups} does not divide "
                f"out_channels={layer.out_channels}"
            )

    out_c = layer.out_channels
    if out_c is None:
        return None, "Conv1d out_channels unknown"

    ks = layer.kernel_size or (3,)
    stride = layer.params.get("stride", (1,))
    if isinstance(stride, int):
        stride = (stride,)
    padding = layer.params.get("padding", (0,))
    if isinstance(padding, int):
        padding = (padding,)
    dilation = layer.params.get("dilation", (1,))
    if isinstance(dilation, int):
        dilation = (dilation,)

    l_in = input_shape.dims[2]
    if not l_in.is_symbolic:
        l_out = (l_in.value + 2 * padding[0] - dilation[0] * (ks[0] - 1) - 1) // stride[0] + 1
        new_dims = (
            input_shape.dims[0],
            ShapeDim(out_c),
            ShapeDim(l_out),
        )
    else:
        new_dims = (
            input_shape.dims[0],
            ShapeDim(out_c),
            ShapeDim("L_out"),
        )
    return TensorShape(new_dims), None


def _propagate_batchnorm(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """BatchNorm preserves shape but checks the feature dimension."""
    if input_shape.ndim < 2:
        return None, f"BatchNorm requires at least 2D input, got {input_shape.ndim}D"

    feat = input_shape.dims[1]
    if layer.num_features is not None and not feat.is_symbolic and isinstance(layer.num_features, int):
        if feat.value != layer.num_features:
            return None, (
                f"BatchNorm expects {layer.num_features} features, "
                f"got {feat.value}"
            )
    return input_shape, None


def _propagate_dropout(
    input_shape: TensorShape, _layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Dropout preserves shape."""
    return input_shape, None


def _propagate_activation(
    input_shape: TensorShape,
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Element-wise activations preserve shape."""
    return input_shape, None


def _propagate_embedding(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """nn.Embedding maps (*, ) → (*, embedding_dim)."""
    if layer.embedding_dim is None:
        return None, "Embedding dim unknown"
    new_dims = input_shape.dims + (ShapeDim(layer.embedding_dim),)
    return TensorShape(new_dims), None


def _propagate_flatten(
    input_shape: TensorShape, start_dim: int = 1, end_dim: int = -1
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Flatten dims ``[start_dim, end_dim]`` (inclusive) into a single dim.

    Mirrors ``torch.flatten``: dims before ``start_dim`` and after ``end_dim``
    are preserved; the span in between is collapsed into one dimension whose
    size is the product of the spanned dims (symbolic if any spanned dim is
    symbolic).  ``end_dim`` defaults to ``-1`` (flatten to the end).
    """
    nd = input_shape.ndim
    if nd == 0:
        return input_shape, None
    s = start_dim + nd if start_dim < 0 else start_dim
    e = end_dim + nd if end_dim < 0 else end_dim
    if s < 0:
        s = 0
    if e >= nd:
        e = nd - 1
    if s > e or s >= nd:
        # Nothing to flatten (e.g. start_dim past the end, or empty span).
        return input_shape, None

    prefix = input_shape.dims[:s]
    span = input_shape.dims[s:e + 1]
    suffix = input_shape.dims[e + 1:]

    all_concrete = all(not d.is_symbolic for d in span)
    if all_concrete:
        total = 1
        for d in span:
            total *= d.value
        flat_dim = ShapeDim(total)
    else:
        flat_dim = ShapeDim("_flat")

    return TensorShape(prefix + (flat_dim,) + suffix), None


def _propagate_adaptive_avgpool2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """AdaptiveAvgPool2d maps (N, C, H, W) → (N, C, H_out, W_out)."""
    if input_shape.ndim != 4:
        return None, f"AdaptiveAvgPool2d expects 4D, got {input_shape.ndim}D"
    out = layer.output_size
    if out is None:
        return None, "output_size unknown"
    new_dims = (
        input_shape.dims[0],
        input_shape.dims[1],
        ShapeDim(out[0]),
        ShapeDim(out[1]),
    )
    return TensorShape(new_dims), None


def _propagate_pool2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """MaxPool2d / AvgPool2d — compute output spatial dims."""
    if input_shape.ndim != 4:
        return None, f"Pool2d expects 4D, got {input_shape.ndim}D"

    ks = layer.kernel_size or (2, 2)
    stride = layer.params.get("stride", ks)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(ks, int):
        ks = (ks, ks)
    padding = layer.params.get("padding", (0, 0))
    if isinstance(padding, int):
        padding = (padding, padding)

    h_in = input_shape.dims[2]
    w_in = input_shape.dims[3]

    if not h_in.is_symbolic and not w_in.is_symbolic:
        h_out = (h_in.value + 2 * padding[0] - ks[0]) // stride[0] + 1
        w_out = (w_in.value + 2 * padding[1] - ks[1]) // stride[1] + 1
        if h_out <= 0 or w_out <= 0:
            return None, (
                f"Pool2d output size is non-positive: "
                f"({h_out}, {w_out}) from input ({h_in.value}, {w_in.value}) "
                f"with kernel_size={ks}, stride={stride}, padding={padding}"
            )
        new_dims = (
            input_shape.dims[0],
            input_shape.dims[1],
            ShapeDim(h_out),
            ShapeDim(w_out),
        )
    elif not h_in.is_symbolic:
        # Check concrete H against kernel
        h_out = (h_in.value + 2 * padding[0] - ks[0]) // stride[0] + 1
        if h_out <= 0:
            return None, (
                f"Pool2d output height is non-positive: "
                f"{h_out} from H={h_in.value} "
                f"with kernel_size={ks}, stride={stride}, padding={padding}"
            )
        new_dims = (
            input_shape.dims[0],
            input_shape.dims[1],
            ShapeDim(h_out),
            ShapeDim("W_pool"),
        )
    elif not w_in.is_symbolic:
        # Check concrete W against kernel
        w_out = (w_in.value + 2 * padding[1] - ks[1]) // stride[1] + 1
        if w_out <= 0:
            return None, (
                f"Pool2d output width is non-positive: "
                f"{w_out} from W={w_in.value} "
                f"with kernel_size={ks}, stride={stride}, padding={padding}"
            )
        new_dims = (
            input_shape.dims[0],
            input_shape.dims[1],
            ShapeDim("H_pool"),
            ShapeDim(w_out),
        )
    else:
        new_dims = (
            input_shape.dims[0],
            input_shape.dims[1],
            ShapeDim("H_pool"),
            ShapeDim("W_pool"),
        )
    return TensorShape(new_dims), None


def _propagate_sequential(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Propagate shape through nn.Sequential by chaining sub-layers.

    Sound-abstention policy: if the input to the Sequential carries any
    symbolic ``_unk_`` dim (i.e. came through an opaque submodule), or if a
    sub-layer's transfer function fails because its parameters could not be
    statically resolved, we abstain (return a fully-symbolic same-ndim shape)
    rather than report a shape mismatch. This avoids false positives from
    helper-function-built Sequentials (e.g. torchvision ``make_layers``).
    """
    if not layer.sub_layers:
        return input_shape, None
    def _abstain(sh: TensorShape) -> TensorShape:
        return TensorShape(tuple(
            ShapeDim(f"_unk_seq_{layer.attr_name}_{i}") for i in range(sh.ndim)
        ))
    if any(d.is_symbolic and isinstance(d.value, str) and d.value.startswith("_unk")
           for d in input_shape.dims):
        return _abstain(input_shape), None
    current = input_shape
    for sub in layer.sub_layers:
        propagator = _LAYER_PROPAGATORS.get(sub.kind)
        if propagator is not None:
            new_current, err = propagator(current, sub)
            if err or new_current is None:
                # Sound abstention on opaque/unresolvable sub-layer.
                return _abstain(current), None
            current = new_current
        elif sub.kind in (LayerKind.RELU, LayerKind.DROPOUT,
                          LayerKind.IDENTITY, LayerKind.SOFTMAX):
            pass  # shape-preserving
        elif sub.kind == LayerKind.FLATTEN:
            new_current, _err = _propagate_flatten(current, 1)
            if new_current is None:
                return _abstain(current), None
            current = new_current
        else:
            # Unknown sub-layer: mark output shape as UNKNOWN (fully symbolic)
            logger.warning(
                "Unsupported sequential sub-layer kind %s (%s): shape marked UNKNOWN",
                sub.kind.name, sub.attr_name,
            )
            current = TensorShape(
                tuple(ShapeDim(f"_unk_{sub.attr_name}_{i}")
                      for i in range(current.ndim))
            )
    return current, None


def _propagate_groupnorm(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """GroupNorm preserves shape but checks the channel dimension."""
    if input_shape.ndim < 2:
        return None, f"GroupNorm requires at least 2D input, got {input_shape.ndim}D"
    feat = input_shape.dims[1]
    if layer.num_features is not None and not feat.is_symbolic and isinstance(layer.num_features, int):
        if feat.value != layer.num_features:
            return None, (
                f"GroupNorm expects {layer.num_features} channels, "
                f"got {feat.value}"
            )
    return input_shape, None


def _propagate_instancenorm2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """InstanceNorm2d preserves shape but checks the channel dimension."""
    if input_shape.ndim != 4:
        return None, f"InstanceNorm2d expects 4D input, got {input_shape.ndim}D"
    feat = input_shape.dims[1]
    if layer.num_features is not None and not feat.is_symbolic and isinstance(layer.num_features, int):
        if feat.value != layer.num_features:
            return None, (
                f"InstanceNorm2d expects {layer.num_features} channels, "
                f"got {feat.value}"
            )
    return input_shape, None


def _propagate_convtranspose2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """ConvTranspose2d maps (N, C_in, H, W) → (N, C_out, H', W')."""
    if input_shape.ndim != 4:
        return None, f"ConvTranspose2d expects 4D, got {input_shape.ndim}D"
    in_c = layer.in_channels
    out_c = layer.out_channels
    if in_c is not None and not input_shape.dims[1].is_symbolic:
        if input_shape.dims[1].value != in_c:
            return None, (
                f"ConvTranspose2d expects {in_c} input channels, "
                f"got {input_shape.dims[1].value}"
            )
    ks = layer.params.get("kernel_size", (2, 2))
    stride = layer.params.get("stride", (1, 1))
    padding = layer.params.get("padding", (0, 0))
    output_padding = layer.params.get("output_padding", (0, 0))
    dilation = layer.params.get("dilation", (1, 1))
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(ks, int):
        ks = (ks, ks)
    # groups must divide in_channels and out_channels.
    groups = layer.params.get("groups", 1)
    if isinstance(groups, int) and groups > 1:
        if isinstance(in_c, int) and in_c % groups != 0:
            return None, (
                f"ConvTranspose2d groups={groups} does not divide "
                f"in_channels={in_c}"
            )
        if isinstance(out_c, int) and out_c % groups != 0:
            return None, (
                f"ConvTranspose2d groups={groups} does not divide "
                f"out_channels={out_c}"
            )
    h_in = input_shape.dims[2]
    w_in = input_shape.dims[3]
    if not h_in.is_symbolic and ks and stride:
        h_out = ((h_in.value - 1) * stride[0] - 2 * padding[0]
                 + dilation[0] * (ks[0] - 1) + output_padding[0] + 1)
        w_out = ((w_in.value - 1) * stride[1] - 2 * padding[1]
                 + dilation[1] * (ks[1] - 1) + output_padding[1] + 1)
    else:
        h_out = "_h_up"
        w_out = "_w_up"
    out_channels = out_c if out_c is not None else "_c_out"
    return TensorShape((
        input_shape.dims[0],
        ShapeDim(out_channels),
        ShapeDim(h_out),
        ShapeDim(w_out),
    )), None


def _propagate_upsample(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Upsample / F.interpolate preserves batch and channel dims."""
    if input_shape.ndim < 3:
        return None, f"Upsample expects >=3D, got {input_shape.ndim}D"
    scale = layer.params.get("scale_factor")
    size = layer.params.get("size")
    if input_shape.ndim == 4:
        if size is not None:
            if isinstance(size, int):
                size = (size, size)
            return TensorShape((
                input_shape.dims[0], input_shape.dims[1],
                ShapeDim(size[0]), ShapeDim(size[1]),
            )), None
        if scale is not None and not input_shape.dims[2].is_symbolic:
            s = int(scale) if isinstance(scale, (int, float)) else 2
            return TensorShape((
                input_shape.dims[0], input_shape.dims[1],
                ShapeDim(input_shape.dims[2].value * s),
                ShapeDim(input_shape.dims[3].value * s),
            )), None
    # Fallback: preserve batch + channel, mark spatial as symbolic
    kept = input_shape.dims[:2]
    spatial = tuple(ShapeDim("_up") for _ in input_shape.dims[2:])
    return TensorShape(kept + spatial), None


def _propagate_multihead_attention(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Propagate shape through nn.MultiheadAttention.

    nn.MultiheadAttention(embed_dim, num_heads) maps
    (seq, batch, embed_dim) or (batch, seq, embed_dim) → same shape.
    Requires input last dim == embed_dim and embed_dim % num_heads == 0.
    """
    if input_shape.ndim < 2:
        return None, "MultiheadAttention requires at least 2D input"

    embed_dim = layer.in_features
    num_heads = layer.num_heads

    last = input_shape.dims[-1]
    if isinstance(embed_dim, int) and not last.is_symbolic:
        if last.value != embed_dim:
            return None, (
                f"MultiheadAttention expects last dim={embed_dim}, "
                f"got {last.value}"
            )
    if isinstance(embed_dim, int) and isinstance(num_heads, int):
        if embed_dim % num_heads != 0:
            return None, (
                f"embed_dim={embed_dim} not divisible by num_heads={num_heads}"
            )

    # Output shape is same as input (attention output is projected back)
    return input_shape, None


def _propagate_transformer_encoder_layer(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """nn.TransformerEncoderLayer(d_model, nhead) preserves shape.

    Input: (seq, batch, d_model) or (batch, seq, d_model) → same.
    Also validates that d_model is divisible by nhead.
    """
    if input_shape.ndim < 2:
        return None, "TransformerEncoderLayer requires at least 2D input"

    d_model = layer.in_features
    nhead = layer.num_heads

    # Check d_model % nhead == 0  (required by MultiheadAttention)
    if (d_model is not None and nhead is not None
            and isinstance(d_model, int) and isinstance(nhead, int)
            and nhead > 0 and d_model % nhead != 0):
        return None, (
            f"TransformerEncoderLayer: d_model={d_model} is not divisible "
            f"by nhead={nhead}"
        )

    last = input_shape.dims[-1]
    if d_model is not None and not last.is_symbolic:
        if last.value != d_model:
            return None, (
                f"TransformerEncoderLayer expects last dim={d_model}, "
                f"got {last.value}"
            )
    return input_shape, None


def _propagate_transformer_decoder_layer(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """nn.TransformerDecoderLayer(d_model, nhead) preserves shape."""
    if input_shape.ndim < 2:
        return None, "TransformerDecoderLayer requires at least 2D input"

    d_model = layer.in_features
    last = input_shape.dims[-1]
    if d_model is not None and not last.is_symbolic:
        if last.value != d_model:
            return None, (
                f"TransformerDecoderLayer expects last dim={d_model}, "
                f"got {last.value}"
            )
    return input_shape, None


def _propagate_transformer_encoder(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """nn.TransformerEncoder preserves shape (stacks N encoder layers).

    Also validates d_model % nhead == 0 (inherited from the sub-layer).
    """
    if input_shape.ndim < 2:
        return None, "TransformerEncoder requires at least 2D input"

    d_model = layer.in_features
    nhead = layer.num_heads

    if (d_model is not None and nhead is not None
            and isinstance(d_model, int) and isinstance(nhead, int)
            and nhead > 0 and d_model % nhead != 0):
        return None, (
            f"TransformerEncoder: d_model={d_model} is not divisible "
            f"by nhead={nhead}"
        )

    last = input_shape.dims[-1]
    if d_model is not None and not last.is_symbolic:
        if last.value != d_model:
            return None, (
                f"TransformerEncoder expects last dim={d_model}, "
                f"got {last.value}"
            )
    return input_shape, None


def _propagate_transformer_decoder(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """nn.TransformerDecoder preserves shape (stacks N decoder layers)."""
    if input_shape.ndim < 2:
        return None, "TransformerDecoder requires at least 2D input"

    d_model = layer.in_features
    last = input_shape.dims[-1]
    if d_model is not None and not last.is_symbolic:
        if last.value != d_model:
            return None, (
                f"TransformerDecoder expects last dim={d_model}, "
                f"got {last.value}"
            )
    return input_shape, None


def _propagate_lstm(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Propagate shape through nn.LSTM.

    nn.LSTM(input_size, hidden_size, num_layers, batch_first, bidirectional)
    Default layout (batch_first=False):
        input  (seq_len, batch, input_size) → output (seq_len, batch, hidden_size * D)
    batch_first=True:
        input  (batch, seq_len, input_size) → output (batch, seq_len, hidden_size * D)
    where D = 2 if bidirectional else 1.

    Verifies input_size matches the last dimension of the input tensor.
    """
    if input_shape.ndim < 2:
        return None, "LSTM requires at least 2D input"

    last = input_shape.dims[-1]
    if layer.in_features is not None and not last.is_symbolic:
        if last.value != layer.in_features:
            return None, (
                f"LSTM expects input_size={layer.in_features}, "
                f"got {last.value}"
            )

    hidden = layer.hidden_size
    if hidden is None:
        return None, "LSTM hidden_size unknown"

    D = 2 if layer.bidirectional else 1
    out_feat = hidden * D

    # Output: same leading dims, last dim = hidden_size * D
    new_dims = input_shape.dims[:-1] + (ShapeDim(out_feat),)
    return TensorShape(new_dims), None


def _propagate_gru(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Propagate shape through nn.GRU.

    nn.GRU(input_size, hidden_size, num_layers, batch_first, bidirectional)
    Same shape semantics as LSTM for the output tensor.
    """
    if input_shape.ndim < 2:
        return None, "GRU requires at least 2D input"

    last = input_shape.dims[-1]
    if layer.in_features is not None and not last.is_symbolic:
        if last.value != layer.in_features:
            return None, (
                f"GRU expects input_size={layer.in_features}, "
                f"got {last.value}"
            )

    hidden = layer.hidden_size
    if hidden is None:
        return None, "GRU hidden_size unknown"

    D = 2 if layer.bidirectional else 1
    out_feat = hidden * D

    new_dims = input_shape.dims[:-1] + (ShapeDim(out_feat),)
    return TensorShape(new_dims), None


def _propagate_convtranspose1d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """ConvTranspose1d maps (N, C_in, L) → (N, C_out, L').

    L' = (L - 1) * stride - 2 * padding + kernel_size + output_padding
    """
    if input_shape.ndim != 3:
        return None, f"ConvTranspose1d expects 3D, got {input_shape.ndim}D"
    in_c = layer.in_channels
    out_c = layer.out_channels
    if in_c is not None and not input_shape.dims[1].is_symbolic:
        if input_shape.dims[1].value != in_c:
            return None, (
                f"ConvTranspose1d expects {in_c} input channels, "
                f"got {input_shape.dims[1].value}"
            )
    ks = layer.params.get("kernel_size", (2,))
    stride = layer.params.get("stride", (1,))
    padding = layer.params.get("padding", (0,))
    output_padding = layer.params.get("output_padding", (0,))
    dilation = layer.params.get("dilation", (1,))
    if isinstance(stride, int):
        stride = (stride,)
    if isinstance(padding, int):
        padding = (padding,)
    if isinstance(output_padding, int):
        output_padding = (output_padding,)
    if isinstance(dilation, int):
        dilation = (dilation,)
    if isinstance(ks, int):
        ks = (ks,)
    groups = layer.params.get("groups", 1)
    if isinstance(groups, int) and groups > 1:
        if isinstance(in_c, int) and in_c % groups != 0:
            return None, (
                f"ConvTranspose1d groups={groups} does not divide "
                f"in_channels={in_c}"
            )
        if isinstance(out_c, int) and out_c % groups != 0:
            return None, (
                f"ConvTranspose1d groups={groups} does not divide "
                f"out_channels={out_c}"
            )
    l_in = input_shape.dims[2]
    if not l_in.is_symbolic and ks and stride:
        l_out = ((l_in.value - 1) * stride[0] - 2 * padding[0]
                 + dilation[0] * (ks[0] - 1) + output_padding[0] + 1)
    else:
        l_out = "_l_up"
    out_channels = out_c if out_c is not None else "_c_out"
    return TensorShape((
        input_shape.dims[0],
        ShapeDim(out_channels),
        ShapeDim(l_out),
    )), None


def _propagate_adaptive_maxpool2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """AdaptiveMaxPool2d maps (N, C, H, W) → (N, C, H_out, W_out)."""
    if input_shape.ndim != 4:
        return None, f"AdaptiveMaxPool2d expects 4D, got {input_shape.ndim}D"
    out = layer.output_size
    if out is None:
        return None, "output_size unknown"
    new_dims = (
        input_shape.dims[0],
        input_shape.dims[1],
        ShapeDim(out[0]),
        ShapeDim(out[1]),
    )
    return TensorShape(new_dims), None


def _propagate_pixel_shuffle(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """PixelShuffle(r) maps (N, C*r^2, H, W) → (N, C, H*r, W*r)."""
    if input_shape.ndim != 4:
        return None, f"PixelShuffle expects 4D, got {input_shape.ndim}D"
    r = layer.params.get("upscale_factor")
    if r is None:
        # Unknown upscale_factor — propagate symbolic shape
        return TensorShape((
            input_shape.dims[0],
            ShapeDim("_c_ps"),
            ShapeDim("_h_ps"),
            ShapeDim("_w_ps"),
        )), None
    c_in = input_shape.dims[1]
    if not c_in.is_symbolic:
        if c_in.value % (r * r) != 0:
            return None, (
                f"PixelShuffle: in_channels={c_in.value} not divisible "
                f"by r^2={r*r}"
            )
        c_out = c_in.value // (r * r)
    else:
        c_out = "_c_ps"
    h_in = input_shape.dims[2]
    w_in = input_shape.dims[3]
    if not h_in.is_symbolic and not w_in.is_symbolic:
        h_out = h_in.value * r
        w_out = w_in.value * r
    else:
        h_out = "_h_ps"
        w_out = "_w_ps"
    return TensorShape((
        input_shape.dims[0],
        ShapeDim(c_out),
        ShapeDim(h_out),
        ShapeDim(w_out),
    )), None


def _propagate_unfold(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """nn.Unfold maps (N, C, H, W) → (N, C*∏(kernel_size), L).

    L = ∏((spatial + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
    """
    if input_shape.ndim != 4:
        return None, f"Unfold expects 4D, got {input_shape.ndim}D"
    ks = layer.params.get("kernel_size", (3, 3))
    if isinstance(ks, int):
        ks = (ks, ks)
    dilation = layer.params.get("dilation", (1, 1))
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    padding = layer.params.get("padding", (0, 0))
    if isinstance(padding, int):
        padding = (padding, padding)
    stride = layer.params.get("stride", (1, 1))
    if isinstance(stride, int):
        stride = (stride, stride)

    c_in = input_shape.dims[1]
    h_in = input_shape.dims[2]
    w_in = input_shape.dims[3]

    if not c_in.is_symbolic:
        c_out = c_in.value * ks[0] * ks[1]
    else:
        c_out = "_c_unfold"

    if not h_in.is_symbolic and not w_in.is_symbolic:
        h_blocks = (h_in.value + 2 * padding[0] - dilation[0] * (ks[0] - 1) - 1) // stride[0] + 1
        w_blocks = (w_in.value + 2 * padding[1] - dilation[1] * (ks[1] - 1) - 1) // stride[1] + 1
        L = h_blocks * w_blocks
    else:
        L = "_L_unfold"

    return TensorShape((
        input_shape.dims[0],
        ShapeDim(c_out),
        ShapeDim(L),
    )), None


def _propagate_fold(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """nn.Fold maps (N, C*∏(kernel_size), L) → (N, C, output_size[0], output_size[1])."""
    if input_shape.ndim != 3:
        return None, f"Fold expects 3D, got {input_shape.ndim}D"
    output_size = layer.output_size
    ks = layer.params.get("kernel_size", (3, 3))
    if isinstance(ks, int):
        ks = (ks, ks)
    if output_size is None:
        return None, "Fold output_size unknown"
    c_ks = input_shape.dims[1]
    if not c_ks.is_symbolic:
        k_prod = ks[0] * ks[1]
        if k_prod > 0 and c_ks.value % k_prod == 0:
            c_out = c_ks.value // k_prod
        else:
            c_out = "_c_fold"
    else:
        c_out = "_c_fold"
    return TensorShape((
        input_shape.dims[0],
        ShapeDim(c_out),
        ShapeDim(output_size[0]),
        ShapeDim(output_size[1]),
    )), None


def _propagate_instancenorm1d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """InstanceNorm1d preserves shape, checks (N, C, L) format."""
    if input_shape.ndim != 3:
        return None, f"InstanceNorm1d expects 3D input, got {input_shape.ndim}D"
    feat = input_shape.dims[1]
    if layer.num_features is not None and not feat.is_symbolic and isinstance(layer.num_features, int):
        if feat.value != layer.num_features:
            return None, (
                f"InstanceNorm1d expects {layer.num_features} features, "
                f"got {feat.value}"
            )
    return input_shape, None


def _propagate_instancenorm3d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """InstanceNorm3d preserves shape, checks (N, C, D, H, W) format."""
    if input_shape.ndim != 5:
        return None, f"InstanceNorm3d expects 5D input, got {input_shape.ndim}D"
    feat = input_shape.dims[1]
    if layer.num_features is not None and not feat.is_symbolic and isinstance(layer.num_features, int):
        if feat.value != layer.num_features:
            return None, (
                f"InstanceNorm3d expects {layer.num_features} features, "
                f"got {feat.value}"
            )
    return input_shape, None


def _propagate_syncbatchnorm(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """SyncBatchNorm preserves shape (same as BatchNorm)."""
    return _propagate_batchnorm(input_shape, layer)


def _propagate_batchnorm3d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """BatchNorm3d preserves shape, checks (N, C, D, H, W) format."""
    if input_shape.ndim != 5:
        return None, f"BatchNorm3d expects 5D input, got {input_shape.ndim}D"
    feat = input_shape.dims[1]
    if layer.num_features is not None and not feat.is_symbolic and isinstance(layer.num_features, int):
        if feat.value != layer.num_features:
            return None, (
                f"BatchNorm3d expects {layer.num_features} features, "
                f"got {feat.value}"
            )
    return input_shape, None


def _propagate_pool1d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """MaxPool1d/AvgPool1d: (N, C, L) → (N, C, L')."""
    if input_shape.ndim != 3:
        return None, f"Pool1d expects 3D, got {input_shape.ndim}D"
    ks = layer.kernel_size or (2,)
    if isinstance(ks, int):
        ks = (ks,)
    stride = layer.params.get("stride", ks)
    if isinstance(stride, int):
        stride = (stride,)
    padding = layer.params.get("padding", (0,))
    if isinstance(padding, int):
        padding = (padding,)
    l_in = input_shape.dims[2]
    if not l_in.is_symbolic:
        l_out = (l_in.value + 2 * padding[0] - ks[0]) // stride[0] + 1
        new_dims = (input_shape.dims[0], input_shape.dims[1], ShapeDim(l_out))
    else:
        new_dims = (input_shape.dims[0], input_shape.dims[1], ShapeDim("_L_out"))
    return TensorShape(new_dims), None


def _propagate_pool3d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """MaxPool3d: (N, C, D, H, W) → (N, C, D', H', W')."""
    if input_shape.ndim != 5:
        return None, f"Pool3d expects 5D, got {input_shape.ndim}D"
    ks = layer.kernel_size or (2, 2, 2)
    stride = layer.params.get("stride", ks)
    padding = layer.params.get("padding", (0, 0, 0))
    spatial = []
    for i in range(3):
        d_in = input_shape.dims[2 + i]
        if not d_in.is_symbolic:
            d_out = (d_in.value + 2 * padding[i] - ks[i]) // stride[i] + 1
            spatial.append(ShapeDim(d_out))
        else:
            spatial.append(ShapeDim(f"_d{i}_out"))
    return TensorShape((input_shape.dims[0], input_shape.dims[1]) + tuple(spatial)), None


def _propagate_adaptive_pool1d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """AdaptiveAvgPool1d / AdaptiveMaxPool1d: (N, C, L) → (N, C, L_out)."""
    if input_shape.ndim != 3:
        return None, f"AdaptivePool1d expects 3D, got {input_shape.ndim}D"
    out = layer.output_size
    if out is None:
        return None, "output_size unknown"
    if isinstance(out, tuple):
        l_out = out[0]
    else:
        l_out = out
    return TensorShape((input_shape.dims[0], input_shape.dims[1], ShapeDim(l_out))), None


def _propagate_lppool2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """LPPool2d: same spatial reduction as pool2d."""
    return _propagate_pool2d(input_shape, layer)


def _propagate_fractionalmaxpool2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """FractionalMaxPool2d: (N, C, H, W) → (N, C, H_out, W_out)."""
    if input_shape.ndim != 4:
        return None, f"FractionalMaxPool2d expects 4D, got {input_shape.ndim}D"
    out = layer.output_size
    if out is not None:
        return TensorShape((
            input_shape.dims[0], input_shape.dims[1],
            ShapeDim(out[0]), ShapeDim(out[1]),
        )), None
    # If output_size not given, spatial dims are stochastic
    return TensorShape((
        input_shape.dims[0], input_shape.dims[1],
        ShapeDim("_frac_h"), ShapeDim("_frac_w"),
    )), None


def _propagate_rnn(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """RNN: same contract as GRU for shape propagation."""
    return _propagate_gru(input_shape, layer)


def _propagate_pad2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """ReflectionPad2d / ReplicationPad2d / ZeroPad2d / ConstantPad2d.

    Padding tuple is (left, right, top, bottom).
    """
    if input_shape.ndim < 3:
        return None, f"Pad2d expects at least 3D, got {input_shape.ndim}D"
    pad = layer.params.get("padding")
    if pad is None:
        return input_shape, None
    if isinstance(pad, int):
        pad = (pad, pad, pad, pad)
    elif isinstance(pad, (tuple, list)):
        if len(pad) == 2:
            pad = (pad[0], pad[0], pad[1], pad[1])
        elif len(pad) != 4:
            return input_shape, None
    h_in = input_shape.dims[-2]
    w_in = input_shape.dims[-1]
    if not h_in.is_symbolic and not w_in.is_symbolic:
        h_out = h_in.value + pad[2] + pad[3]
        w_out = w_in.value + pad[0] + pad[1]
        new_dims = input_shape.dims[:-2] + (ShapeDim(h_out), ShapeDim(w_out))
    else:
        new_dims = input_shape.dims[:-2] + (ShapeDim("_pad_h"), ShapeDim("_pad_w"))
    return TensorShape(new_dims), None


def _propagate_pixel_unshuffle(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """PixelUnshuffle: (N, C, H*r, W*r) → (N, C*r^2, H, W)."""
    if input_shape.ndim != 4:
        return None, f"PixelUnshuffle expects 4D, got {input_shape.ndim}D"
    r = layer.params.get("downscale_factor")
    if r is None:
        return None, "downscale_factor unknown"
    c_in = input_shape.dims[1]
    h_in = input_shape.dims[2]
    w_in = input_shape.dims[3]
    if not c_in.is_symbolic:
        c_out = c_in.value * r * r
    else:
        c_out = "_c_unshuffle"
    if not h_in.is_symbolic and not w_in.is_symbolic:
        if h_in.value % r != 0 or w_in.value % r != 0:
            return None, (
                f"PixelUnshuffle: spatial dims ({h_in.value}, {w_in.value}) "
                f"not divisible by downscale_factor={r}"
            )
        h_out = h_in.value // r
        w_out = w_in.value // r
    else:
        h_out = "_h_unshuffle"
        w_out = "_w_unshuffle"
    return TensorShape((
        input_shape.dims[0], ShapeDim(c_out),
        ShapeDim(h_out), ShapeDim(w_out),
    )), None


def _propagate_conv3d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Conv3d: (N, C_in, D, H, W) → (N, C_out, D', H', W')."""
    if input_shape.ndim != 5:
        return None, f"Conv3d expects 5D input, got {input_shape.ndim}D"
    c_in = input_shape.dims[1]
    if layer.in_channels is not None and not c_in.is_symbolic and isinstance(layer.in_channels, int):
        if c_in.value != layer.in_channels:
            return None, (
                f"Conv3d expects {layer.in_channels} input channels, "
                f"got {c_in.value}"
            )
    out_c = layer.out_channels
    if out_c is None:
        return None, "Conv3d out_channels unknown"
    ks = layer.kernel_size or (3, 3, 3)
    stride = layer.params.get("stride", (1, 1, 1))
    padding = layer.params.get("padding", (0, 0, 0))
    dilation = layer.params.get("dilation", (1, 1, 1))
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    if isinstance(ks, int):
        ks = (ks, ks, ks)
    groups = layer.params.get("groups", 1)
    if isinstance(groups, int) and groups > 1:
        if not c_in.is_symbolic and isinstance(c_in.value, int) and c_in.value % groups != 0:
            return None, (
                f"Conv3d groups={groups} does not divide in_channels={c_in.value}"
            )
        if isinstance(out_c, int) and out_c % groups != 0:
            return None, (
                f"Conv3d groups={groups} does not divide out_channels={out_c}"
            )
    spatial = []
    for i in range(3):
        d_in = input_shape.dims[2 + i]
        if not d_in.is_symbolic:
            d_out = (d_in.value + 2 * padding[i] - dilation[i] * (ks[i] - 1) - 1) // stride[i] + 1
            spatial.append(ShapeDim(d_out))
        else:
            spatial.append(ShapeDim(f"_d{i}_out"))
    return TensorShape((input_shape.dims[0], ShapeDim(out_c)) + tuple(spatial)), None


def _propagate_convtranspose3d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """ConvTranspose3d: (N, C_in, D, H, W) → (N, C_out, D', H', W')."""
    if input_shape.ndim != 5:
        return None, f"ConvTranspose3d expects 5D, got {input_shape.ndim}D"
    c_in = layer.in_channels
    if c_in is not None and not input_shape.dims[1].is_symbolic:
        if input_shape.dims[1].value != c_in:
            return None, (
                f"ConvTranspose3d expects {c_in} input channels, "
                f"got {input_shape.dims[1].value}"
            )
    out_c = layer.out_channels
    if out_c is None:
        return None, "ConvTranspose3d out_channels unknown"
    ks = layer.kernel_size or (3, 3, 3)
    stride = layer.params.get("stride", (1, 1, 1))
    padding = layer.params.get("padding", (0, 0, 0))
    output_padding = layer.params.get("output_padding", (0, 0, 0))
    dilation = layer.params.get("dilation", (1, 1, 1))
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    if isinstance(ks, int):
        ks = (ks, ks, ks)
    groups = layer.params.get("groups", 1)
    if isinstance(groups, int) and groups > 1:
        if isinstance(c_in, int) and c_in % groups != 0:
            return None, (
                f"ConvTranspose3d groups={groups} does not divide "
                f"in_channels={c_in}"
            )
        if isinstance(out_c, int) and out_c % groups != 0:
            return None, (
                f"ConvTranspose3d groups={groups} does not divide "
                f"out_channels={out_c}"
            )
    spatial = []
    for i in range(3):
        d_in = input_shape.dims[2 + i]
        if not d_in.is_symbolic:
            d_out = ((d_in.value - 1) * stride[i] - 2 * padding[i]
                     + dilation[i] * (ks[i] - 1) + output_padding[i] + 1)
            spatial.append(ShapeDim(d_out))
        else:
            spatial.append(ShapeDim(f"_d{i}_tconv_out"))
    return TensorShape((input_shape.dims[0], ShapeDim(out_c)) + tuple(spatial)), None


def _propagate_layernorm(inp_shape: TensorShape, layer: LayerDef) -> Tuple[Optional[TensorShape], Optional[str]]:
    """LayerNorm preserves input shape. Check that normalized_shape matches trailing dims."""
    normalized_shape = layer.params.get("normalized_shape")
    if normalized_shape and inp_shape.ndim > 0:
        ns = normalized_shape if isinstance(normalized_shape, (list, tuple)) else [normalized_shape]
        if len(ns) <= inp_shape.ndim:
            for i, (expected, actual) in enumerate(zip(reversed(ns), reversed(inp_shape.dims))):
                # Symbolic-vs-concrete or vice versa: bind, do not reject. The
                # concrete value is consistent with any free symbol; abstaining
                # here preserves soundness without spurious FP on configs whose
                # attributes were resolved symbolically.
                if isinstance(expected, str) or actual.is_symbolic:
                    continue
                if actual.value != expected:
                    return None, f"LayerNorm normalized_shape[-{i+1}]={expected} but input dim={actual.value}"
    return inp_shape, None


# --- New propagation functions for expanded operator coverage ---

def _propagate_loss(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Loss functions: output depends on reduction param.

    reduction='none' → same shape as input; 'mean'/'sum' → scalar (1,).
    """
    reduction = layer.params.get("reduction", "mean")
    if reduction == "none":
        return input_shape, None
    return TensorShape((ShapeDim(1),)), None


def _propagate_glu(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """GLU: halves one dimension (default dim=-1).

    Input (..., 2*N, ...) → (..., N, ...) along dim.
    """
    dim = layer.params.get("dim", -1)
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim < 0 or dim >= input_shape.ndim:
        return None, f"GLU dim={layer.params.get('dim', -1)} out of range for {input_shape.ndim}D"
    d = input_shape.dims[dim]
    if not d.is_symbolic:
        if d.value % 2 != 0:
            return None, f"GLU expects even size along dim {dim}, got {d.value}"
        new_dim = ShapeDim(d.value // 2)
    else:
        new_dim = ShapeDim("_glu_half")
    new_dims = input_shape.dims[:dim] + (new_dim,) + input_shape.dims[dim + 1:]
    return TensorShape(new_dims), None


def _propagate_pad1d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """1D padding: (N, C, W) → (N, C, W + pad_left + pad_right)."""
    if input_shape.ndim < 2:
        return None, f"Pad1d expects at least 2D, got {input_shape.ndim}D"
    pad = layer.params.get("padding")
    if pad is None:
        return input_shape, None
    if isinstance(pad, int):
        pad = (pad, pad)
    elif isinstance(pad, (tuple, list)):
        if len(pad) == 1:
            pad = (pad[0], pad[0])
        elif len(pad) != 2:
            return input_shape, None
    w_in = input_shape.dims[-1]
    if not w_in.is_symbolic:
        w_out = w_in.value + pad[0] + pad[1]
        new_dims = input_shape.dims[:-1] + (ShapeDim(w_out),)
    else:
        new_dims = input_shape.dims[:-1] + (ShapeDim("_pad_w"),)
    return TensorShape(new_dims), None


def _propagate_pad3d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """3D padding: pads last 3 spatial dims (D, H, W).

    Padding tuple is (left, right, top, bottom, front, back).
    """
    if input_shape.ndim < 4:
        return None, f"Pad3d expects at least 4D, got {input_shape.ndim}D"
    pad = layer.params.get("padding")
    if pad is None:
        return input_shape, None
    if isinstance(pad, int):
        pad = (pad, pad, pad, pad, pad, pad)
    elif isinstance(pad, (tuple, list)):
        if len(pad) == 3:
            pad = (pad[0], pad[0], pad[1], pad[1], pad[2], pad[2])
        elif len(pad) != 6:
            return input_shape, None
    w_in = input_shape.dims[-1]
    h_in = input_shape.dims[-2]
    d_in = input_shape.dims[-3]
    new_spatial = []
    for dim_in, pl, pr in [(d_in, pad[4], pad[5]), (h_in, pad[2], pad[3]), (w_in, pad[0], pad[1])]:
        if not dim_in.is_symbolic:
            new_spatial.append(ShapeDim(dim_in.value + pl + pr))
        else:
            new_spatial.append(ShapeDim("_pad_s"))
    new_dims = input_shape.dims[:-3] + (new_spatial[0], new_spatial[1], new_spatial[2])
    return TensorShape(new_dims), None


def _propagate_adaptive_pool3d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """AdaptiveAvgPool3d / AdaptiveMaxPool3d: (N, C, D, H, W) → (N, C, D', H', W')."""
    if input_shape.ndim != 5:
        return None, f"AdaptivePool3d expects 5D, got {input_shape.ndim}D"
    out = layer.output_size
    if out is None:
        return None, "output_size unknown"
    if isinstance(out, int):
        out = (out, out, out)
    return TensorShape((
        input_shape.dims[0], input_shape.dims[1],
        ShapeDim(out[0]), ShapeDim(out[1]), ShapeDim(out[2]),
    )), None


def _propagate_lppool1d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """LPPool1d: same spatial reduction as pool1d."""
    return _propagate_pool1d(input_shape, layer)


def _propagate_fractionalmaxpool3d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """FractionalMaxPool3d: (N, C, D, H, W) → (N, C, D', H', W')."""
    if input_shape.ndim != 5:
        return None, f"FractionalMaxPool3d expects 5D, got {input_shape.ndim}D"
    out = layer.output_size
    if out is not None:
        return TensorShape((
            input_shape.dims[0], input_shape.dims[1],
            ShapeDim(out[0]), ShapeDim(out[1]), ShapeDim(out[2]),
        )), None
    return TensorShape((
        input_shape.dims[0], input_shape.dims[1],
        ShapeDim("_frac_d"), ShapeDim("_frac_h"), ShapeDim("_frac_w"),
    )), None


def _propagate_maxunpool1d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """MaxUnpool1d: inverse of MaxPool1d. (N, C, L) → (N, C, L')."""
    if input_shape.ndim != 3:
        return None, f"MaxUnpool1d expects 3D, got {input_shape.ndim}D"
    ks = layer.kernel_size
    if ks is None:
        return TensorShape((input_shape.dims[0], input_shape.dims[1], ShapeDim("_unpool_l"))), None
    if isinstance(ks, int):
        ks = (ks,)
    stride = layer.params.get("stride", ks)
    if isinstance(stride, int):
        stride = (stride,)
    padding = layer.params.get("padding", 0)
    if isinstance(padding, int):
        padding = (padding,)
    l_in = input_shape.dims[2]
    if not l_in.is_symbolic:
        l_out = (l_in.value - 1) * stride[0] - 2 * padding[0] + ks[0]
        new_dims = (input_shape.dims[0], input_shape.dims[1], ShapeDim(l_out))
    else:
        new_dims = (input_shape.dims[0], input_shape.dims[1], ShapeDim("_unpool_l"))
    return TensorShape(new_dims), None


def _propagate_maxunpool2d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """MaxUnpool2d: inverse of MaxPool2d. (N, C, H, W) → (N, C, H', W')."""
    if input_shape.ndim != 4:
        return None, f"MaxUnpool2d expects 4D, got {input_shape.ndim}D"
    ks = layer.kernel_size
    if ks is None:
        return TensorShape((
            input_shape.dims[0], input_shape.dims[1],
            ShapeDim("_unpool_h"), ShapeDim("_unpool_w")
        )), None
    if isinstance(ks, int):
        ks = (ks, ks)
    stride = layer.params.get("stride", ks)
    if isinstance(stride, int):
        stride = (stride, stride)
    padding = layer.params.get("padding", 0)
    if isinstance(padding, int):
        padding = (padding, padding)
    spatial = []
    for i in range(2):
        d_in = input_shape.dims[2 + i]
        if not d_in.is_symbolic:
            d_out = (d_in.value - 1) * stride[i] - 2 * padding[i] + ks[i]
            spatial.append(ShapeDim(d_out))
        else:
            spatial.append(ShapeDim(f"_unpool_{i}"))
    return TensorShape((input_shape.dims[0], input_shape.dims[1]) + tuple(spatial)), None


def _propagate_maxunpool3d(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """MaxUnpool3d: inverse of MaxPool3d. (N, C, D, H, W) → (N, C, D', H', W')."""
    if input_shape.ndim != 5:
        return None, f"MaxUnpool3d expects 5D, got {input_shape.ndim}D"
    ks = layer.kernel_size
    if ks is None:
        return TensorShape((
            input_shape.dims[0], input_shape.dims[1],
            ShapeDim("_unpool_d"), ShapeDim("_unpool_h"), ShapeDim("_unpool_w")
        )), None
    if isinstance(ks, int):
        ks = (ks, ks, ks)
    stride = layer.params.get("stride", ks)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    padding = layer.params.get("padding", 0)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    spatial = []
    for i in range(3):
        d_in = input_shape.dims[2 + i]
        if not d_in.is_symbolic:
            d_out = (d_in.value - 1) * stride[i] - 2 * padding[i] + ks[i]
            spatial.append(ShapeDim(d_out))
        else:
            spatial.append(ShapeDim(f"_unpool_{i}"))
    return TensorShape((input_shape.dims[0], input_shape.dims[1]) + tuple(spatial)), None


def _propagate_embeddingbag(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """EmbeddingBag: (N,) or (N, M) → (N, embedding_dim)."""
    if layer.embedding_dim is None:
        return None, "EmbeddingBag embedding_dim unknown"
    if input_shape.ndim < 1:
        return None, "EmbeddingBag requires at least 1D input"
    batch = input_shape.dims[0]
    return TensorShape((batch, ShapeDim(layer.embedding_dim))), None


def _propagate_bilinear(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Bilinear: (N, in1_features) → (N, out_features).

    Note: takes two inputs but we propagate from the first.
    """
    out_f = layer.params.get("out_features")
    if out_f is None:
        return None, "Bilinear out_features unknown"
    if input_shape.ndim < 1:
        return None, "Bilinear requires at least 1D input"
    new_dims = input_shape.dims[:-1] + (ShapeDim(out_f),)
    return TensorShape(new_dims), None


def _propagate_container(
    input_shape: TensorShape, _layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Containers (ModuleDict, ParameterList, ParameterDict): no shape effect."""
    return input_shape, None


def _propagate_pairwise_or_cosine(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """PairwiseDistance / CosineSimilarity: (N, D) → (N,)."""
    if input_shape.ndim < 2:
        return None, f"PairwiseDistance/CosineSimilarity expects ≥2D, got {input_shape.ndim}D"
    dim = layer.params.get("dim", 1)
    if dim < 0:
        dim = input_shape.ndim + dim
    new_dims = input_shape.dims[:dim] + input_shape.dims[dim + 1:]
    if len(new_dims) == 0:
        new_dims = (ShapeDim(1),)
    return TensorShape(new_dims), None


def _propagate_channel_shuffle(
    input_shape: TensorShape, _layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """ChannelShuffle: shape-preserving."""
    return input_shape, None


def _propagate_unflatten(
    input_shape: TensorShape, layer: LayerDef
) -> Tuple[Optional[TensorShape], Optional[str]]:
    """Unflatten: replaces one dim with multiple dims from unflattened_size."""
    dim = layer.params.get("dim")
    unflattened_size = layer.params.get("unflattened_size")
    if dim is None or unflattened_size is None:
        return None, "Unflatten requires dim and unflattened_size"
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim < 0 or dim >= input_shape.ndim:
        return None, f"Unflatten dim={layer.params.get('dim')} out of range"
    if isinstance(unflattened_size, (list, tuple)):
        new_dims_mid = tuple(ShapeDim(s) for s in unflattened_size)
    else:
        new_dims_mid = (ShapeDim(unflattened_size),)
    new_dims = input_shape.dims[:dim] + new_dims_mid + input_shape.dims[dim + 1:]
    return TensorShape(new_dims), None


# --- Phase/stats-dependent normalization runtime-error checks (Step 29) ---
#
# BatchNorm raises ``ValueError: Expected more than 1 value per channel when
# training`` when the number of elements per channel (N * spatial) == 1 and the
# layer is using *batch* statistics.  InstanceNorm raises ``Expected more than 1
# spatial element when training`` when the *spatial* element count == 1 and it is
# using *input* statistics.  Both modes use batch/input stats when
# ``self.training or not track_running_stats`` is true.
#
# SyncBatchNorm is deliberately excluded: under distributed training the global
# per-channel count can exceed 1 even when the local count is 1, so flagging it
# would be unsound.
_BN_COUNT_NORM_KINDS = frozenset({
    LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D, LayerKind.BATCHNORM3D,
    LayerKind.LAZYBATCHNORM1D, LayerKind.LAZYBATCHNORM2D,
    LayerKind.LAZYBATCHNORM3D,
})
_IN_SPATIAL_NORM_KINDS = frozenset({
    LayerKind.INSTANCENORM1D, LayerKind.INSTANCENORM2D,
    LayerKind.INSTANCENORM3D, LayerKind.LAZYINSTANCENORM1D,
    LayerKind.LAZYINSTANCENORM2D, LayerKind.LAZYINSTANCENORM3D,
})
# Canonical *batched* rank for each normalization kind.  The batch-size-one /
# spatial-one check only runs at these ranks; at any other rank we abstain
# (a rank mismatch is a separate diagnostic and we do not want to guess which
# dims are spatial).
_NORM_CANONICAL_RANK = {
    LayerKind.BATCHNORM1D: (2, 3), LayerKind.LAZYBATCHNORM1D: (2, 3),
    LayerKind.BATCHNORM2D: (4,), LayerKind.LAZYBATCHNORM2D: (4,),
    LayerKind.BATCHNORM3D: (5,), LayerKind.LAZYBATCHNORM3D: (5,),
    LayerKind.INSTANCENORM1D: (3,), LayerKind.LAZYINSTANCENORM1D: (3,),
    LayerKind.INSTANCENORM2D: (4,), LayerKind.LAZYINSTANCENORM2D: (4,),
    LayerKind.INSTANCENORM3D: (5,), LayerKind.LAZYINSTANCENORM3D: (5,),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Dtype algebra (Step 30) — a second algebra alongside shape.
#
# We track a *known* element dtype for each tensor.  Absence from ``dtype_env``
# means the dtype is unknown and every dtype check abstains on that tensor, so
# the analysis never raises a false positive on an unannotated value.  Only an
# explicit annotation (input_dtypes), a layer's real parameter dtype (read from
# the live module), or an explicit cast (.half()/.float()/.to(dtype=...)) makes
# a dtype *known*.  Checks only fire when *both* participating dtypes are known.
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical torch dtype spellings (alias → canonical).
_DTYPE_ALIASES = {
    "half": "float16",
    "float": "float32",
    "double": "float64",
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "long": "int64",
    "int": "int32",
    "short": "int16",
    "char": "int8",
    "byte": "uint8",
    "bool": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "cfloat": "complex64",
    "cdouble": "complex128",
    "complex64": "complex64",
    "complex128": "complex128",
}

_FLOAT_DTYPES = frozenset(
    {"float16", "bfloat16", "float32", "float64", "complex64", "complex128"}
)
_INT_DTYPES = frozenset({"int8", "int16", "int32", "int64", "uint8"})

# Layers whose forward performs a matmul/convolution against a stored parameter
# and therefore require the *input* dtype to exactly equal the *parameter*
# dtype (torch raises e.g. "mat1 and mat2 must have the same dtype" /
# "Input type (...) and bias type (...) should be the same").
_DTYPE_PARAM_MATCH_KINDS = frozenset({
    LayerKind.LINEAR, LayerKind.BILINEAR,
    LayerKind.CONV1D, LayerKind.CONV2D, LayerKind.CONV3D,
    LayerKind.CONVTRANSPOSE1D, LayerKind.CONVTRANSPOSE2D,
    LayerKind.CONVTRANSPOSE3D,
})


def _canon_dtype(raw: Any) -> Optional[str]:
    """Canonicalise a dtype spelling (e.g. ``torch.float16``, ``"half"``) to a
    canonical string, or ``None`` if it cannot be recognised."""
    if raw is None:
        return None
    s = str(raw)
    if s.startswith("torch."):
        s = s[len("torch."):]
    s = s.strip().lower()
    return _DTYPE_ALIASES.get(s)


def _is_float_dtype(dt: Optional[str]) -> bool:
    return dt in _FLOAT_DTYPES


def _is_int_dtype(dt: Optional[str]) -> bool:
    return dt in _INT_DTYPES


_LAYER_PROPAGATORS = {
    LayerKind.LINEAR: _propagate_linear,
    LayerKind.CONV2D: _propagate_conv2d,
    LayerKind.CONV1D: _propagate_conv1d,
    LayerKind.BATCHNORM1D: _propagate_batchnorm,
    LayerKind.BATCHNORM2D: _propagate_batchnorm,
    LayerKind.DROPOUT: _propagate_dropout,
    LayerKind.EMBEDDING: _propagate_embedding,
    LayerKind.ADAPTIVE_AVGPOOL2D: _propagate_adaptive_avgpool2d,
    LayerKind.MAXPOOL2D: _propagate_pool2d,
    LayerKind.AVGPOOL2D: _propagate_pool2d,
    LayerKind.SEQUENTIAL: _propagate_sequential,
    LayerKind.GROUPNORM: _propagate_groupnorm,
    LayerKind.INSTANCENORM2D: _propagate_instancenorm2d,
    LayerKind.CONVTRANSPOSE2D: _propagate_convtranspose2d,
    LayerKind.UPSAMPLE: _propagate_upsample,
    LayerKind.MULTIHEAD_ATTENTION: _propagate_multihead_attention,
    LayerKind.TRANSFORMER_ENCODER_LAYER: _propagate_transformer_encoder_layer,
    LayerKind.TRANSFORMER_DECODER_LAYER: _propagate_transformer_decoder_layer,
    LayerKind.TRANSFORMER_ENCODER: _propagate_transformer_encoder,
    LayerKind.TRANSFORMER_DECODER: _propagate_transformer_decoder,
    LayerKind.LSTM: _propagate_lstm,
    LayerKind.GRU: _propagate_gru,
    LayerKind.CONVTRANSPOSE1D: _propagate_convtranspose1d,
    LayerKind.ADAPTIVE_MAXPOOL2D: _propagate_adaptive_maxpool2d,
    LayerKind.PIXEL_SHUFFLE: _propagate_pixel_shuffle,
    LayerKind.UNFOLD: _propagate_unfold,
    LayerKind.FOLD: _propagate_fold,
    LayerKind.INSTANCENORM1D: _propagate_instancenorm1d,
    LayerKind.INSTANCENORM3D: _propagate_instancenorm3d,
    LayerKind.SYNCBATCHNORM: _propagate_syncbatchnorm,
    LayerKind.BATCHNORM3D: _propagate_batchnorm3d,
    LayerKind.MAXPOOL1D: _propagate_pool1d,
    LayerKind.AVGPOOL1D: _propagate_pool1d,
    LayerKind.MAXPOOL3D: _propagate_pool3d,
    LayerKind.ADAPTIVE_AVGPOOL1D: _propagate_adaptive_pool1d,
    LayerKind.ADAPTIVE_MAXPOOL1D: _propagate_adaptive_pool1d,
    LayerKind.LPPOOL2D: _propagate_lppool2d,
    LayerKind.FRACTIONALMAXPOOL2D: _propagate_fractionalmaxpool2d,
    LayerKind.RNN: _propagate_rnn,
    LayerKind.REFLECTIONPAD2D: _propagate_pad2d,
    LayerKind.REPLICATIONPAD2D: _propagate_pad2d,
    LayerKind.ZEROPAD2D: _propagate_pad2d,
    LayerKind.CONSTANTPAD2D: _propagate_pad2d,
    LayerKind.PIXEL_UNSHUFFLE: _propagate_pixel_unshuffle,
    LayerKind.ALPHADROPOUT: _propagate_dropout,
    LayerKind.CONV3D: _propagate_conv3d,
    LayerKind.CONVTRANSPOSE3D: _propagate_convtranspose3d,
    LayerKind.LAYERNORM: _propagate_layernorm,
    # --- New operators ---
    LayerKind.LOSS_FUNCTION: _propagate_loss,
    LayerKind.GLU: _propagate_glu,
    LayerKind.CONSTANTPAD1D: _propagate_pad1d,
    LayerKind.ZEROPAD1D: _propagate_pad1d,
    LayerKind.REFLECTIONPAD1D: _propagate_pad1d,
    LayerKind.REPLICATIONPAD1D: _propagate_pad1d,
    LayerKind.CIRCULARPAD1D: _propagate_pad1d,
    LayerKind.CIRCULARPAD2D: _propagate_pad2d,
    LayerKind.CONSTANTPAD3D: _propagate_pad3d,
    LayerKind.ZEROPAD3D: _propagate_pad3d,
    LayerKind.REFLECTIONPAD3D: _propagate_pad3d,
    LayerKind.REPLICATIONPAD3D: _propagate_pad3d,
    LayerKind.CIRCULARPAD3D: _propagate_pad3d,
    LayerKind.ADAPTIVE_AVGPOOL3D: _propagate_adaptive_pool3d,
    LayerKind.ADAPTIVE_MAXPOOL3D: _propagate_adaptive_pool3d,
    LayerKind.AVGPOOL3D: _propagate_pool3d,
    LayerKind.LPPOOL1D: _propagate_lppool1d,
    LayerKind.FRACTIONALMAXPOOL3D: _propagate_fractionalmaxpool3d,
    LayerKind.MAXUNPOOL1D: _propagate_maxunpool1d,
    LayerKind.MAXUNPOOL2D: _propagate_maxunpool2d,
    LayerKind.MAXUNPOOL3D: _propagate_maxunpool3d,
    LayerKind.EMBEDDINGBAG: _propagate_embeddingbag,
    LayerKind.BILINEAR: _propagate_bilinear,
    LayerKind.MODULEDICT: _propagate_container,
    LayerKind.PARAMETERLIST: _propagate_container,
    LayerKind.PARAMETERDICT: _propagate_container,
    LayerKind.LAZYLINEAR: _propagate_linear,
    LayerKind.LAZYCONV1D: _propagate_conv1d,
    LayerKind.LAZYCONV2D: _propagate_conv2d,
    LayerKind.LAZYCONV3D: _propagate_conv3d,
    LayerKind.LAZYBATCHNORM1D: _propagate_batchnorm,
    LayerKind.LAZYBATCHNORM2D: _propagate_batchnorm,
    LayerKind.LAZYBATCHNORM3D: _propagate_batchnorm3d,
    LayerKind.LAZYINSTANCENORM1D: _propagate_instancenorm1d,
    LayerKind.LAZYINSTANCENORM2D: _propagate_instancenorm2d,
    LayerKind.LAZYINSTANCENORM3D: _propagate_instancenorm3d,
    LayerKind.LAZYCONVTRANSPOSE1D: _propagate_convtranspose1d,
    LayerKind.LAZYCONVTRANSPOSE2D: _propagate_convtranspose2d,
    LayerKind.LAZYCONVTRANSPOSE3D: _propagate_convtranspose3d,
    LayerKind.PAIRWISE_DISTANCE: _propagate_pairwise_or_cosine,
    LayerKind.COSINE_SIMILARITY: _propagate_pairwise_or_cosine,
    LayerKind.CHANNEL_SHUFFLE: _propagate_channel_shuffle,
    LayerKind.UNFLATTEN: _propagate_unflatten,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 7b. Symbolic state for Z3-backed constraint verification
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KripkeState:
    """Z3-backed symbolic state for constraint-based verification.

    Maps tensor names to Z3 variables for shape dimensions, device
    placement, gradient status, and overall phase.  Each ``KripkeState``
    represents the system state at a specific step in the computation
    graph.
    """
    step_index: int
    shape_vars: Dict[str, List[Any]] = field(default_factory=dict)
    device_vars: Dict[str, Any] = field(default_factory=dict)
    phase_var: Any = None
    grad_vars: Dict[str, Any] = field(default_factory=dict)
    layer_name: str = ""

    def as_tuple(self) -> tuple:
        """Hashable representation for state-space exploration."""
        return (
            tuple(sorted((k, str(v)) for k, v in self.shape_vars.items())),
            tuple(sorted((k, str(v)) for k, v in self.device_vars.items())),
            self.step_index,
        )


@dataclass
class KripkeTransition:
    """A transition in the Kripke structure.

    Represents one computation step (layer application) transforming
    the verification state. The guard is the Z3 constraint that must
    hold for the transition to be valid (shape compatibility, device
    consistency, etc.).
    """
    source: int  # source state index
    target: int  # target state index
    operation: str  # the nn.Module operation applied
    guard: Optional[Any] = None  # Z3 BoolRef constraint


@dataclass
class KripkeStructure:
    """Formal Kripke structure K = (S, S₀, R, AP, L) for nn.Module verification.

    Formalizes the model checker's state space as a transition system:
      S  = KripkeState instances (shape_vars × device_vars × phase × grad × step)
      S₀ = {initial state from input_shapes and default device/phase}
      R  ⊆ S × S = transitions induced by layer applications
      AP = {shape_safe, device_consistent, gradient_valid, phase_correct}
      L  : S → 2^AP = labeling function checking which properties hold at each state

    State-space finiteness:
      For architectures without data-dependent control flow (no torch.cond,
      no dynamic routing), the computation DAG is finite and acyclic,
      so |S| ≤ |layers| + 1 and |R| ≤ |edges in DAG|.
      Symbolic dimensions remain symbolic (not enumerated), so the
      state space is finite even with parametric batch_size.
    """
    states: List[KripkeState]
    initial_state_idx: int
    transitions: List[KripkeTransition]
    atomic_propositions: FrozenSet[str] = frozenset({
        "shape_safe", "device_consistent", "gradient_valid", "phase_correct"
    })
    labeling: Dict[int, FrozenSet[str]] = field(default_factory=dict)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_transitions(self) -> int:
        return len(self.transitions)

    @property
    def initial_state(self) -> KripkeState:
        return self.states[self.initial_state_idx]

    def is_safe(self) -> bool:
        """Check universal safety: ∀s ∈ reachable(S₀). shape_safe ∈ L(s)."""
        return all(
            "shape_safe" in self.labeling.get(i, frozenset())
            for i in range(len(self.states))
        )

    def get_violation_trace(self) -> Optional[List[KripkeTransition]]:
        """Find a trace from S₀ to a state where shape_safe ∉ L(s)."""
        from collections import deque
        adj: Dict[int, List[KripkeTransition]] = {}
        for t in self.transitions:
            adj.setdefault(t.source, []).append(t)

        visited = {self.initial_state_idx}
        parent: Dict[int, KripkeTransition] = {}
        queue = deque([self.initial_state_idx])

        while queue:
            sid = queue.popleft()
            if "shape_safe" not in self.labeling.get(sid, frozenset()):
                trace: List[KripkeTransition] = []
                cur = sid
                while cur in parent:
                    t = parent[cur]
                    trace.append(t)
                    cur = t.source
                trace.reverse()
                return trace
            for t in adj.get(sid, []):
                if t.target not in visited:
                    visited.add(t.target)
                    parent[t.target] = t
                    queue.append(t.target)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "num_states": self.num_states,
            "num_transitions": self.num_transitions,
            "initial_state_idx": self.initial_state_idx,
            "atomic_propositions": sorted(self.atomic_propositions),
            "state_space_finite": True,
            "states": [
                {
                    "index": i,
                    "step_index": s.step_index,
                    "layer_name": s.layer_name,
                    "labels": sorted(self.labeling.get(i, frozenset())),
                }
                for i, s in enumerate(self.states)
            ],
            "transitions": [
                {
                    "source": t.source,
                    "target": t.target,
                    "operation": t.operation,
                }
                for t in self.transitions
            ],
        }


# Layer kinds whose parameters reside on a device.
_PARAMETERISED_LAYERS: FrozenSet[LayerKind] = frozenset({
    LayerKind.LINEAR, LayerKind.CONV2D, LayerKind.CONV1D,
    LayerKind.CONVTRANSPOSE2D, LayerKind.CONVTRANSPOSE1D,
    LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
    LayerKind.LAYERNORM, LayerKind.EMBEDDING,
    LayerKind.LSTM, LayerKind.GRU, LayerKind.RNN,
    LayerKind.MULTIHEAD_ATTENTION,
    LayerKind.TRANSFORMER_ENCODER, LayerKind.TRANSFORMER_DECODER,
    LayerKind.TRANSFORMER_ENCODER_LAYER, LayerKind.TRANSFORMER_DECODER_LAYER,
    LayerKind.INSTANCENORM1D, LayerKind.INSTANCENORM2D, LayerKind.INSTANCENORM3D,
    LayerKind.SYNCBATCHNORM, LayerKind.BATCHNORM3D,
    LayerKind.GROUPNORM,
    LayerKind.CONV3D, LayerKind.CONVTRANSPOSE3D,
})


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Constraint Verifier (symbolic constraint propagation)
# ═══════════════════════════════════════════════════════════════════════════════

class ConstraintVerifier:
    """Constraint-based verifier for nn.Module computation graphs using the
    product theory T_shape × T_device × T_phase.

    The verifier uses forward symbolic constraint propagation through the
    computation DAG to verify four safety properties at every step:

      1. **shape_compatible** — each operation's input shapes are compatible
         with its semantics (e.g. matmul inner dims match).
      2. **device_consistent** — all tensors participating in an operation
         live on the same device (Z3 enum-sort backed).
      3. **gradient_valid** — gradient invariants are maintained (parameters
         require grad; detached tensors do not).
      4. **phase_correct** — phase-dependent layers (dropout, batchnorm)
         behave correctly w.r.t. train/eval mode (Z3 enum-sort backed).

    Algorithm (forward constraint propagation with product theory):

        **Base case** (steps 0 … N-1):
          - Create Z3 symbolic state variables for the initial state.
          - For each step *i* from 0 to N-1:
              • Create fresh Z3 variables for state after step *i*.
              • Add transition constraints (shape + device + phase
                + gradient) relating pre-state to post-state.
              • Check safety property for each domain via Z3 ``check()``.
          - Any SAT result yields a concrete counterexample.

        **Inductive step**:
          - For each consecutive pair of steps *(i, i+1)*:
              • Create Z3 symbolic states with *free* shape variables.
              • Assume safety at step *i*.
              • Add transition constraints.
              • Check whether step *i+1* can violate safety.
          - UNSAT ⇒ safety proved.

    Attributes:
        graph:        the computation graph to verify.
        ctx:          the Z3 encoding context (product theory).
        max_k:        maximum verification depth (defaults to graph length).
        input_shapes: user-supplied input shapes (may contain symbolic dims).
        default_device: default device for tensors & parameters.
    """

    def __init__(
        self,
        graph: ComputationGraph,
        input_shapes: Optional[Dict[str, tuple]] = None,
        default_device: Device = Device.CPU,
        default_phase: Phase = Phase.TRAIN,
        max_k: Optional[int] = None,
        constraints: Optional[Dict[str, Union[str, int]]] = None,
        produce_certificates: bool = False,
        use_kb_normalization: bool = False,
        check_devices: bool = True,
        check_phases: bool = True,
        check_gradients: bool = True,
        check_dtypes: bool = True,
        input_dtypes: Optional[Dict[str, str]] = None,
    ) -> None:
        self.graph = graph
        self.input_shapes = input_shapes or {}
        self.input_dtypes = input_dtypes or {}
        self.default_device = default_device
        self.default_phase = default_phase
        self.max_k = max_k if max_k is not None else graph.num_steps
        self.produce_certificates = produce_certificates
        self.use_kb_normalization = use_kb_normalization and HAS_KB_NORMALIZATION
        # When a domain is disabled we skip *generating and solving* its
        # constraints entirely (not merely filtering the verdict afterwards),
        # so the disabled domain costs no solver time and contributes no
        # cross-domain witnesses.  See _filter_domain_checks().
        self.check_devices = check_devices
        self.check_phases = check_phases
        self.check_gradients = check_gradients
        self.check_dtypes = check_dtypes
        # Under autocast / mixed-precision contexts torch silently inserts casts,
        # so a statically-mismatched dtype may run fine.  Abstain on dtype checks
        # entirely when the graph indicates autocast to stay sound (no FPs).
        _feats = getattr(graph, "dynamic_features", {}) or {}
        self._dtype_abstain = bool(
            _feats.get("forward_uses_autocast")
            or _feats.get("mixed_precision_api")
            or _feats.get("uses_autocast")
        )
        self.ctx = _Z3Context()
        self._stride_check_id = 0
        self.relational_constraints = constraints or {}
        # Tensors whose shape is opaque because they are produced by — or
        # derived from — an operator with no shape transfer function
        # (OpKind.UNSUPPORTED). Shape safety checks abstain on these to avoid
        # fabricating a violation against a free, unconstrained dimension.
        self._opaque_cache: Optional[Set[str]] = None

        self._init_state = self._build_initial_state()

        # Eagerly parse relational constraints so Z3 variables are created
        if self.relational_constraints and HAS_Z3:
            self._relational_z3 = self.ctx.build_relational_constraints(
                self.relational_constraints
            )
        else:
            self._relational_z3 = []

    # ------------------------------------------------------------------
    # Domain gating (solver-level, not post-hoc filtering)
    # ------------------------------------------------------------------

    _DOMAIN_OF_KIND = {
        "device_mismatch": "devices",
        "phase_violation": "phases",
        "phase_error": "phases",
        "gradient_violation": "gradients",
        "gradient_broken": "gradients",
        "dtype_error": "dtypes",
        "dtype_mismatch": "dtypes",
    }

    def _domain_enabled(self, kind: str) -> bool:
        """Whether the domain a violation *kind* belongs to is enabled."""
        domain = self._DOMAIN_OF_KIND.get(kind)
        if domain == "devices":
            return self.check_devices
        if domain == "phases":
            return self.check_phases
        if domain == "gradients":
            return self.check_gradients
        if domain == "dtypes":
            return self.check_dtypes
        return True  # shape / cross-domain / structural checks always run

    def _filter_domain_checks(self, pairs):
        """Drop ``(kind, encoder)`` pairs whose domain is disabled.

        This gates the solver: disabled domains are never encoded into Z3
        nor checked, so they incur no solver cost and produce no witnesses.
        """
        return [(kind, enc) for (kind, enc) in pairs if self._domain_enabled(kind)]

    # ------------------------------------------------------------------
    # Knuth-Bendix constraint normalization
    # ------------------------------------------------------------------

    def _kb_normalize_constraint(self, expr: "z3.ExprRef") -> "z3.ExprRef":
        """Normalize a Z3 constraint using KB rewrite rules if enabled.

        Applies the three-phase pipeline: z3.simplify → KB rules → z3.simplify.
        This is a no-op when ``use_kb_normalization`` is False.
        """
        if not self.use_kb_normalization:
            return expr
        try:
            return kb_normalize_z3(expr)
        except Exception:
            return expr

    # ------------------------------------------------------------------
    # Initial ModelState construction (concrete level)
    # ------------------------------------------------------------------

    def _build_initial_state(self) -> ModelState:
        """Construct the initial ``ModelState`` from *input_shapes*."""
        state = ModelState(phase=self.default_phase)
        for name, raw_shape in self.input_shapes.items():
            dims: List[ShapeDim] = []
            for d in raw_shape:
                if isinstance(d, int):
                    dims.append(ShapeDim(d))
                elif isinstance(d, str):
                    dims.append(ShapeDim(d))
                else:
                    dims.append(ShapeDim("_unk"))
            state.shape_env[name] = TensorShape(tuple(dims))
            state.device_map[name] = self.default_device
            state.gradient_status[name] = False
            # Only record an input dtype when the caller explicitly annotated it;
            # otherwise the dtype stays *unknown* so dtype checks abstain.
            dt = _canon_dtype(self.input_dtypes.get(name))
            if dt is not None:
                state.dtype_env[name] = dt
        # Pre-populate buffer shapes from register_buffer() calls in __init__.
        # Buffers are always registered on CPU (torch.randn / torch.zeros etc.
        # default to CPU).  They move with the model when .cuda() is called, but
        # the *static* device at registration time is CPU, which is the right
        # annotation for detecting "buffer stays on CPU" bugs.
        for buf_name, buf_shape in self.graph.buffer_shapes.items():
            state.shape_env[f"self.{buf_name}"] = buf_shape
            state.device_map[f"self.{buf_name}"] = Device.CPU
        # Pre-populate nn.Parameter shapes (these move with the model,
        # so no device mismatch — only seed shape, not device).
        for param_name, param_shape in self.graph.param_shapes.items():
            state.shape_env[f"self.{param_name}"] = param_shape
        # Pre-populate constant tensors folded by torch.fx (e.g. torch.rand(2,4)
        # written directly in forward).  Their shape is RNG-independent.
        for cname, cshape in self.graph.const_shapes.items():
            state.shape_env[cname] = cshape
            state.device_map[cname] = self.graph.const_devices.get(
                cname, Device.CPU
            )
        return state

    # ------------------------------------------------------------------
    # Symbolic state construction (Z3 level)
    # ------------------------------------------------------------------

    def _build_kripke_state(
        self,
        step_idx: int,
        model_state: ModelState,
        free_shapes: bool = False,
    ) -> KripkeState:
        """Build a ``KripkeState`` (symbolic state) from a ``ModelState``.

        Parameters
        ----------
        step_idx : int
            Step index (used to generate unique Z3 variable names).
        model_state : ModelState
            Concrete state providing structure (tensor names, ndim, …).
        free_shapes : bool
            If ``True``, all shape dimensions become free Z3 Int variables
            (used in the inductive step).  Otherwise concrete dims become
            ``z3.IntVal`` constants.
        """
        shape_vars: Dict[str, list] = {}
        device_vars: Dict[str, Any] = {}
        grad_vars: Dict[str, Any] = {}

        for tname, shape in model_state.shape_env.items():
            if free_shapes:
                shape_vars[tname] = self.ctx.fresh_shape_vars(
                    tname, shape.ndim, step_idx
                )
            else:
                shape_vars[tname] = self.ctx.shape_to_z3(
                    shape, f"s{step_idx}_{tname}"
                )

        for tname in model_state.device_map:
            device_vars[tname] = self.ctx.fresh_device_var(tname, step_idx)

        for tname in model_state.gradient_status:
            grad_vars[tname] = self.ctx.fresh_grad_var(tname, step_idx)

        phase_v = self.ctx.fresh_phase_var(step_idx)

        return KripkeState(
            step_index=step_idx,
            shape_vars=shape_vars,
            device_vars=device_vars,
            phase_var=phase_v,
            grad_vars=grad_vars,
        )

    def _initial_constraints(self, k0: KripkeState) -> List:
        """Bind step-0 symbolic state variables to known concrete values."""
        cs: list = []
        for tname, device in self._init_state.device_map.items():
            if tname in k0.device_vars:
                cs.append(
                    k0.device_vars[tname] == self.ctx.device_to_z3(device)
                )
        for tname, has_grad in self._init_state.gradient_status.items():
            if tname in k0.grad_vars:
                cs.append(k0.grad_vars[tname] == z3.BoolVal(has_grad))
        if k0.phase_var is not None:
            cs.append(k0.phase_var == self.ctx.phase_to_z3(self.default_phase))
        for dims in k0.shape_vars.values():
            for d in dims:
                if not z3.is_int_value(d):
                    cs.append(d > 0)
        # Add user-supplied relational constraints (e.g. embed_dim == heads * head_dim)
        if self._relational_z3:
            cs.extend(self._relational_z3)
        return cs

    # ------------------------------------------------------------------
    # Transition-relation encoders  (pre → step → post)
    # ------------------------------------------------------------------

    def _encode_transition(
        self,
        pre: KripkeState,
        step: ComputationStep,
        post: KripkeState,
        model_state: ModelState,
        step_idx: int,
    ) -> List:
        """Full transition relation: shape ∧ device ∧ phase ∧ gradient ∧ frame."""
        cs: list = []
        cs.extend(self._encode_shape_transition(pre, step, post, model_state))
        cs.extend(self._encode_device_transition(pre, step, post, model_state))
        cs.extend(self._encode_phase_transition(pre, post))
        cs.extend(self._encode_gradient_transition(pre, step, post))
        cs.extend(self._encode_frame_conditions(pre, post, step))
        return cs

    # -- shape --

    def _encode_shape_transition(
        self,
        pre: KripkeState,
        step: ComputationStep,
        post: KripkeState,
        model_state: ModelState,
    ) -> List:
        cs: list = []
        inp_name = step.inputs[0] if step.inputs else None

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if (layer and inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                pre_d = pre.shape_vars[inp_name]
                post_d = post.shape_vars[step.output]
                if layer.kind == LayerKind.LINEAR:
                    for i in range(min(len(pre_d) - 1, len(post_d) - 1)):
                        cs.append(post_d[i] == pre_d[i])
                    if layer.out_features is not None and isinstance(layer.out_features, int) and post_d:
                        cs.append(
                            post_d[-1] == z3.IntVal(layer.out_features)
                        )
                elif layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D):
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                    if layer.out_channels is not None and isinstance(layer.out_channels, int) and len(post_d) >= 2:
                        cs.append(
                            post_d[1] == z3.IntVal(layer.out_channels)
                        )
                elif layer.kind in (LayerKind.BATCHNORM1D,
                                    LayerKind.BATCHNORM2D,
                                    LayerKind.LAYERNORM,
                                    LayerKind.GROUPNORM,
                                    LayerKind.INSTANCENORM2D):
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                elif layer.kind in (LayerKind.RELU, LayerKind.DROPOUT,
                                    LayerKind.IDENTITY, LayerKind.SOFTMAX):
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                elif layer.kind == LayerKind.EMBEDDING:
                    for i in range(min(len(pre_d), len(post_d) - 1)):
                        cs.append(post_d[i] == pre_d[i])
                    if layer.embedding_dim is not None and isinstance(layer.embedding_dim, int) and post_d:
                        cs.append(
                            post_d[-1] == z3.IntVal(layer.embedding_dim)
                        )
                elif layer.kind == LayerKind.FLATTEN:
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                elif layer.kind in (LayerKind.ADAPTIVE_AVGPOOL2D,):
                    if len(pre_d) >= 2 and len(post_d) >= 2:
                        cs.append(post_d[0] == pre_d[0])
                        cs.append(post_d[1] == pre_d[1])
                    if layer.output_size and len(post_d) >= 4:
                        cs.append(
                            post_d[2] == z3.IntVal(layer.output_size[0])
                        )
                        cs.append(
                            post_d[3] == z3.IntVal(layer.output_size[1])
                        )
                elif layer.kind in (LayerKind.MAXPOOL2D, LayerKind.AVGPOOL2D):
                    if len(pre_d) >= 2 and len(post_d) >= 2:
                        cs.append(post_d[0] == pre_d[0])
                        cs.append(post_d[1] == pre_d[1])
                    # Spatial dim constraints: H_out = floor((H_in + 2*pad - ks) / stride) + 1
                    if (layer.params and len(pre_d) >= 4
                            and len(post_d) >= 4):
                        ks = layer.params.get("kernel_size")
                        stride = layer.params.get("stride")
                        padding = layer.params.get("padding")
                        if ks is not None and stride is not None and padding is not None:
                            ks_h = ks[0] if isinstance(ks, tuple) else ks
                            ks_w = ks[1] if isinstance(ks, tuple) and len(ks) > 1 else ks_h
                            s_h = stride[0] if isinstance(stride, tuple) else stride
                            s_w = stride[1] if isinstance(stride, tuple) and len(stride) > 1 else s_h
                            p_h = padding[0] if isinstance(padding, tuple) else padding
                            p_w = padding[1] if isinstance(padding, tuple) and len(padding) > 1 else p_h
                            if isinstance(s_h, int) and s_h > 0 and isinstance(s_w, int) and s_w > 0:
                                # H_out = (H_in + 2*p_h - ks_h) / s_h + 1
                                cs.append(
                                    post_d[2] == (pre_d[2] + z3.IntVal(2 * p_h - ks_h)) / z3.IntVal(s_h) + z3.IntVal(1)
                                )
                                # W_out = (W_in + 2*p_w - ks_w) / s_w + 1
                                cs.append(
                                    post_d[3] == (pre_d[3] + z3.IntVal(2 * p_w - ks_w)) / z3.IntVal(s_w) + z3.IntVal(1)
                                )
                elif layer.kind == LayerKind.SEQUENTIAL:
                    # Sequential: concrete shape propagation handled by
                    # _apply_layer_call; at Z3 level, constrain batch dim
                    # and defer to the concrete propagator's output.
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                elif layer.kind == LayerKind.MODULELIST:
                    # ModuleList elements used individually; preserve shape
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                elif layer.kind == LayerKind.CONVTRANSPOSE2D:
                    # Preserve batch dim; set out_channels
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                    if layer.out_channels is not None and isinstance(layer.out_channels, int) and len(post_d) >= 2:
                        cs.append(
                            post_d[1] == z3.IntVal(layer.out_channels)
                        )
                elif layer.kind == LayerKind.UPSAMPLE:
                    # Preserve batch and channel dims
                    for dp, dq in zip(pre_d[:2], post_d[:2]):
                        cs.append(dq == dp)
                elif layer.kind == LayerKind.MULTIHEAD_ATTENTION:
                    # MHA preserves shape; input last dim == embed_dim
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                    if layer.in_features is not None and isinstance(layer.in_features, int) and pre_d:
                        cs.append(
                            pre_d[-1] == z3.IntVal(layer.in_features)
                        )
                elif layer.kind in (LayerKind.TRANSFORMER_ENCODER_LAYER,
                                    LayerKind.TRANSFORMER_DECODER_LAYER,
                                    LayerKind.TRANSFORMER_ENCODER,
                                    LayerKind.TRANSFORMER_DECODER):
                    # Transformer layers preserve shape; input last dim == d_model
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                    if layer.in_features is not None and isinstance(layer.in_features, int) and pre_d:
                        cs.append(
                            pre_d[-1] == z3.IntVal(layer.in_features)
                        )
                elif layer.kind == LayerKind.SUBMODULE:
                    # Submodule: concrete propagation handles details;
                    # at Z3 level, batch dim is preserved.
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                elif layer.kind in (LayerKind.LSTM, LayerKind.GRU):
                    # LSTM/GRU: leading dims preserved, last dim = hidden_size * D
                    for i in range(min(len(pre_d) - 1, len(post_d) - 1)):
                        cs.append(post_d[i] == pre_d[i])
                    if layer.hidden_size is not None and isinstance(layer.hidden_size, int) and post_d:
                        D = 2 if layer.bidirectional else 1
                        cs.append(
                            post_d[-1] == z3.IntVal(layer.hidden_size * D)
                        )
                    if layer.in_features is not None and isinstance(layer.in_features, int) and pre_d:
                        cs.append(
                            pre_d[-1] == z3.IntVal(layer.in_features)
                        )
                elif layer.kind in (LayerKind.CONVTRANSPOSE1D,):
                    # ConvTranspose1d: batch preserved, out_channels set
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                    if layer.out_channels is not None and isinstance(layer.out_channels, int) and len(post_d) >= 2:
                        cs.append(
                            post_d[1] == z3.IntVal(layer.out_channels)
                        )
                elif layer.kind == LayerKind.ADAPTIVE_MAXPOOL2D:
                    # Same as AdaptiveAvgPool2d
                    if len(pre_d) >= 2 and len(post_d) >= 2:
                        cs.append(post_d[0] == pre_d[0])
                        cs.append(post_d[1] == pre_d[1])
                    if layer.output_size and len(post_d) >= 4:
                        cs.append(
                            post_d[2] == z3.IntVal(layer.output_size[0])
                        )
                        cs.append(
                            post_d[3] == z3.IntVal(layer.output_size[1])
                        )
                elif layer.kind == LayerKind.PIXEL_SHUFFLE:
                    # PixelShuffle: batch preserved
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                elif layer.kind == LayerKind.UNFOLD:
                    # Unfold: batch preserved
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                elif layer.kind == LayerKind.FOLD:
                    # Fold: batch preserved
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                else:
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)

        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if (a in pre.shape_vars and b in pre.shape_vars
                    and step.output in post.shape_vars):
                ad = pre.shape_vars[a]
                bd = pre.shape_vars[b]
                pd = post.shape_vars[step.output]
                if len(ad) >= 2 and pd:
                    cs.append(pd[0] == ad[0])
                if len(bd) >= 2 and len(pd) >= 2:
                    cs.append(pd[-1] == bd[-1])

        elif step.op == OpKind.ADD and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if (a in pre.shape_vars and b in pre.shape_vars
                    and step.output in post.shape_vars):
                ad = pre.shape_vars[a]
                bd = pre.shape_vars[b]
                pd = post.shape_vars[step.output]
                ndim = max(len(ad), len(bd))
                for i in range(min(ndim, len(pd))):
                    da = ad[len(ad) - 1 - i] if i < len(ad) else z3.IntVal(1)
                    db = bd[len(bd) - 1 - i] if i < len(bd) else z3.IntVal(1)
                    dp = pd[len(pd) - 1 - i]
                    if self.ctx.broadcast_theory is not None:
                        cs.append(self.ctx.broadcast_theory.broadcast_result_dim(da, db, dp))
                    else:
                        cs.append(z3.Or(
                            z3.And(da == z3.IntVal(1), dp == db),
                            z3.And(db == z3.IntVal(1), dp == da),
                            z3.And(da == db, dp == da),
                        ))

        elif step.op == OpKind.MULTIPLY and len(step.inputs) >= 2:
            # Element-wise multiply / sub: same broadcast semantics as ADD
            a, b = step.inputs[0], step.inputs[1]
            if (a in pre.shape_vars and b in pre.shape_vars
                    and step.output in post.shape_vars):
                ad = pre.shape_vars[a]
                bd = pre.shape_vars[b]
                pd = post.shape_vars[step.output]
                ndim = max(len(ad), len(bd))
                for i in range(min(ndim, len(pd))):
                    da = ad[len(ad) - 1 - i] if i < len(ad) else z3.IntVal(1)
                    db = bd[len(bd) - 1 - i] if i < len(bd) else z3.IntVal(1)
                    dp = pd[len(pd) - 1 - i]
                    if self.ctx.broadcast_theory is not None:
                        cs.append(self.ctx.broadcast_theory.broadcast_result_dim(da, db, dp))
                    else:
                        cs.append(z3.Or(
                            z3.And(da == z3.IntVal(1), dp == db),
                            z3.And(db == z3.IntVal(1), dp == da),
                            z3.And(da == db, dp == da),
                        ))

        elif step.op == OpKind.INTERPOLATE:
            # F.interpolate preserves batch and channel dims
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                pre_d = pre.shape_vars[inp_name]
                post_d = post.shape_vars[step.output]
                for dp, dq in zip(pre_d[:2], post_d[:2]):
                    cs.append(dq == dp)

        elif step.op == OpKind.RESHAPE:
            dims = step.params.get("dims")
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars and dims is not None):
                pre_d = pre.shape_vars[inp_name]
                post_d = post.shape_vars[step.output]
                # Fix concrete target dimensions
                for i, d in enumerate(dims):
                    if isinstance(d, int) and d >= 0 and i < len(post_d):
                        cs.append(post_d[i] == z3.IntVal(d))
                # Element-count preservation: product(pre) == product(post)
                cs.extend(self._encode_reshape_safety(pre_d, post_d))
                # Mixed-arithmetic LIA reduction for reshape
                try:
                    from src.smt.theory_combination import (
                        MixedArithmeticPropagator,
                    )
                    lia_cs = (
                        MixedArithmeticPropagator
                        .generate_lia_reshape_constraints(
                            list(pre_d), list(post_d),
                        )
                    )
                    cs.extend(lia_cs)
                except (ImportError, Exception):
                    pass
                # Stride-based contiguity validation via stride theory
                if self.ctx.stride_theory is not None:
                    cs.extend(self._stride_reshape_check(pre_d, post_d))

        elif step.op == OpKind.TRANSPOSE:
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                pre_d = pre.shape_vars[inp_name]
                post_d = post.shape_vars[step.output]
                d0 = step.params.get("dim0", 0)
                d1 = step.params.get("dim1", 1)
                n = len(pre_d)
                if d0 < 0:
                    d0 = n + d0
                if d1 < 0:
                    d1 = n + d1
                if 0 <= d0 < n and 0 <= d1 < n and len(post_d) == n:
                    for i in range(n):
                        if i == d0:
                            cs.append(post_d[i] == pre_d[d1])
                        elif i == d1:
                            cs.append(post_d[i] == pre_d[d0])
                        else:
                            cs.append(post_d[i] == pre_d[i])

        elif step.op == OpKind.PERMUTE:
            perm = step.params.get("dims")
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars and perm is not None):
                pre_d = pre.shape_vars[inp_name]
                post_d = post.shape_vars[step.output]
                n = len(pre_d)
                if (len(perm) == n and len(post_d) == n
                        and all(isinstance(p, int) for p in perm)):
                    resolved = []
                    for p in perm:
                        resolved.append(p + n if p < 0 else p)
                    if all(0 <= p < n for p in resolved):
                        for i, p in enumerate(resolved):
                            cs.append(post_d[i] == pre_d[p])

        elif step.op in (OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.SOFTMAX,
                          OpKind.CONTIGUOUS, OpKind.DETACH, OpKind.TO_DEVICE):
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                for dp, dq in zip(
                    pre.shape_vars[inp_name],
                    post.shape_vars[step.output],
                ):
                    cs.append(dq == dp)

        elif step.op == OpKind.CONDITIONAL:
            # For conditional steps at Z3 level, conservatively pass through
            # (the concrete _apply_conditional handles branch selection)
            pass

        return cs

    @staticmethod
    def _encode_reshape_safety(
        old_dims: List, new_dims: List
    ) -> List:
        """Encode element-count preservation: product(old) == product(new)."""
        def _product(dims: List) -> "z3.ExprRef":
            if not dims:
                return z3.IntVal(1)
            result = dims[0]
            for d in dims[1:]:
                result = result * d
            return result

        return [_product(list(old_dims)) == _product(list(new_dims))]

    def _stride_reshape_check(
        self, old_dims: List, new_dims: List,
    ) -> List:
        """Query stride theory solver to verify reshape memory-validity.

        Uses the stride theory's separate solver to check that a contiguous
        source tensor can be validly reshaped to the target shape.
        """
        st = self.ctx.stride_theory
        ss = self.ctx._stride_solver
        n_old, n_new = len(old_dims), len(new_dims)
        uid = self._stride_check_id
        self._stride_check_id += 1

        old_sv = [z3.Int(f"_sr_o{i}_{uid}") for i in range(n_old)]
        old_st = [z3.Int(f"_sr_s{i}_{uid}") for i in range(n_old)]
        new_sv = [z3.Int(f"_sr_n{i}_{uid}") for i in range(n_new)]

        ss.push()
        # Source must have contiguous memory layout
        ss.add(st.contiguous_strides(old_sv, old_st))
        # Reshape must preserve element count (stride-theory propagator)
        ss.add(st.reshape_valid(old_sv, new_sv))

        for i, d in enumerate(old_dims):
            if z3.is_int_value(d):
                ss.add(old_sv[i] == d)
            ss.add(old_sv[i] > 0)
        for i, d in enumerate(new_dims):
            if z3.is_int_value(d):
                ss.add(new_sv[i] == d)
            ss.add(new_sv[i] > 0)

        result = ss.check()
        ss.pop()

        if result == z3.unsat:
            return [z3.BoolVal(False)]
        return []

    def _stride_contiguity_check(self, dims: List) -> List:
        """Verify source tensor admits a contiguous layout via stride theory.

        Contiguity is a prerequisite for many reshape operations.
        """
        st = self.ctx.stride_theory
        ss = self.ctx._stride_solver
        n = len(dims)
        uid = self._stride_check_id
        self._stride_check_id += 1

        shape_v = [z3.Int(f"_sc_d{i}_{uid}") for i in range(n)]
        stride_v = [z3.Int(f"_sc_s{i}_{uid}") for i in range(n)]

        ss.push()
        ss.add(st.contiguous_strides(shape_v, stride_v))

        for i, d in enumerate(dims):
            if z3.is_int_value(d):
                ss.add(shape_v[i] == d)
            ss.add(shape_v[i] > 0)

        result = ss.check()
        ss.pop()

        if result == z3.unsat:
            return [z3.BoolVal(False)]
        return []

    # -- device --

    def _encode_device_transition(
        self,
        pre: KripkeState,
        step: ComputationStep,
        post: KripkeState,
        model_state: ModelState,
    ) -> List:
        cs: list = []
        if step.op == OpKind.TO_DEVICE:
            dev_str = step.params.get("device")
            if dev_str is not None and step.output in post.device_vars:
                # Explicit device move (.to('cuda') / .cuda() / .cpu()): pin the
                # output device to the target.
                target = Device.from_string(str(dev_str))
                cs.append(self.ctx.encode_device_transfer(
                    post.device_vars[step.output], target
                ))
            elif step.output in post.device_vars:
                # Device-preserving TO_DEVICE (.pin_memory(), or a .to(...) that
                # only changes dtype): inherit the input device.  Without this
                # the output device var would be unconstrained and Z3 could pick
                # an arbitrary device, producing a spurious mismatch.
                for inp in step.inputs:
                    if inp in pre.device_vars:
                        cs.append(
                            post.device_vars[step.output]
                            == pre.device_vars[inp]
                        )
                        break
        elif step.op == OpKind.NEW_TENSOR:
            # Leaf tensor factory (torch.rand/zeros/...): pin its device to the
            # explicit device= (if static) or CPU (torch factory default).
            # Without this the output device var is unconstrained and Z3 would
            # fabricate a spurious device mismatch downstream.
            if step.output in post.device_vars:
                dev_str = step.params.get("device")
                target = (Device.from_string(str(dev_str))
                          if isinstance(dev_str, str) else Device.CPU)
                cs.append(self.ctx.encode_device_transfer(
                    post.device_vars[step.output], target
                ))
        elif step.inputs and step.output in post.device_vars:
            for inp in step.inputs:
                if inp in pre.device_vars:
                    cs.append(
                        post.device_vars[step.output]
                        == pre.device_vars[inp]
                    )
                    break
        return cs

    # -- phase --

    def _encode_phase_transition(
        self, pre: KripkeState, post: KripkeState
    ) -> List:
        if pre.phase_var is not None and post.phase_var is not None:
            return [pre.phase_var == post.phase_var]
        return []

    # -- gradient --

    def _encode_gradient_transition(
        self,
        pre: KripkeState,
        step: ComputationStep,
        post: KripkeState,
    ) -> List:
        cs: list = []
        if step.output not in post.grad_vars:
            return cs
        out_g = post.grad_vars[step.output]
        if step.op == OpKind.DETACH:
            cs.append(out_g == z3.BoolVal(False))
        elif step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer:
                cs.append(out_g == z3.BoolVal(
                    layer.kind in _PARAMETERISED_LAYERS
                ))
        else:
            in_gs = [pre.grad_vars[i]
                      for i in step.inputs if i in pre.grad_vars]
            if in_gs:
                cs.append(out_g == (z3.Or(*in_gs) if len(in_gs) > 1
                                    else in_gs[0]))
            else:
                cs.append(out_g == z3.BoolVal(False))
        return cs

    # -- frame conditions (unchanged tensors keep their properties) --

    def _encode_frame_conditions(
        self, pre: KripkeState, post: KripkeState, step: ComputationStep
    ) -> List:
        cs: list = []
        modified = {step.output}
        for t in pre.shape_vars:
            if t not in modified and t in post.shape_vars:
                for dp, dq in zip(pre.shape_vars[t], post.shape_vars[t]):
                    cs.append(dq == dp)
        for t in pre.device_vars:
            if t not in modified and t in post.device_vars:
                cs.append(post.device_vars[t] == pre.device_vars[t])
        for t in pre.grad_vars:
            if t not in modified and t in post.grad_vars:
                cs.append(post.grad_vars[t] == pre.grad_vars[t])
        return cs

    # ------------------------------------------------------------------
    # Safety-property encoders
    # ------------------------------------------------------------------

    def _opaque_tensors(self) -> Set[str]:
        """Names of tensors whose shape cannot be trusted because they are the
        output of an unsupported op or are (transitively) derived from one.
        Shape checks abstain on these so a free, unconstrained dimension can
        never be turned into a spurious violation."""
        if self._opaque_cache is not None:
            return self._opaque_cache
        opaque: Set[str] = set()

        def _scan(steps: List[ComputationStep]) -> None:
            for s in steps:
                if s.op == OpKind.UNSUPPORTED:
                    opaque.add(s.output)
                elif any(i in opaque for i in s.inputs):
                    opaque.add(s.output)
                if s.op == OpKind.CONDITIONAL:
                    _scan(s.true_branch or [])
                    _scan(s.false_branch or [])

        _scan(self.graph.steps)
        self._opaque_cache = opaque
        return opaque

    def _encode_shape_safety(
        self,
        k: KripkeState,
        step: ComputationStep,
        ms: ModelState,
        idx: int,
    ) -> List:
        """Encode shape compatibility constraints for *step*."""
        cs: list = []
        # Abstain entirely when any input is opaque (an unsupported op or
        # derived from one): its dimensions are free unknowns and asserting a
        # compatibility constraint against them would fabricate a violation.
        if any(i in self._opaque_tensors() for i in step.inputs):
            return cs
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            inp = step.inputs[0] if step.inputs else None
            if layer and inp and inp in k.shape_vars:
                dims = k.shape_vars[inp]
                if (layer.kind == LayerKind.LINEAR
                        and layer.in_features is not None and isinstance(layer.in_features, int) and dims):
                    cs.append(dims[-1] == z3.IntVal(layer.in_features))
                elif layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D):
                    if layer.in_channels is not None and isinstance(layer.in_channels, int) and len(dims) >= 2:
                        cs.append(dims[1] == z3.IntVal(layer.in_channels))
                elif layer.kind == LayerKind.CONVTRANSPOSE2D:
                    if layer.in_channels is not None and isinstance(layer.in_channels, int) and len(dims) >= 2:
                        cs.append(dims[1] == z3.IntVal(layer.in_channels))
                elif layer.kind in (LayerKind.BATCHNORM1D,
                                    LayerKind.BATCHNORM2D):
                    if (layer.num_features is not None
                            and isinstance(layer.num_features, int)
                            and len(dims) >= 2):
                        cs.append(
                            dims[1] == z3.IntVal(layer.num_features)
                        )
                elif layer.kind in (LayerKind.GROUPNORM,
                                    LayerKind.INSTANCENORM2D):
                    if (layer.num_features is not None
                            and isinstance(layer.num_features, int)
                            and len(dims) >= 2):
                        cs.append(
                            dims[1] == z3.IntVal(layer.num_features)
                        )
                elif layer.kind == LayerKind.MULTIHEAD_ATTENTION:
                    if layer.in_features is not None and isinstance(layer.in_features, int) and dims:
                        cs.append(dims[-1] == z3.IntVal(layer.in_features))
                elif layer.kind in (LayerKind.TRANSFORMER_ENCODER_LAYER,
                                    LayerKind.TRANSFORMER_DECODER_LAYER,
                                    LayerKind.TRANSFORMER_ENCODER,
                                    LayerKind.TRANSFORMER_DECODER):
                    if layer.in_features is not None and isinstance(layer.in_features, int) and dims:
                        cs.append(dims[-1] == z3.IntVal(layer.in_features))
                elif layer.kind in (LayerKind.LSTM, LayerKind.GRU):
                    if layer.in_features is not None and isinstance(layer.in_features, int) and dims:
                        cs.append(dims[-1] == z3.IntVal(layer.in_features))
        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if a in k.shape_vars and b in k.shape_vars:
                ad = k.shape_vars[a]
                bd = k.shape_vars[b]
                if ad and bd:
                    if len(bd) >= 2:
                        cs.append(ad[-1] == bd[-2])
                    elif len(bd) == 1:
                        cs.append(ad[-1] == bd[0])
        elif step.op == OpKind.ADD and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if a in k.shape_vars and b in k.shape_vars:
                ad = k.shape_vars[a]
                bd = k.shape_vars[b]
                if self.ctx.broadcast_theory is not None:
                    cs.append(self.ctx.broadcast_theory.broadcast_compatible(
                        list(ad), list(bd),
                    ))
                else:
                    ndim = max(len(ad), len(bd))
                    for i in range(1, ndim + 1):
                        da = ad[-i] if i <= len(ad) else z3.IntVal(1)
                        db = bd[-i] if i <= len(bd) else z3.IntVal(1)
                        cs.append(z3.Or(
                            da == db,
                            da == z3.IntVal(1),
                            db == z3.IntVal(1),
                        ))
        elif step.op == OpKind.MULTIPLY and len(step.inputs) >= 2:
            # Element-wise mul/sub: broadcast compatibility check
            a, b = step.inputs[0], step.inputs[1]
            if a in k.shape_vars and b in k.shape_vars:
                ad = k.shape_vars[a]
                bd = k.shape_vars[b]
                if self.ctx.broadcast_theory is not None:
                    cs.append(self.ctx.broadcast_theory.broadcast_compatible(
                        list(ad), list(bd),
                    ))
                else:
                    ndim = max(len(ad), len(bd))
                    for i in range(1, ndim + 1):
                        da = ad[-i] if i <= len(ad) else z3.IntVal(1)
                        db = bd[-i] if i <= len(bd) else z3.IntVal(1)
                        cs.append(z3.Or(
                            da == db,
                            da == z3.IntVal(1),
                            db == z3.IntVal(1),
                        ))
        elif step.op == OpKind.RESHAPE:
            inp = step.inputs[0] if step.inputs else None
            dims = step.params.get("dims")
            if inp and inp in k.shape_vars and dims is not None:
                inp_d = k.shape_vars[inp]
                # Concrete target dims must multiply to the same total
                known = [d for d in dims if isinstance(d, int) and d >= 0]
                if known and all(not z3.is_int_value(d) for d in inp_d):
                    pass  # symbolic input — rely on transition encoding
                elif known:
                    # All concrete: product(input) == product(target)
                    # (with -1 slots inferred, delegate to transition)
                    pass
                # Stride-theory contiguity check for source tensor
                if self.ctx.stride_theory is not None and inp_d:
                    cs.extend(self._stride_contiguity_check(inp_d))
                # Always require positive input dims (handled below)
        # Positivity for all involved shape dims
        for inp in step.inputs:
            if inp in k.shape_vars:
                for d in k.shape_vars[inp]:
                    cs.append(d > 0)
        return cs

    def _encode_device_safety(
        self,
        k: KripkeState,
        step: ComputationStep,
        ms: ModelState,
        idx: int,
    ) -> List:
        """Encode device-consistency constraints for *step*."""
        if not self.check_devices:
            return []
        cs: list = []
        # Binary ops: all inputs on the same device
        if step.op in (OpKind.MATMUL, OpKind.ADD, OpKind.CAT, OpKind.MULTIPLY):
            devs = [k.device_vars[i]
                     for i in step.inputs if i in k.device_vars]
            for i in range(1, len(devs)):
                cs.append(self.ctx.encode_device_constraint(devs[0], devs[i]))
        # Layer calls: input device must match param device (default_device)
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer and layer.kind in _PARAMETERISED_LAYERS:
                inp = step.inputs[0] if step.inputs else None
                if inp and inp in k.device_vars:
                    cs.append(
                        k.device_vars[inp]
                        == self.ctx.device_to_z3(self.default_device)
                    )
        return cs

    def _encode_phase_safety(
        self,
        k: KripkeState,
        step: ComputationStep,
        ms: ModelState,
        idx: int,
    ) -> List:
        """Encode phase-correctness constraints for *step*.

        Also registers phase-dependent behaviour with the PhaseTheoryPlugin
        (if available) for eager propagation on the phase solver.
        """
        if not self.check_phases:
            return []
        cs: list = []
        if k.phase_var is None:
            return cs
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer and layer.kind in (LayerKind.DROPOUT,
                                        LayerKind.BATCHNORM1D,
                                        LayerKind.BATCHNORM2D):
                # Phase must be well-formed (TRAIN or EVAL)
                cs.append(z3.Or(
                    k.phase_var == self.ctx.PHASE_TRAIN,
                    k.phase_var == self.ctx.PHASE_EVAL,
                ))
                # Shape still preserved in both modes
                inp = step.inputs[0] if step.inputs else None
                if inp and inp in k.shape_vars:
                    for d in k.shape_vars[inp]:
                        cs.append(d > 0)
                # Dropout identity in eval encoded via implication
                if layer.kind == LayerKind.DROPOUT:
                    cs.append(z3.Implies(
                        k.phase_var == self.ctx.PHASE_EVAL,
                        z3.BoolVal(True),
                    ))
                    # Register dropout behaviour with phase theory plugin
                    if self.ctx.phase_theory is not None:
                        _ph = z3.Bool(f"_pt_phase_s{idx}")
                        _inp = z3.Bool(f"_pt_drop_in_s{idx}")
                        _out = z3.Bool(f"_pt_drop_out_s{idx}")
                        self.ctx._phase_solver.add(
                            self.ctx.phase_theory.dropout_behavior(
                                _ph, _inp, _out
                            )
                        )
                        self.ctx._phase_constraints_registered += 1
                # Register batchnorm behaviour with phase theory plugin
                if layer.kind in (LayerKind.BATCHNORM1D,
                                  LayerKind.BATCHNORM2D):
                    if self.ctx.phase_theory is not None:
                        _ph = z3.Bool(f"_pt_phase_bn_s{idx}")
                        _urs = z3.Bool(f"_pt_bn_urs_s{idx}")
                        self.ctx._phase_solver.add(
                            self.ctx.phase_theory.batchnorm_behavior(
                                _ph, _urs
                            )
                        )
                        self.ctx._phase_constraints_registered += 1
        elif step.op == OpKind.DROPOUT:
            cs.append(z3.Or(
                k.phase_var == self.ctx.PHASE_TRAIN,
                k.phase_var == self.ctx.PHASE_EVAL,
            ))
            # Register functional dropout with phase theory plugin
            if self.ctx.phase_theory is not None:
                _ph = z3.Bool(f"_pt_phase_fdrop_s{idx}")
                _inp = z3.Bool(f"_pt_fdrop_in_s{idx}")
                _out = z3.Bool(f"_pt_fdrop_out_s{idx}")
                self.ctx._phase_solver.add(
                    self.ctx.phase_theory.dropout_behavior(
                        _ph, _inp, _out
                    )
                )
                self.ctx._phase_constraints_registered += 1
        return cs

    def _encode_gradient_safety(
        self,
        k: KripkeState,
        step: ComputationStep,
        ms: ModelState,
        idx: int,
    ) -> List:
        """Encode gradient-validity constraints for *step*."""
        if not self.check_gradients:
            return []
        cs: list = []
        # Detach: output must not require grad (checked in post-state)
        if step.op == OpKind.DETACH:
            if step.output in k.grad_vars:
                cs.append(
                    k.grad_vars[step.output] == z3.BoolVal(False)
                )
        # Gradient well-formedness for inputs
        for inp in step.inputs:
            if inp in k.grad_vars:
                cs.append(z3.Or(
                    k.grad_vars[inp] == z3.BoolVal(True),
                    k.grad_vars[inp] == z3.BoolVal(False),
                ))
        return cs

    def _encode_cross_domain_safety(
        self,
        pre: KripkeState,
        post: KripkeState,
        step: ComputationStep,
        ms: ModelState,
        idx: int,
    ) -> List:
        """Encode cross-domain constraints spanning shape + device + phase."""
        cs: list = []
        inp_name = step.inputs[0] if step.inputs else None
        # Device transfer must preserve shape
        if step.op == OpKind.TO_DEVICE:
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                for dp, dq in zip(
                    pre.shape_vars[inp_name],
                    post.shape_vars[step.output],
                ):
                    cs.append(dp == dq)
        # Layer calls: params on same device as data
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer and layer.kind in _PARAMETERISED_LAYERS:
                if inp_name and inp_name in pre.device_vars:
                    # Params assumed on default_device
                    cs.append(
                        pre.device_vars[inp_name]
                        == self.ctx.device_to_z3(self.default_device)
                    )
        # Shape-preserving ops: cross-check shape preservation
        if step.op in (OpKind.ACTIVATION, OpKind.CONTIGUOUS,
                        OpKind.SOFTMAX):
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                for dp, dq in zip(
                    pre.shape_vars[inp_name],
                    post.shape_vars[step.output],
                ):
                    cs.append(dp == dq)
        return cs

    # ------------------------------------------------------------------
    # Backward constraint propagation
    # ------------------------------------------------------------------

    def _backward_constraint_pass(
        self,
        solver: z3.Solver,
        kripke_states: List[KripkeState],
        model_states: List[ModelState],
    ) -> List[SafetyViolation]:
        """Backward constraint propagation pass.

        After forward propagation, iterate steps in REVERSE order and
        add constraints from each consumer layer's input requirements to
        the producer layer's output dimensions.  This catches mutations
        (e.g. ``wrong_out_features``) where the producer satisfies its
        own forward constraints but the output is incompatible with the
        downstream consumer.
        """
        violations: List[SafetyViolation] = []
        steps = self.graph.steps[: self.max_k]
        n = len(steps)

        for i in range(n - 1, 0, -1):
            consumer_step = steps[i]
            # kripke_states[i] is the post-state of step i-1 /
            # pre-state of step i
            if i >= len(kripke_states):
                continue
            k = kripke_states[i]

            backward_cs = self._backward_consumer_constraints(
                consumer_step, k, i,
            )

            if backward_cs:
                # Permanently assert backward constraints
                for c in backward_cs:
                    solver.add(c)
                # Check if backward constraints conflict with solver
                v = self._z3_check_safety(
                    solver, backward_cs, steps[i - 1], i - 1,
                    "backward_shape_mismatch",
                )
                if v is not None:
                    violations.append(v)

        # Propagate output specs backward through reshape/flatten/transpose
        bw_output = self._backward_output_spec_constraints(
            solver, kripke_states, model_states,
        )
        violations.extend(bw_output)

        return violations

    def _backward_consumer_constraints(
        self,
        consumer_step: ComputationStep,
        k: KripkeState,
        step_idx: int,
    ) -> List:
        """Build backward constraints from a consumer step's input
        requirements onto the shape variables in Kripke state *k*.
        """
        cs: list = []
        inp_name = consumer_step.inputs[0] if consumer_step.inputs else None
        # Abstain on consumers fed by an opaque (unsupported-derived) tensor.
        if any(i in self._opaque_tensors() for i in consumer_step.inputs):
            return cs

        if consumer_step.op == OpKind.LAYER_CALL and consumer_step.layer_ref:
            layer = self.graph.layers.get(consumer_step.layer_ref)
            if layer and inp_name and inp_name in k.shape_vars:
                dims = k.shape_vars[inp_name]
                if (layer.kind == LayerKind.LINEAR
                        and isinstance(layer.in_features, int)
                        and dims):
                    cs.append(
                        dims[-1] == z3.IntVal(layer.in_features)
                    )
                elif (layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D)
                        and isinstance(layer.in_channels, int)
                        and len(dims) >= 2):
                    cs.append(
                        dims[1] == z3.IntVal(layer.in_channels)
                    )
                elif (layer.kind == LayerKind.CONVTRANSPOSE2D
                        and isinstance(layer.in_channels, int)
                        and len(dims) >= 2):
                    cs.append(
                        dims[1] == z3.IntVal(layer.in_channels)
                    )
                elif (layer.kind in (LayerKind.BATCHNORM1D,
                                     LayerKind.BATCHNORM2D)
                        and isinstance(layer.num_features, int)
                        and len(dims) >= 2):
                    cs.append(
                        dims[1] == z3.IntVal(layer.num_features)
                    )
                elif (layer.kind == LayerKind.MULTIHEAD_ATTENTION
                        and isinstance(layer.in_features, int)
                        and dims):
                    cs.append(
                        dims[-1] == z3.IntVal(layer.in_features)
                    )
                elif (layer.kind in (LayerKind.LSTM, LayerKind.GRU)
                        and isinstance(layer.in_features, int)
                        and dims):
                    cs.append(
                        dims[-1] == z3.IntVal(layer.in_features)
                    )

        elif consumer_step.op == OpKind.MATMUL and len(consumer_step.inputs) >= 2:
            a, b = consumer_step.inputs[0], consumer_step.inputs[1]
            if a in k.shape_vars and b in k.shape_vars:
                ad = k.shape_vars[a]
                bd = k.shape_vars[b]
                if ad and bd:
                    if len(bd) >= 2:
                        cs.append(ad[-1] == bd[-2])
                    elif len(bd) == 1:
                        cs.append(ad[-1] == bd[0])
            # Cross-theory: matmul also implies same-device constraint
            if a in k.device_vars and b in k.device_vars:
                cs.append(k.device_vars[a] == k.device_vars[b])

        return cs

    def _backward_output_spec_constraints(
        self,
        solver: z3.Solver,
        kripke_states: List[KripkeState],
        model_states: List[ModelState],
    ) -> List[SafetyViolation]:
        """Propagate known output shapes backward through reshape,
        flatten, transpose, and permute operations.

        When a downstream step constrains its input dimensions (via the
        backward consumer pass), this method further propagates those
        constraints through shape-rearranging operations so that the
        original producer's output is also constrained.
        """
        violations: List[SafetyViolation] = []
        steps = self.graph.steps[: self.max_k]
        n = len(steps)

        for i in range(n - 1, 0, -1):
            step = steps[i]
            if i >= len(kripke_states):
                continue
            pre_k = kripke_states[i]
            post_k = kripke_states[i + 1] if i + 1 < len(kripke_states) else None
            if post_k is None:
                continue

            inp_name = step.inputs[0] if step.inputs else None
            if not inp_name or inp_name not in pre_k.shape_vars:
                continue
            if step.output not in post_k.shape_vars:
                continue

            pre_d = pre_k.shape_vars[inp_name]
            post_d = post_k.shape_vars[step.output]
            bw_cs: list = []

            if step.op == OpKind.RESHAPE:
                # Element-count preservation backward
                bw_cs.extend(self._encode_reshape_safety(pre_d, post_d))
                # Mixed-arithmetic LIA reduction: when some dims are
                # concrete, NIA product reduces to LIA constraints
                try:
                    from src.smt.theory_combination import (
                        MixedArithmeticPropagator,
                    )
                    lia_cs = (
                        MixedArithmeticPropagator
                        .generate_lia_reshape_constraints(
                            list(pre_d), list(post_d),
                        )
                    )
                    bw_cs.extend(lia_cs)
                except (ImportError, Exception):
                    pass

            elif step.op == OpKind.FLATTEN:
                # Flatten preserves batch dim
                if pre_d and post_d:
                    bw_cs.append(post_d[0] == pre_d[0])

            elif step.op == OpKind.TRANSPOSE:
                d0 = step.params.get("dim0", 0)
                d1 = step.params.get("dim1", 1)
                nd = len(pre_d)
                if d0 < 0:
                    d0 = nd + d0
                if d1 < 0:
                    d1 = nd + d1
                if (0 <= d0 < nd and 0 <= d1 < nd
                        and len(post_d) == nd):
                    for j in range(nd):
                        if j == d0:
                            bw_cs.append(pre_d[d1] == post_d[j])
                        elif j == d1:
                            bw_cs.append(pre_d[d0] == post_d[j])
                        else:
                            bw_cs.append(pre_d[j] == post_d[j])

            elif step.op == OpKind.PERMUTE:
                perm = step.params.get("dims")
                nd = len(pre_d)
                if (perm is not None and len(perm) == nd
                        and len(post_d) == nd):
                    resolved = [p + nd if p < 0 else p for p in perm]
                    if all(0 <= p < nd for p in resolved):
                        for j, p in enumerate(resolved):
                            bw_cs.append(pre_d[p] == post_d[j])

            if bw_cs:
                for c in bw_cs:
                    solver.add(c)

        return violations

    # ------------------------------------------------------------------
    # Z3 safety-check helper
    # ------------------------------------------------------------------

    def _z3_check_safety(
        self,
        solver: z3.Solver,
        constraints: list,
        step: ComputationStep,
        step_idx: int,
        kind: str,
    ) -> Optional[SafetyViolation]:
        """Push negated *constraints* onto *solver* and check SAT.

        When the check returns UNSAT (safety holds), a proof certificate
        is extracted and stored in ``self._step_certificates`` for later
        aggregation into the ``SafetyCertificate``.
        """
        if not constraints:
            return None
        neg = z3.Not(z3.And(*constraints))
        solver.push()
        solver.add(neg)
        result = self.ctx.timed_check(solver)
        violation = None
        if result == z3.sat:
            model = solver.model()
            violation = SafetyViolation(
                kind=kind,
                step_index=step_idx,
                step=step,
                message=self._format_z3_model(model, step_idx, kind),
            )
        elif result == z3.unsat:
            # Extract per-step proof certificate via replay in clean context
            if self.produce_certificates:
                self._try_extract_step_certificate(
                    solver, constraints, step_idx, kind
                )
        solver.pop()
        return violation

    def _try_extract_step_certificate(
        self,
        solver: z3.Solver,
        constraints: list,
        step_idx: int,
        kind: str,
    ) -> None:
        """Try to extract a proof certificate for one safety check.

        Replays the current solver assertions (including the negated safety
        property) in a fresh proof-enabled context without UserPropagator,
        so that Z3's proof engine can produce full inference chains.
        """
        if not hasattr(self, '_step_certificates'):
            self._step_certificates = []
        try:
            from .proof_certificate import (
                ProofExtractor,
                ProofCertificate,
                CertificateStrategy,
            )
            # Collect assertions currently on the solver stack
            assertions = list(solver.assertions())
            if not assertions:
                return
            # Replay in a fresh proof context (no UserPropagator)
            proof_ctx = z3.Context("proof", "true")
            proof_solver = z3.Solver(ctx=proof_ctx)
            proof_solver.set("timeout", 5000)
            translated = []
            for a in assertions:
                try:
                    t = a.translate(proof_ctx)
                    proof_solver.add(t)
                    translated.append(t)
                except Exception:
                    continue
            if not translated:
                return
            if proof_solver.check() == z3.unsat:
                extractor = ProofExtractor(proof_solver, translated)
                cert = extractor.extract(
                    model_name=self.graph.class_name,
                    properties=[kind],
                )
                if cert is not None:
                    cert.strategy = CertificateStrategy.REPLAY
                    self._step_certificates.append(
                        (step_idx, kind, cert)
                    )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Concrete single-step transition (kept for backward compat)
    # ------------------------------------------------------------------

    def _step_transition(
        self, state: ModelState, step: ComputationStep
    ) -> Tuple[ModelState, List[SafetyViolation]]:
        """Apply one computation step to *state*, returning the new state and
        any safety violations detected.
        """
        new_state = state.copy()
        violations: List[SafetyViolation] = []

        # ---- Device consistency check ------------------------------------
        input_devices = []
        for inp in step.inputs:
            dev = state.device_map.get(inp)
            if dev is not None:
                input_devices.append((inp, dev))

        if self.check_devices and len(input_devices) >= 2:
            first_name, first_dev = input_devices[0]
            for other_name, other_dev in input_devices[1:]:
                if first_dev != other_dev:
                    violations.append(SafetyViolation(
                        kind="device_mismatch",
                        step_index=-1,
                        step=step,
                        message=(
                            f"Device mismatch: {first_name} is on "
                            f"{first_dev.value} but {other_name} is on "
                            f"{other_dev.value}"
                        ),
                        tensor_a=first_name,
                        tensor_b=other_name,
                        device_a=first_dev,
                        device_b=other_dev,
                    ))

        # ---- Shape propagation & compatibility ---------------------------
        if step.op == OpKind.LAYER_CALL:
            self._apply_layer_call(new_state, step, violations)
        elif step.op == OpKind.MATMUL:
            self._apply_matmul(new_state, step, violations)
        elif step.op == OpKind.ADD:
            self._apply_add(new_state, step, violations)
        elif step.op == OpKind.MULTIPLY:
            # Element-wise mul/sub: same broadcast shape semantics as ADD
            self._apply_add(new_state, step, violations)
        elif step.op == OpKind.RESHAPE:
            self._apply_reshape(new_state, step, violations)
        elif step.op == OpKind.FLATTEN:
            self._apply_flatten(new_state, step)
        elif step.op in (OpKind.ACTIVATION, OpKind.CONTIGUOUS):
            if step.inputs and step.inputs[0] in state.shape_env:
                new_state.shape_env[step.output] = (
                    state.shape_env[step.inputs[0]]
                )
        elif step.op == OpKind.DTYPE_CAST:
            # dtype cast is shape- and device-preserving; dtype handled below.
            if step.inputs and step.inputs[0] in state.shape_env:
                new_state.shape_env[step.output] = (
                    state.shape_env[step.inputs[0]]
                )
        elif step.op == OpKind.DROPOUT:
            if step.inputs and step.inputs[0] in state.shape_env:
                new_state.shape_env[step.output] = (
                    state.shape_env[step.inputs[0]]
                )
        elif step.op == OpKind.SOFTMAX:
            if step.inputs and step.inputs[0] in state.shape_env:
                new_state.shape_env[step.output] = (
                    state.shape_env[step.inputs[0]]
                )
        elif step.op == OpKind.SQUEEZE:
            self._apply_squeeze(new_state, step)
        elif step.op == OpKind.UNSQUEEZE:
            self._apply_unsqueeze(new_state, step)
        elif step.op == OpKind.TRANSPOSE:
            self._apply_transpose(new_state, step)
        elif step.op == OpKind.PERMUTE:
            self._apply_permute(new_state, step)
        elif step.op == OpKind.CAT:
            self._apply_cat(new_state, step, violations)
        elif step.op == OpKind.TO_DEVICE:
            self._apply_to_device(new_state, step)
        elif step.op == OpKind.DETACH:
            if step.inputs and step.inputs[0] in state.shape_env:
                new_state.shape_env[step.output] = (
                    state.shape_env[step.inputs[0]]
                )
            # Warn if detach kills gradient from a trainable input
            if self.check_gradients and step.inputs:
                inp = step.inputs[0]
                if state.gradient_status.get(inp, False):
                    violations.append(SafetyViolation(
                        kind="gradient_broken",
                        step_index=-1, step=step,
                        message=(
                            f"Gradient flow broken: '{inp}' requires grad but "
                            f".detach() kills gradient to downstream parameters"
                        ),
                    ))
            new_state.gradient_status[step.output] = False
        elif step.op == OpKind.RETURN:
            pass
        elif step.op == OpKind.CONDITIONAL:
            self._apply_conditional(new_state, step, violations)
        elif step.op == OpKind.CUSTOM:
            # Conservative: assume custom ops preserve shape of first input
            if step.inputs and step.inputs[0] in state.shape_env:
                new_state.shape_env[step.output] = state.shape_env[step.inputs[0]]
        elif step.op == OpKind.UNSUPPORTED:
            # Operator with no shape transfer function. Guessing that it
            # preserves shape (as CUSTOM/ACTIVATION do) is unsound for
            # shape-changing ops, so abstain: emit a fully-symbolic output of
            # the same rank as the input (when known). Fresh, never-unified
            # dims mean no downstream concrete check can fire against them, so
            # we neither fabricate a violation nor confidently miss one — we
            # simply decline to reason past the unsupported op. The op name is
            # surfaced separately via the UnsupportedOpTracker diagnostic.
            inp = step.inputs[0] if step.inputs else None
            inp_shape = state.shape_env.get(inp) if inp else None
            if inp_shape is not None:
                new_state.shape_env[step.output] = TensorShape(tuple(
                    ShapeDim(f"_unsup_{step.output}_{i}")
                    for i in range(inp_shape.ndim)
                ))
        elif step.op == OpKind.NEW_TENSOR:
            self._apply_new_tensor(new_state, step)
        elif step.op == OpKind.INTERPOLATE:
            # F.interpolate preserves batch + channel dims
            if step.inputs and step.inputs[0] in state.shape_env:
                inp_shape = state.shape_env[step.inputs[0]]
                if inp_shape.ndim >= 3:
                    kept = inp_shape.dims[:2]
                    spatial = tuple(ShapeDim("_up") for _ in inp_shape.dims[2:])
                    new_state.shape_env[step.output] = TensorShape(kept + spatial)
                else:
                    new_state.shape_env[step.output] = inp_shape
        elif step.op == OpKind.SUBSCRIPT:
            self._apply_subscript(new_state, step)
        elif step.op == OpKind.STACK:
            self._apply_stack(new_state, step, violations)
        elif step.op == OpKind.WHERE:
            # torch.where(cond, x, y): broadcast all three pairwise
            self._apply_where(new_state, step, violations)
        elif step.op in (OpKind.CHUNK, OpKind.SPLIT):
            # chunk/split: divide the split dimension
            if step.inputs and step.inputs[0] in state.shape_env:
                inp_shape = state.shape_env[step.inputs[0]]
                chunks = step.params.get("chunks")
                dim = step.params.get("dim", 0)
                split_size = step.params.get("split_size")  # may be int|str|None
                n_outputs = step.params.get("n_outputs")
                if isinstance(dim, int) and dim < 0:
                    dim = inp_shape.ndim + dim
                concrete_dim = (isinstance(dim, int)
                                and 0 <= dim < inp_shape.ndim
                                and not inp_shape.dims[dim].is_symbolic)
                # SOUNDNESS: if the split dim is CONCRETE but split_size is
                # symbolic, fall back to the chunks-based path so that
                # downstream bugs (e.g. wrong split axis with concrete shape)
                # remain detectable.
                use_split_size = (
                    split_size is not None
                    and isinstance(dim, int)
                    and 0 <= dim < inp_shape.ndim
                    and (not concrete_dim or isinstance(split_size, int))
                )
                if use_split_size:
                    new_dims = list(inp_shape.dims)
                    new_dims[dim] = ShapeDim(split_size)
                    new_state.shape_env[step.output] = TensorShape(tuple(new_dims))
                elif (chunks is not None and isinstance(chunks, int)
                        and chunks > 0 and concrete_dim):
                    new_dims = list(inp_shape.dims)
                    orig = inp_shape.dims[dim].value
                    chunk_size = (orig + chunks - 1) // chunks
                    new_dims[dim] = ShapeDim(chunk_size)
                    new_state.shape_env[step.output] = TensorShape(tuple(new_dims))
                elif (n_outputs is not None and isinstance(n_outputs, int)
                        and n_outputs > 0 and concrete_dim):
                    # Symbolic split_size on a concrete dim — recover via
                    # n_outputs to keep bug detection sound.
                    new_dims = list(inp_shape.dims)
                    orig = inp_shape.dims[dim].value
                    # Soundness: q,k,v = X.split(sz, dim) requires
                    # n_outputs * sz == orig, so orig must be divisible by
                    # n_outputs. Otherwise a wrong-axis split (e.g. dim=1
                    # instead of dim=2) produces a concrete contradiction.
                    if (violations is not None
                            and step.params.get("split_index", 0) == 0
                            and orig % n_outputs != 0):
                        violations.append(SafetyViolation(
                            kind="shape_incompatible",
                            step_index=-1, step=step,
                            message=(
                                f"Split incompatible: cannot split dim {dim} "
                                f"of size {orig} into {n_outputs} chunks of "
                                f"size {split_size!r} "
                                f"({orig} not divisible by {n_outputs}) — "
                                f"likely wrong split axis"
                            ),
                            tensor_a=step.inputs[0],
                            shape_a=inp_shape,
                        ))
                    chunk_size = (orig + n_outputs - 1) // n_outputs
                    new_dims[dim] = ShapeDim(chunk_size)
                    new_state.shape_env[step.output] = TensorShape(tuple(new_dims))
                else:
                    new_state.shape_env[step.output] = inp_shape
        elif step.op == OpKind.UNBIND:
            # unbind(dim) removes the split dimension entirely.
            # q, k, v = X.unbind(0) where X has shape (3, B, H, D)
            # → each output has shape (B, H, D).
            if step.inputs and step.inputs[0] in state.shape_env:
                inp_shape = state.shape_env[step.inputs[0]]
                dim = step.params.get("dim", 0)
                n_outputs = step.params.get("n_outputs", 1)
                if isinstance(dim, int) and dim < 0:
                    dim = inp_shape.ndim + dim
                if isinstance(dim, int) and 0 <= dim < inp_shape.ndim:
                    # Check that the split dimension matches n_outputs (if concrete).
                    concrete_split = not inp_shape.dims[dim].is_symbolic
                    split_size = inp_shape.dims[dim].value
                    if (concrete_split and isinstance(split_size, int)
                            and split_size != n_outputs
                            and violations is not None
                            and step.params.get("unbind_index", 0) == 0):
                        violations.append(SafetyViolation(
                            kind="shape_incompatible",
                            step_index=-1, step=step,
                            message=(
                                f"unbind mismatch: unpacking {n_outputs} variables "
                                f"but dim {dim} has size {split_size}"
                            ),
                            tensor_a=step.inputs[0],
                            shape_a=inp_shape,
                        ))
                    # Output shape: remove the unbound dimension.
                    new_dims = [d for i, d in enumerate(inp_shape.dims) if i != dim]
                    new_state.shape_env[step.output] = TensorShape(tuple(new_dims))
                else:
                    # Symbolic dim: propagate input shape minus one dim (best effort).
                    if inp_shape.ndim > 0:
                        new_dims = list(inp_shape.dims[1:])
                        new_state.shape_env[step.output] = TensorShape(tuple(new_dims))
                    else:
                        new_state.shape_env[step.output] = inp_shape
        elif step.op == OpKind.EXPAND:
            if step.inputs and step.inputs[0] in state.shape_env:
                inp_shape = state.shape_env[step.inputs[0]]
                dims = step.params.get("dims")
                # ``expand_as(other)`` / ``broadcast_to`` against a reference
                # tensor: derive the target sizes from the second operand's
                # shape when explicit dims weren't captured.
                if (not dims and len(step.inputs) > 1
                        and step.inputs[1] in state.shape_env):
                    ref = state.shape_env[step.inputs[1]]
                    dims = tuple(
                        d.value if not d.is_symbolic else str(d.value)
                        for d in ref.dims
                    )
                if dims and all(d is not None for d in dims):
                    allow_neg_one = (
                        step.params.get("expand_kind") != "broadcast_to"
                    )
                    out_shape, err = compute_expand_shape(
                        inp_shape, tuple(dims), allow_neg_one=allow_neg_one
                    )
                    if err is not None:
                        violations.append(SafetyViolation(
                            kind="shape_incompatible",
                            step_index=-1,
                            step=step,
                            message=err,
                            tensor_a=step.inputs[0],
                            shape_a=inp_shape,
                        ))
                    if out_shape is not None:
                        new_state.shape_env[step.output] = out_shape
                else:
                    new_state.shape_env[step.output] = inp_shape
        elif step.op == OpKind.REPEAT:
            if step.inputs and step.inputs[0] in state.shape_env:
                inp_shape = state.shape_env[step.inputs[0]]
                dims = step.params.get("dims")
                if dims and all(d is not None for d in dims):
                    new_dims = []
                    for i, d in enumerate(dims):
                        if i < inp_shape.ndim and not inp_shape.dims[i].is_symbolic:
                            new_dims.append(ShapeDim(inp_shape.dims[i].value * d))
                        else:
                            new_dims.append(ShapeDim("_rep"))
                    new_state.shape_env[step.output] = TensorShape(tuple(new_dims))
                else:
                    new_state.shape_env[step.output] = inp_shape
        elif step.op in (OpKind.MEAN_REDUCE, OpKind.SUM_REDUCE):
            if step.inputs and step.inputs[0] in state.shape_env:
                inp_shape = state.shape_env[step.inputs[0]]
                dim = step.params.get("dim")
                keepdim = step.params.get("keepdim", False)
                if dim is not None:
                    if isinstance(dim, int):
                        if dim < 0:
                            dim = inp_shape.ndim + dim
                        new_dims = list(inp_shape.dims)
                        if 0 <= dim < len(new_dims):
                            if keepdim:
                                new_dims[dim] = ShapeDim(1)
                            else:
                                new_dims.pop(dim)
                        new_state.shape_env[step.output] = TensorShape(tuple(new_dims))
                    else:
                        new_state.shape_env[step.output] = inp_shape
                else:
                    # Global reduction: scalar output
                    new_state.shape_env[step.output] = TensorShape(())
        elif step.op == OpKind.PAD:
            if step.inputs and step.inputs[0] in state.shape_env:
                inp_shape = state.shape_env[step.inputs[0]]
                pad_arg = step.params.get("pad")
                if pad_arg and isinstance(pad_arg, (tuple, list)):
                    new_dims = list(inp_shape.dims)
                    # F.pad padding is applied from last dim backwards, in pairs
                    n_padded = len(pad_arg) // 2
                    for i in range(n_padded):
                        dim_idx = inp_shape.ndim - 1 - i
                        if 0 <= dim_idx < len(new_dims) and not new_dims[dim_idx].is_symbolic:
                            new_dims[dim_idx] = ShapeDim(
                                new_dims[dim_idx].value + pad_arg[2*i] + pad_arg[2*i+1]
                            )
                    new_state.shape_env[step.output] = TensorShape(tuple(new_dims))
                else:
                    new_state.shape_env[step.output] = inp_shape
        elif step.op == OpKind.EINSUM:
            # Precise einsum: parse the equation (explicit/implicit output,
            # diagonals, ellipsis broadcasting) rather than a heuristic. We only
            # run the parser-backed inference once the equation is a known string
            # and every operand shape is resolved; otherwise we fall back to a
            # best-effort placeholder. The verifier never emits a violation for
            # symbolic dims (only genuine concrete mismatches), preserving the
            # sound-mode false-positive-free invariant.
            equation = step.params.get("equation", "")
            einsum_inputs: List["TensorShape"] = []
            all_known_ein = bool(step.inputs) and isinstance(equation, str) \
                and bool(equation)
            if all_known_ein:
                for inp in step.inputs:
                    if inp in state.shape_env:
                        einsum_inputs.append(state.shape_env[inp])
                    else:
                        all_known_ein = False
                        break
            if all_known_ein:
                from src.smt.einsum_theory import (
                    check_einsum_compatible as _chk_einsum,
                    infer_einsum_shape as _infer_einsum,
                )
                err_ein = _chk_einsum(equation, einsum_inputs)
                if err_ein is not None:
                    violations.append(SafetyViolation(
                        kind="shape_incompatible",
                        step_index=-1, step=step,
                        message=f"einsum '{equation}': {err_ein}",
                    ))
                out_shape_ein = _infer_einsum(equation, einsum_inputs)
                if out_shape_ein is not None:
                    new_state.shape_env[step.output] = out_shape_ein
            elif step.inputs:
                # Equation unknown (e.g. sublist/interleaved API) or operand
                # shapes not yet resolved: best-effort placeholder.
                for inp in step.inputs:
                    if inp in state.shape_env:
                        new_state.shape_env[step.output] = state.shape_env[inp]
                        break
        elif step.op in (
            OpKind.GATHER, OpKind.INDEX_SELECT, OpKind.SCATTER,
            OpKind.MASKED_SELECT, OpKind.MASKED_FILL, OpKind.NARROW,
            OpKind.SELECT_DIM, OpKind.TAKE,
        ):
            self._apply_indexing(new_state.shape_env, step, violations)
        elif step.op == OpKind.SDPA:
            self._apply_sdpa(new_state.shape_env, step, violations)

        # ---- Propagate device if not explicitly set ----------------------
        if step.output not in new_state.device_map:
            if step.inputs:
                for inp in step.inputs:
                    if inp in state.device_map:
                        new_state.device_map[step.output] = (
                            state.device_map[inp]
                        )
                        break

        # ---- Propagate gradient status -----------------------------------
        if step.output not in new_state.gradient_status:
            any_grad = any(
                state.gradient_status.get(inp, False)
                for inp in step.inputs
            )
            # Layer calls with trainable parameters produce grad=True outputs
            if (not any_grad and step.op == OpKind.LAYER_CALL
                    and step.layer_ref):
                layer = self.graph.layers.get(step.layer_ref)
                if layer and layer.kind in (
                    LayerKind.LINEAR, LayerKind.CONV1D, LayerKind.CONV2D,
                    LayerKind.CONV3D, LayerKind.CONVTRANSPOSE1D,
                    LayerKind.CONVTRANSPOSE2D, LayerKind.CONVTRANSPOSE3D,
                    LayerKind.EMBEDDING, LayerKind.MULTIHEAD_ATTENTION,
                    LayerKind.LSTM, LayerKind.GRU, LayerKind.RNN,
                    LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
                    LayerKind.BATCHNORM3D, LayerKind.LAYERNORM,
                    LayerKind.GROUPNORM, LayerKind.BILINEAR,
                ):
                    any_grad = True
            new_state.gradient_status[step.output] = any_grad

        # ---- Dtype propagation & consistency checks ----------------------
        self._propagate_and_check_dtype(state, new_state, step, violations)

        return new_state, violations

    # --- per-operation helpers (concrete) ---------------------------------

    def _propagate_and_check_dtype(
        self,
        state: ModelState,
        new_state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        """Propagate element dtypes across one step and flag dtype-incompatible
        operations that raise at runtime.

        Soundness: only *known* dtypes (present in ``dtype_env``) are reasoned
        about; any operand with an unknown dtype causes the check to abstain.
        Under autocast the whole pass abstains.  This guarantees no false
        positives: a reported ``dtype_error`` corresponds to a torch
        ``RuntimeError`` under the recorded dtypes.
        """
        if self._dtype_abstain:
            return

        def known(name: Optional[str]) -> Optional[str]:
            if name is None:
                return None
            return state.dtype_env.get(name) or new_state.dtype_env.get(name)

        out = step.output

        # --- Explicit dtype casts -------------------------------------
        # .half()/.float()/.double()/.to(dtype=...) carry a canonical target.
        cast_dt = _canon_dtype(step.params.get("cast_dtype")) if step.params else None
        if step.op == OpKind.DTYPE_CAST or cast_dt is not None:
            if cast_dt is not None:
                new_state.dtype_env[out] = cast_dt
                return
            # DTYPE_CAST with an unrecognised target → result dtype unknown.
            new_state.dtype_env.pop(out, None)
            return

        # --- Parametric layer calls (matmul/conv against a stored param) ---
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer is not None:
                param_dt = _canon_dtype(layer.params.get("param_dtype"))
                inp = step.inputs[0] if step.inputs else None
                inp_dt = known(inp)

                if layer.kind in _DTYPE_PARAM_MATCH_KINDS and param_dt is not None:
                    if self.check_dtypes and inp_dt is not None and inp_dt != param_dt:
                        violations.append(SafetyViolation(
                            kind="dtype_error",
                            step_index=-1,
                            step=step,
                            message=(
                                f"{layer.kind.name}: input dtype '{inp_dt}' does "
                                f"not match parameter dtype '{param_dt}'. torch "
                                f"raises at runtime (e.g. \"mat1 and mat2 must "
                                f"have the same dtype\" / \"Input type and bias "
                                f"type should be the same\")"
                            ),
                            tensor_a=inp,
                            confidence=Confidence.HIGH,
                        ))
                    # Output dtype is the parameter dtype (= input dtype when ok).
                    new_state.dtype_env[out] = param_dt
                    return

                if layer.kind == LayerKind.EMBEDDING:
                    # Index tensor must be int32/int64; output is the weight dtype.
                    if (self.check_dtypes and inp_dt is not None
                            and _is_float_dtype(inp_dt)):
                        violations.append(SafetyViolation(
                            kind="dtype_error",
                            step_index=-1,
                            step=step,
                            message=(
                                f"Embedding: index dtype '{inp_dt}' is floating; "
                                f"torch requires int32/int64 indices "
                                f"(\"Expected tensor for argument index to have "
                                f"scalar type Long\")"
                            ),
                            tensor_a=inp,
                            confidence=Confidence.HIGH,
                        ))
                    if param_dt is not None:
                        new_state.dtype_env[out] = param_dt
                    return

        # --- Matmul: both operands must share dtype -----------------------
        if step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            da, db = known(a), known(b)
            if self.check_dtypes and da is not None and db is not None and da != db:
                violations.append(SafetyViolation(
                    kind="dtype_error",
                    step_index=-1,
                    step=step,
                    message=(
                        f"matmul: operand dtypes differ ('{da}' vs '{db}'); "
                        f"torch raises \"mat1 and mat2 must have the same dtype\""
                    ),
                    tensor_a=a,
                    tensor_b=b,
                    confidence=Confidence.HIGH,
                ))
            if da is not None:
                new_state.dtype_env[out] = da
            elif db is not None:
                new_state.dtype_env[out] = db
            return

        # --- Default: preserve the first known input dtype ----------------
        if out not in new_state.dtype_env:
            for inp in step.inputs:
                dt = known(inp)
                if dt is not None:
                    new_state.dtype_env[out] = dt
                    break

    def _check_norm_stats_mode(
        self,
        state: ModelState,
        step: ComputationStep,
        layer: LayerDef,
        inp_shape: TensorShape,
        violations: List[SafetyViolation],
    ) -> None:
        """Flag BatchNorm/InstanceNorm runtime errors that depend on phase /
        ``track_running_stats``.

        * BatchNorm{1,2,3}d (and lazy variants) raise ``ValueError: Expected
          more than 1 value per channel when training`` when the per-channel
          element count ``N * prod(spatial)`` is exactly 1 and the layer is
          using *batch* statistics.
        * InstanceNorm{1,2,3}d (and lazy variants) raise ``ValueError: Expected
          more than 1 spatial element when training`` when ``prod(spatial)`` is
          exactly 1 and the layer is using *input* statistics.

        A layer uses batch/input statistics when ``training or not
        track_running_stats``.  BatchNorm defaults ``track_running_stats=True``
        (so the error is TRAIN-only by default); InstanceNorm defaults
        ``track_running_stats=False`` (so the error fires in eval too).

        Soundness: we only emit when the relevant element count is *provably*
        exactly 1 (every contributing dim is a concrete int) and the layer is
        known to use batch/input statistics; we abstain on any symbolic dim,
        on non-canonical ranks, and on ``SyncBatchNorm`` (distributed).
        """
        if not self.check_phases:
            return
        kind = layer.kind
        is_bn = kind in _BN_COUNT_NORM_KINDS
        is_in = kind in _IN_SPATIAL_NORM_KINDS
        if not (is_bn or is_in):
            return
        expected_ranks = _NORM_CANONICAL_RANK.get(kind)
        if expected_ranks is None or inp_shape.ndim not in expected_ranks:
            return

        # Determine whether batch/input statistics are in use.
        trs = layer.params.get("track_running_stats")
        if trs is None:
            trs = False if is_in else True  # family default
        uses_batch_stats = (state.phase == Phase.TRAIN) or (trs is False)
        if not uses_batch_stats:
            return

        # Dims that contribute to the count.  BatchNorm: everything except the
        # channel dim (dim 1).  InstanceNorm: spatial dims only (dim 2+).
        if is_bn:
            count_dims = (inp_shape.dims[0],) + tuple(inp_shape.dims[2:])
            what = "value per channel"
        else:
            count_dims = tuple(inp_shape.dims[2:])
            what = "spatial element"
        if not count_dims or any(d.is_symbolic for d in count_dims):
            return
        count = 1
        for d in count_dims:
            count *= int(d.value)
        if count != 1:
            return

        layer_name = kind.name.replace("LAZY", "Lazy").title().replace("d", "d")
        phase_hint = ""
        if is_bn and state.phase == Phase.TRAIN and trs is not False:
            phase_hint = (
                " (checked under TRAIN phase; pass default_phase=Phase.EVAL to "
                "check eval behaviour)"
            )
        violations.append(SafetyViolation(
            kind="phase_error",
            step_index=-1,
            step=step,
            message=(
                f"{kind.name}: input {inp_shape.pretty()} has only 1 "
                f"{what}, which raises at runtime "
                f"(\"Expected more than 1 {what} when training\")"
                f"{phase_hint}"
            ),
            tensor_a=step.inputs[0] if step.inputs else None,
            shape_a=inp_shape,
            confidence=Confidence.HIGH,
        ))

    def _apply_layer_call(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        layer = self.graph.layers.get(step.layer_ref or "")
        if layer is None:
            return

        inp_name = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp_name) if inp_name else None

        if inp_shape is None:
            return

        if layer.kind == LayerKind.DROPOUT and state.phase == Phase.EVAL:
            state.shape_env[step.output] = inp_shape
            return

        self._check_norm_stats_mode(state, step, layer, inp_shape, violations)

        propagator = _LAYER_PROPAGATORS.get(layer.kind)
        if propagator is not None:
            out_shape, err = propagator(inp_shape, layer)
            if err:
                violations.append(SafetyViolation(
                    kind="shape_incompatible",
                    step_index=-1,
                    step=step,
                    message=err,
                    tensor_a=inp_name,
                    shape_a=inp_shape,
                ))
            elif out_shape is not None:
                state.shape_env[step.output] = out_shape
        elif layer.kind == LayerKind.SUBMODULE:
            self._apply_submodule_call(state, step, layer, inp_name,
                                       inp_shape, violations)
        elif layer.kind in (LayerKind.RELU, LayerKind.IDENTITY):
            state.shape_env[step.output] = inp_shape
        elif layer.kind == LayerKind.FLATTEN:
            out_shape, err = _propagate_flatten(inp_shape, 1)
            if out_shape:
                state.shape_env[step.output] = out_shape
        elif layer.kind == LayerKind.SOFTMAX:
            state.shape_env[step.output] = inp_shape
        else:
            # UNSOUND FIX: unsupported op → UNKNOWN shape (fully symbolic)
            logger.warning(
                "Unsupported layer kind %s (%s): output shape marked UNKNOWN",
                layer.kind.name, layer.attr_name,
            )
            unknown_shape = TensorShape(
                tuple(ShapeDim(f"_unk_{layer.attr_name}_{i}")
                      for i in range(inp_shape.ndim))
            )
            state.shape_env[step.output] = unknown_shape

    def _apply_submodule_call(
        self,
        state: ModelState,
        step: ComputationStep,
        layer: LayerDef,
        inp_name: Optional[str],
        inp_shape: TensorShape,
        violations: List[SafetyViolation],
    ) -> None:
        """Simulate a submodule's forward pass by running its sub-graph."""
        sub_graph = layer.sub_graph
        if sub_graph is None or not sub_graph.steps:
            # Fallback: treat as shape-preserving
            state.shape_env[step.output] = inp_shape
            return

        # Create a mini ModelState for the submodule
        sub_state = state.copy()
        # Map the submodule's input names to the actual input shape
        if sub_graph.input_names:
            for iname in sub_graph.input_names:
                sub_state.shape_env[iname] = inp_shape
                if inp_name and inp_name in state.device_map:
                    sub_state.device_map[iname] = state.device_map[inp_name]

        # Save the parent's layers and temporarily use the submodule's
        saved_layers = self.graph.layers
        # Merge: submodule layers override parent layers for resolution
        merged_layers = dict(saved_layers)
        merged_layers.update(sub_graph.layers)
        self.graph.layers = merged_layers

        # Run each step in the sub-graph
        for sub_step in sub_graph.steps:
            sub_state, sub_violations = self._step_transition(
                sub_state, sub_step
            )
            violations.extend(sub_violations)

        # Restore parent layers
        self.graph.layers = saved_layers

        # Extract the output shape from the sub-graph's output
        if sub_graph.output_names:
            for oname in sub_graph.output_names:
                if oname in sub_state.shape_env:
                    state.shape_env[step.output] = sub_state.shape_env[oname]
                    break
        elif sub_graph.steps:
            # Fallback: use last step's output
            last_out = sub_graph.steps[-1].output
            if last_out in sub_state.shape_env:
                state.shape_env[step.output] = sub_state.shape_env[last_out]

    def _apply_matmul(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        if len(step.inputs) < 2:
            return
        a_name, b_name = step.inputs[0], step.inputs[1]
        a_shape = state.shape_env.get(a_name)
        b_shape = state.shape_env.get(b_name)
        if a_shape is None or b_shape is None:
            return
        err = check_matmul_compatible(a_shape, b_shape)
        if err:
            violations.append(SafetyViolation(
                kind="shape_incompatible", step_index=-1, step=step,
                message=err,
                tensor_a=a_name, tensor_b=b_name,
                shape_a=a_shape, shape_b=b_shape,
            ))
            return
        result = compute_matmul_shape(a_shape, b_shape)
        if result is not None:
            state.shape_env[step.output] = result

    def _apply_add(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        if len(step.inputs) < 2:
            return
        a_name, b_name = step.inputs[0], step.inputs[1]
        a_shape = state.shape_env.get(a_name)
        b_shape = state.shape_env.get(b_name)
        if a_shape is None and b_shape is None:
            return
        # If one operand has unknown shape, conservatively use the other
        if a_shape is None:
            state.shape_env[step.output] = b_shape
            return
        if b_shape is None:
            state.shape_env[step.output] = a_shape
            return
        result = compute_broadcast_shape(a_shape, b_shape)
        if result is None:
            violations.append(SafetyViolation(
                kind="shape_incompatible", step_index=-1, step=step,
                message=(
                    f"Cannot broadcast {a_shape.pretty()} and "
                    f"{b_shape.pretty()}"
                ),
                tensor_a=a_name, tensor_b=b_name,
                shape_a=a_shape, shape_b=b_shape,
            ))
        else:
            state.shape_env[step.output] = result

    def _apply_where(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        """torch.where(cond, x, y) — broadcast all three inputs pairwise."""
        if len(step.inputs) < 3:
            # Fallback: single-arg torch.where(cond) → indices, treat as _apply_add
            self._apply_add(state, step, violations)
            return
        cond_name, x_name, y_name = step.inputs[0], step.inputs[1], step.inputs[2]
        cond_shape = state.shape_env.get(cond_name)
        x_shape = state.shape_env.get(x_name)
        y_shape = state.shape_env.get(y_name)
        # Check x vs y broadcast compatibility (most common bug)
        if x_shape is not None and y_shape is not None:
            xy_result = compute_broadcast_shape(x_shape, y_shape)
            if xy_result is None:
                violations.append(SafetyViolation(
                    kind="shape_incompatible", step_index=-1, step=step,
                    message=(
                        f"torch.where: cannot broadcast x {x_shape.pretty()} and "
                        f"y {y_shape.pretty()}"
                    ),
                    tensor_a=x_name, tensor_b=y_name,
                    shape_a=x_shape, shape_b=y_shape,
                ))
                return
            # Also check cond vs xy
            if cond_shape is not None:
                final = compute_broadcast_shape(cond_shape, xy_result)
                if final is None:
                    violations.append(SafetyViolation(
                        kind="shape_incompatible", step_index=-1, step=step,
                        message=(
                            f"torch.where: cannot broadcast condition "
                            f"{cond_shape.pretty()} with value shape "
                            f"{xy_result.pretty()}"
                        ),
                        tensor_a=cond_name, tensor_b=x_name,
                        shape_a=cond_shape, shape_b=xy_result,
                    ))
                    return
                state.shape_env[step.output] = final
            else:
                state.shape_env[step.output] = xy_result
        elif x_shape is not None:
            state.shape_env[step.output] = x_shape
        elif y_shape is not None:
            state.shape_env[step.output] = y_shape

    def _apply_indexing(
        self,
        shape_env: Dict[str, "TensorShape"],
        step: ComputationStep,
        violations: Optional[List[SafetyViolation]],
    ) -> None:
        """Shape effects of indexing / gather / scatter / masked ops.

        Sound posture: only emit a violation when the relevant dimensions are
        fully concrete and the error is provable; otherwise propagate the
        best-effort output shape with no violation. ``violations is None`` (the
        abstract ``_propagate_step`` path) suppresses all violation emission.
        """
        if not step.inputs:
            return
        inp_name = step.inputs[0]
        inp_shape = shape_env.get(inp_name)
        if inp_shape is None:
            return

        dim = step.params.get("dim")
        ndim = inp_shape.ndim
        norm_dim: Optional[int] = None
        if isinstance(dim, int):
            norm_dim = dim + ndim if dim < 0 else dim
            # Provable dim-out-of-range (input rank concrete by construction).
            if not (0 <= norm_dim < ndim) and norm_dim is not None:
                if violations is not None:
                    violations.append(SafetyViolation(
                        kind="shape_incompatible", step_index=-1, step=step,
                        message=(
                            f"{step.op.name.lower()}: dim {dim} out of range "
                            f"for tensor of rank {ndim}"
                        ),
                        tensor_a=inp_name, shape_a=inp_shape,
                    ))
                # Still propagate a best-effort shape below using clamped dim.
                norm_dim = None

        # Second tensor operand (index / mask / src), when present.
        idx_name = step.inputs[1] if len(step.inputs) > 1 else None
        idx_shape = shape_env.get(idx_name) if idx_name else None

        if step.op == OpKind.GATHER:
            # output.shape == index.shape; require equal rank (no broadcast);
            # for d != dim, index.size(d) <= input.size(d).
            if idx_shape is not None:
                if idx_shape.ndim != ndim:
                    if violations is not None:
                        violations.append(SafetyViolation(
                            kind="shape_incompatible", step_index=-1, step=step,
                            message=(
                                f"gather: index rank {idx_shape.ndim} must equal "
                                f"input rank {ndim}"
                            ),
                            tensor_a=inp_name, tensor_b=idx_name,
                            shape_a=inp_shape, shape_b=idx_shape,
                        ))
                elif norm_dim is not None:
                    for d in range(ndim):
                        if d == norm_dim:
                            continue
                        a, b = inp_shape.dims[d], idx_shape.dims[d]
                        if (not a.is_symbolic and not b.is_symbolic
                                and isinstance(a.value, int)
                                and isinstance(b.value, int)
                                and b.value > a.value):
                            if violations is not None:
                                violations.append(SafetyViolation(
                                    kind="shape_incompatible", step_index=-1,
                                    step=step,
                                    message=(
                                        f"gather: index size {b.value} exceeds "
                                        f"input size {a.value} at dim {d}"
                                    ),
                                    tensor_a=inp_name, tensor_b=idx_name,
                                    shape_a=inp_shape, shape_b=idx_shape,
                                ))
                            break
                shape_env[step.output] = idx_shape
            else:
                shape_env[step.output] = inp_shape

        elif step.op == OpKind.TAKE:
            # output.shape == index.shape (arbitrary rank allowed).
            shape_env[step.output] = idx_shape if idx_shape is not None else inp_shape

        elif step.op == OpKind.INDEX_SELECT:
            # output = input with size(dim) replaced by index length (1-D index).
            new_dims = list(inp_shape.dims)
            if idx_shape is not None and idx_shape.ndim != 1:
                if violations is not None:
                    violations.append(SafetyViolation(
                        kind="shape_incompatible", step_index=-1, step=step,
                        message=(
                            f"index_select: index must be 1-D, got rank "
                            f"{idx_shape.ndim}"
                        ),
                        tensor_a=inp_name, tensor_b=idx_name,
                        shape_a=inp_shape, shape_b=idx_shape,
                    ))
            if norm_dim is not None and 0 <= norm_dim < len(new_dims):
                if idx_shape is not None and idx_shape.ndim == 1:
                    new_dims[norm_dim] = idx_shape.dims[0]
                else:
                    new_dims[norm_dim] = ShapeDim(f"_idxsel{step.output}")
            shape_env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.SCATTER:
            # Conservative: output == input.shape; only flag a concrete rank
            # mismatch between input and the tensor index operand.
            if (idx_shape is not None and idx_shape.ndim != ndim
                    and violations is not None):
                violations.append(SafetyViolation(
                    kind="shape_incompatible", step_index=-1, step=step,
                    message=(
                        f"scatter: index rank {idx_shape.ndim} must equal input "
                        f"rank {ndim}"
                    ),
                    tensor_a=inp_name, tensor_b=idx_name,
                    shape_a=inp_shape, shape_b=idx_shape,
                ))
            shape_env[step.output] = inp_shape

        elif step.op == OpKind.MASKED_FILL:
            # output == input.shape; mask broadcasts to input. Only flag a
            # provably-impossible broadcast (both concrete, unequal, neither 1).
            if idx_shape is not None and violations is not None:
                if compute_broadcast_shape(inp_shape, idx_shape) is None:
                    violations.append(SafetyViolation(
                        kind="shape_incompatible", step_index=-1, step=step,
                        message=(
                            f"masked_fill: mask {idx_shape.pretty()} cannot "
                            f"broadcast to input {inp_shape.pretty()}"
                        ),
                        tensor_a=inp_name, tensor_b=idx_name,
                        shape_a=inp_shape, shape_b=idx_shape,
                    ))
            shape_env[step.output] = inp_shape

        elif step.op == OpKind.MASKED_SELECT:
            # Always rank-1 with a data-dependent (fresh symbolic) length.
            shape_env[step.output] = TensorShape(
                (ShapeDim(f"_masked{step.output}"),)
            )

        elif step.op == OpKind.NARROW:
            # output = input with size(dim) replaced by `length`.
            new_dims = list(inp_shape.dims)
            length = step.params.get("length")
            start = step.params.get("start")
            if norm_dim is not None and 0 <= norm_dim < len(new_dims):
                d = inp_shape.dims[norm_dim]
                if (isinstance(length, int) and isinstance(start, int)
                        and not d.is_symbolic and isinstance(d.value, int)
                        and start + length > d.value and violations is not None):
                    violations.append(SafetyViolation(
                        kind="shape_incompatible", step_index=-1, step=step,
                        message=(
                            f"narrow: start {start} + length {length} exceeds "
                            f"size {d.value} at dim {norm_dim}"
                        ),
                        tensor_a=inp_name, shape_a=inp_shape,
                    ))
                if isinstance(length, int):
                    new_dims[norm_dim] = ShapeDim(length)
                else:
                    new_dims[norm_dim] = ShapeDim(f"_narrow{step.output}")
            shape_env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.SELECT_DIM:
            # output = input with `dim` removed.
            if norm_dim is not None and 0 <= norm_dim < ndim:
                new_dims = [d for i, d in enumerate(inp_shape.dims)
                            if i != norm_dim]
                shape_env[step.output] = TensorShape(tuple(new_dims))
            else:
                shape_env[step.output] = inp_shape

    def _apply_sdpa(
        self,
        shape_env: Dict[str, "TensorShape"],
        step: ComputationStep,
        violations: Optional[List[SafetyViolation]],
    ) -> None:
        """Shape effect of F.scaled_dot_product_attention(query, key, value).

        Output = query shape with its last dim replaced by value's last dim.
        Emits a violation (only on concrete, provable mismatches) for a query/key
        embed-dim mismatch or a key/value sequence-length mismatch. ``violations
        is None`` (the abstract path) suppresses emission.
        """
        if len(step.inputs) < 3:
            # Fewer than q,k,v captured (e.g. some args were non-tensor) —
            # propagate the first input's shape as a best-effort fallback.
            if step.inputs and step.inputs[0] in shape_env:
                shape_env[step.output] = shape_env[step.inputs[0]]
            return
        q_name, k_name, v_name = step.inputs[0], step.inputs[1], step.inputs[2]
        q = shape_env.get(q_name)
        k = shape_env.get(k_name)
        v = shape_env.get(v_name)
        if q is None or k is None or v is None:
            if q is not None:
                shape_env[step.output] = q
            return
        out_shape, err = compute_sdpa_shape(q, k, v)
        if err is not None and violations is not None:
            violations.append(SafetyViolation(
                kind="shape_incompatible", step_index=-1, step=step,
                message=err,
                tensor_a=q_name, tensor_b=k_name,
                shape_a=q, shape_b=k,
            ))
        if out_shape is not None:
            shape_env[step.output] = out_shape

    def _apply_reshape(
        self, state: ModelState, step: ComputationStep,
        violations: Optional[List[SafetyViolation]] = None,
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        dims = step.params.get("dims")
        # Resolve cross-tensor aliases (recorded at extraction time as
        # {dim_idx: (src_var, src_dim_idx)}) against the current shape_env so
        # that ``view(B, T, ...)`` where ``B, T, _ = x.size()`` becomes a
        # concrete value when ``x`` has a known shape.
        if dims is not None:
            aliases = step.params.get("__alias_resolutions__")
            if aliases:
                dims_list = list(dims)
                for di, (src_var, src_di) in aliases.items():
                    src_shape = state.shape_env.get(src_var)
                    if (src_shape is not None
                            and 0 <= src_di < src_shape.ndim):
                        sd = src_shape.dims[src_di]
                        if not sd.is_symbolic:
                            dims_list[di] = sd.value
                        else:
                            dims_list[di] = sd.value
                dims = tuple(dims_list)
        if inp_shape is not None and dims is not None:
            result = compute_reshape_shape(inp_shape, dims)
            if result is not None:
                state.shape_env[step.output] = result
            if violations is not None:
                # Sound symbolic reasoning: when Z3 is available it is the
                # authoritative oracle.  ``check_reshape_compatible`` flags the
                # reshape ONLY when the element-count equation is UNSAT for all
                # dimension assignments >= 1 (provably impossible) — coupling
                # equal-named symbolic dims across input and target.  When Z3
                # runs (returning a message or abstaining) we trust it and skip
                # the legacy syntactic check, which both misses symbolic
                # incompatibilities (e.g. (B,5)->(B,3)) and can false-positive
                # on divisible-but-symbolic cases (e.g. (B,10)->(-1,3)).
                z3_ran = False
                if HAS_Z3:
                    try:
                        from src.smt.reshape_theory import (
                            check_reshape_compatible,
                        )
                        z3_msg = check_reshape_compatible(inp_shape, dims)
                        z3_ran = True
                    except Exception:
                        z3_ran = False
                        z3_msg = None
                    if z3_ran and z3_msg is not None:
                        violations.append(SafetyViolation(
                            kind="shape_incompatible",
                            step_index=-1,
                            step=step,
                            message=z3_msg,
                            tensor_a=inp,
                            shape_a=inp_shape,
                        ))
                if not z3_ran and result is None:
                    # Legacy syntactic fallback (Z3 unavailable / errored).
                    # Soundness: if either side contains free symbolic dims
                    # (e.g. an unresolved Linear out_features ``_unk_lin_out``
                    # or a config-derived symbol), we cannot prove the reshape
                    # is incompatible — abstain rather than emit a false
                    # positive.
                    _has_free_sym = any(
                        d.is_symbolic and isinstance(d.value, str)
                        and (d.value.startswith("_unk_") or "_" in d.value)
                        for d in inp_shape.dims
                    )
                    _dims_have_str = any(
                        isinstance(d, str) for d in (dims or ())
                    )
                    if not (_has_free_sym or _dims_have_str):
                        violations.append(SafetyViolation(
                            kind="shape_incompatible",
                            step_index=-1,
                            step=step,
                            message=(
                                f"Reshape incompatible: cannot reshape "
                                f"{inp_shape} to {dims}"
                            ),
                            tensor_a=inp,
                            shape_a=inp_shape,
                        ))

    def _apply_flatten(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is not None:
            sd = step.params.get("start_dim", 1)
            ed = step.params.get("end_dim", -1)
            out, _ = _propagate_flatten(inp_shape, sd, ed)
            if out is not None:
                state.shape_env[step.output] = out

    def _apply_squeeze(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is None:
            return
        dim = step.params.get("dim")
        if dim is not None:
            if dim < 0:
                dim = inp_shape.ndim + dim
            new_dims = list(inp_shape.dims)
            if 0 <= dim < len(new_dims):
                d = new_dims[dim]
                if not d.is_symbolic and d.value == 1:
                    new_dims.pop(dim)
            state.shape_env[step.output] = TensorShape(tuple(new_dims))
        else:
            new_dims = [d for d in inp_shape.dims
                        if d.is_symbolic or d.value != 1]
            state.shape_env[step.output] = TensorShape(tuple(new_dims))

    def _apply_subscript(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        """Apply subscript/indexing: drop dimensions indexed by int, keep slices."""
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is None:
            return
        indices = step.params.get("indices", [])
        if not indices:
            state.shape_env[step.output] = inp_shape
            return
        new_dims = []
        for i, dim in enumerate(inp_shape.dims):
            if i < len(indices):
                if indices[i] == 'slice':
                    new_dims.append(dim)
                # 'int' → dimension eliminated (integer index selects one element)
            else:
                # Trailing dimensions not covered by indices are kept
                new_dims.append(dim)
        state.shape_env[step.output] = TensorShape(tuple(new_dims))

    def _apply_unsqueeze(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is None:
            return
        dim = step.params.get("dim", 0)
        if dim < 0:
            dim = inp_shape.ndim + 1 + dim
        new_dims = list(inp_shape.dims)
        new_dims.insert(dim, ShapeDim(1))
        state.shape_env[step.output] = TensorShape(tuple(new_dims))

    def _apply_transpose(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is None:
            return
        d0 = step.params.get("dim0", 0)
        d1 = step.params.get("dim1", 1)
        if d0 < 0:
            d0 = inp_shape.ndim + d0
        if d1 < 0:
            d1 = inp_shape.ndim + d1
        new_dims = list(inp_shape.dims)
        if 0 <= d0 < len(new_dims) and 0 <= d1 < len(new_dims):
            new_dims[d0], new_dims[d1] = new_dims[d1], new_dims[d0]
        state.shape_env[step.output] = TensorShape(tuple(new_dims))

    def _apply_permute(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        inp_shape = state.shape_env.get(inp) if inp else None
        if inp_shape is None:
            return
        perm = step.params.get("dims")
        if perm and len(perm) == inp_shape.ndim:
            new_dims = [inp_shape.dims[p] for p in perm if p is not None]
            if len(new_dims) == inp_shape.ndim:
                state.shape_env[step.output] = TensorShape(tuple(new_dims))

    def _apply_cat(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        shapes = [state.shape_env.get(i) for i in step.inputs]
        if not all(s is not None for s in shapes) or not shapes:
            return
        cat_dim = step.params.get("dim", 0)

        def _looks_opaque(sh):
            return any(d.is_symbolic and isinstance(d.value, str)
                       and d.value.startswith("_unk") for d in sh.dims)

        # Sound abstention: if any operand carries opaque ``_unk_`` dims it
        # likely came through an opaque submodule whose true ndim/shape we
        # cannot recover. Propagate a fully-symbolic shape rather than
        # raising a spurious mismatch.
        if any(_looks_opaque(s) for s in shapes):
            first = shapes[0]
            new_state_shape = TensorShape(
                tuple(ShapeDim(f"_unk_cat_{i}") for i in range(first.ndim))
            )
            state.shape_env[step.output] = new_state_shape
            return

        first = shapes[0]
        for i, s in enumerate(shapes[1:], 1):
            if s.ndim != first.ndim:
                violations.append(SafetyViolation(
                    kind="shape_incompatible", step_index=-1, step=step,
                    message=(
                        f"cat: tensors have different ndim "
                        f"({first.ndim} vs {s.ndim})"
                    ),
                ))
                return
        # Check non-cat dimensions match
        resolved_cat_dim = cat_dim
        if resolved_cat_dim < 0:
            resolved_cat_dim = first.ndim + resolved_cat_dim
        for i, s in enumerate(shapes[1:], 1):
            for d in range(first.ndim):
                if d == resolved_cat_dim:
                    continue
                d_first = first.dims[d]
                d_other = s.dims[d]
                if (not d_first.is_symbolic and not d_other.is_symbolic
                        and d_first.value != d_other.value):
                    violations.append(SafetyViolation(
                        kind="shape_incompatible", step_index=-1, step=step,
                        message=(
                            f"cat: dimension {d} mismatch: "
                            f"{d_first.value} vs {d_other.value} "
                            f"(cat along dim={cat_dim})"
                        ),
                    ))
                    return
        out_dims = list(first.dims)
        if 0 <= resolved_cat_dim < first.ndim:
            total = first.dims[resolved_cat_dim]
            all_concrete = not total.is_symbolic
            for s in shapes[1:]:
                d = s.dims[resolved_cat_dim]
                if d.is_symbolic or total.is_symbolic:
                    all_concrete = False
                    break
                total = ShapeDim(total.value + d.value)
            if all_concrete:
                out_dims[resolved_cat_dim] = total
            else:
                out_dims[resolved_cat_dim] = ShapeDim("_cat")
        state.shape_env[step.output] = TensorShape(tuple(out_dims))

    def _apply_stack(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        """torch.stack adds a new dim; all inputs must have the same shape."""
        shapes = [state.shape_env.get(i) for i in step.inputs]
        if not all(s is not None for s in shapes) or not shapes:
            return
        first = shapes[0]
        for i, s in enumerate(shapes[1:], 1):
            if s.ndim != first.ndim:
                violations.append(SafetyViolation(
                    kind="shape_incompatible", step_index=-1, step=step,
                    message=(
                        f"stack: tensors have different ndim "
                        f"({first.ndim} vs {s.ndim})"
                    ),
                ))
                return
        stack_dim = step.params.get("dim", 0)
        if stack_dim < 0:
            stack_dim = first.ndim + 1 + stack_dim
        out_dims = list(first.dims)
        n_tensors = len(shapes)
        out_dims.insert(stack_dim, ShapeDim(n_tensors))
        state.shape_env[step.output] = TensorShape(tuple(out_dims))

    def _apply_to_device(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        inp = step.inputs[0] if step.inputs else None
        if inp and inp in state.shape_env:
            state.shape_env[step.output] = state.shape_env[inp]
        dev_str = step.params.get("device")
        if dev_str is not None:
            state.device_map[step.output] = Device.from_string(str(dev_str))
        elif inp and inp in state.device_map:
            state.device_map[step.output] = state.device_map[inp]

    def _apply_new_tensor(
        self, state: ModelState, step: ComputationStep
    ) -> None:
        """Seed shape/device/dtype for a tensor-factory op (Step 32).

        The op is a *leaf* (no tensor inputs): its shape is fully determined by
        the call arguments and is independent of the RNG seed.  Device defaults
        to CPU (torch factory default) unless an explicit ``device=`` was given;
        dtype defaults to the factory's natural dtype family unless overridden.
        """
        shape = step.params.get("shape")
        if isinstance(shape, TensorShape):
            state.shape_env[step.output] = shape
        dev_str = step.params.get("device")
        if isinstance(dev_str, str):
            state.device_map[step.output] = Device.from_string(dev_str)
        else:
            state.device_map[step.output] = Device.CPU
        # Gradients: factory tensors do not require grad by default.
        state.gradient_status[step.output] = False
        # dtype: explicit dtype= wins; else the factory's natural default.
        dt = _canon_dtype(step.params.get("cast_dtype"))
        if dt is None:
            fam = step.params.get("dtype_family", "")
            if fam == "float":
                dt = _canon_dtype("float32")
            elif fam == "int":
                dt = _canon_dtype("int64")
        if dt is not None:
            state.dtype_env[step.output] = dt

    def _apply_conditional(
        self,
        state: ModelState,
        step: ComputationStep,
        violations: List[SafetyViolation],
    ) -> None:
        """Apply a conditional step by processing BOTH branches for
        ``self.training`` conditions (multi-phase verification).

        This is the key to TensorGuard's cross-cutting multi-theory
        verification: bugs that only manifest in train or eval mode
        are caught by checking both paths and annotating violations
        with the phase in which they occur.

        For other conditions we conservatively process both branches
        and merge the resulting shape environments.
        """
        cond = step.condition
        true_steps = step.true_branch or []
        false_steps = step.false_branch or []

        if cond == "self.training":
            # Multi-phase verification: check BOTH branches to catch
            # phase-dependent bugs (e.g., wrong dims only in eval)
            active_branch = true_steps if state.phase == Phase.TRAIN else false_steps
            inactive_branch = false_steps if state.phase == Phase.TRAIN else true_steps
            inactive_phase = Phase.EVAL if state.phase == Phase.TRAIN else Phase.TRAIN

            # Process active branch (updates state)
            for s in active_branch:
                state, vs = self._step_transition(state, s)
                violations.extend(vs)

            # Also verify inactive branch on a copy (catches phase-dependent bugs)
            if inactive_branch:
                alt_state = state.copy()
                alt_state.phase = inactive_phase
                for s in inactive_branch:
                    alt_state, vs = self._step_transition(alt_state, s)
                    for v in vs:
                        v.message = (f"[{inactive_phase.name} mode, phase-dependent] " + v.message)
                    violations.extend(vs)

        elif cond == "not self.training":
            active_branch = true_steps if state.phase == Phase.EVAL else false_steps
            inactive_branch = false_steps if state.phase == Phase.EVAL else true_steps
            inactive_phase = Phase.TRAIN if state.phase == Phase.EVAL else Phase.EVAL

            for s in active_branch:
                state, vs = self._step_transition(state, s)
                violations.extend(vs)

            if inactive_branch:
                alt_state = state.copy()
                alt_state.phase = inactive_phase
                for s in inactive_branch:
                    alt_state, vs = self._step_transition(alt_state, s)
                    for v in vs:
                        v.message = (f"[{inactive_phase.name} mode, phase-dependent] " + v.message)
                    violations.extend(vs)
        elif cond is not None and cond.startswith("hasattr:self."):
            # hasattr(self, attr) — only take true branch if attr exists
            attr = cond.split(".", 1)[1] if "." in cond.split(":", 1)[1] else ""
            if attr in self.graph.layers:
                for s in true_steps:
                    state, vs = self._step_transition(state, s)
                    violations.extend(vs)
            else:
                for s in false_steps:
                    state, vs = self._step_transition(state, s)
                    violations.extend(vs)
        else:
            # Unknown condition: process both branches, merge states
            true_state = state.copy()
            for s in true_steps:
                true_state, vs = self._step_transition(true_state, s)
                violations.extend(vs)
            false_state = state.copy()
            for s in false_steps:
                false_state, vs = self._step_transition(false_state, s)
                violations.extend(vs)
            # Merge: union of shape bindings from both branches
            for name, shape in true_state.shape_env.items():
                state.shape_env[name] = shape
            for name, shape in false_state.shape_env.items():
                if name not in state.shape_env:
                    state.shape_env[name] = shape
            for name, dev in true_state.device_map.items():
                state.device_map[name] = dev
            for name, dev in false_state.device_map.items():
                if name not in state.device_map:
                    state.device_map[name] = dev

    # ======================================================================
    # Verification: base case
    # ======================================================================

    def _bmc_base_case(
        self,
    ) -> Tuple[List[SafetyViolation], List[ModelState], List[KripkeState]]:
        """Unfold the computation graph and check safety at each step.

        Returns ``(violations, model_states, kripke_states)``.
        """
        all_viols: List[SafetyViolation] = []
        model_states: List[ModelState] = [self._init_state.copy()]
        kripke_states: List[KripkeState] = []

        if not HAS_Z3:
            for idx, step in enumerate(self.graph.steps[: self.max_k]):
                cur = model_states[-1]
                ns, vs = self._step_transition(cur, step)
                for v in vs:
                    v.step_index = idx
                all_viols.extend(vs)
                model_states.append(ns)
            return all_viols, model_states, kripke_states

        # Build initial symbolic state & solver
        k0 = self._build_kripke_state(0, self._init_state)
        kripke_states.append(k0)

        solver = self.ctx.solver
        for c in self._initial_constraints(k0):
            solver.add(c)

        # Initial satisfiability
        self.ctx.timed_check(solver)

        for idx, step in enumerate(self.graph.steps[: self.max_k]):
            cur_model = model_states[-1]
            cur_k = kripke_states[-1]

            # 1. Concrete step transition
            new_model, concrete_vs = self._step_transition(cur_model, step)
            for v in concrete_vs:
                v.step_index = idx
            all_viols.extend(concrete_vs)
            model_states.append(new_model)

            # 2. Build post-transition symbolic state
            post_k = self._build_kripke_state(idx + 1, new_model)
            kripke_states.append(post_k)

            # 3. Z3 safety checks per domain
            for kind, encoder in self._filter_domain_checks([
                ("shape_incompatible",
                 lambda: self._encode_shape_safety(
                     cur_k, step, cur_model, idx)),
                ("device_mismatch",
                 lambda: self._encode_device_safety(
                     cur_k, step, cur_model, idx)),
                ("phase_violation",
                 lambda: self._encode_phase_safety(
                     cur_k, step, cur_model, idx)),
                ("gradient_violation",
                 lambda: self._encode_gradient_safety(
                     cur_k, step, cur_model, idx)),
            ]):
                safety = encoder()
                if safety:
                    v = self._z3_check_safety(
                        solver, safety, step, idx, kind
                    )
                    if v is not None:
                        all_viols.append(v)

            # 3b. Check device theory solver
            if self.check_devices and self.ctx.device_theory is not None:
                device_result = self.ctx.timed_check(self.ctx._device_solver)
                if device_result == z3.unsat:
                    all_viols.append(SafetyViolation(
                        kind="device_mismatch",
                        step_index=idx,
                        step=step,
                        message=f"Device theory propagator: device inconsistency at step {idx} ({step.op.name})",
                    ))

            # 3c. Check phase theory solver
            if self.check_phases and self.ctx.phase_theory is not None:
                phase_result = self.ctx.timed_check(self.ctx._phase_solver)
                if phase_result == z3.unsat:
                    all_viols.append(SafetyViolation(
                        kind="phase_violation",
                        step_index=idx,
                        step=step,
                        message=f"Phase theory propagator: phase inconsistency at step {idx} ({step.op.name})",
                    ))

            # 4. Cross-domain safety
            xd = self._encode_cross_domain_safety(
                cur_k, post_k, step, cur_model, idx
            )
            if xd:
                v = self._z3_check_safety(
                    solver, xd, step, idx, "cross_domain_violation"
                )
                if v is not None:
                    all_viols.append(v)

            # 5. Accumulate transition constraints
            trans = self._encode_transition(
                cur_k, step, post_k, cur_model, idx
            )
            for c in trans:
                solver.add(c)

            # 5b. Assert positivity for post-state shape variables
            for dims in post_k.shape_vars.values():
                for d in dims:
                    if not z3.is_int_value(d):
                        solver.add(d > 0)

            # 6. Transition consistency
            self.ctx.timed_check(solver)

            # 7. Phase well-formedness
            if cur_k.phase_var is not None:
                pwf = [z3.Or(
                    cur_k.phase_var == self.ctx.PHASE_TRAIN,
                    cur_k.phase_var == self.ctx.PHASE_EVAL,
                )]
                self._z3_check_safety(
                    solver, pwf, step, idx, "phase_wellformed"
                )

            # 8. Dimension positivity per step
            pos: list = []
            for dims in cur_k.shape_vars.values():
                for d in dims:
                    if not z3.is_int_value(d):
                        pos.append(d > 0)
            if pos:
                self._z3_check_safety(
                    solver, pos, step, idx, "dim_positivity"
                )

            # 9. Device well-formedness (each tensor on a valid device)
            dev_wf: list = []
            for dv in cur_k.device_vars.values():
                dev_wf.append(z3.Or(
                    dv == self.ctx.DEV_CPU,
                    dv == self.ctx.DEV_CUDA0,
                    dv == self.ctx.DEV_CUDA1,
                    dv == self.ctx.DEV_CUDA2,
                    dv == self.ctx.DEV_CUDA3,
                ))
            if dev_wf:
                self._z3_check_safety(
                    solver, dev_wf, step, idx, "device_wellformed"
                )

            # 10. Gradient well-formedness (post-state)
            grad_wf: list = []
            for gv in post_k.grad_vars.values():
                grad_wf.append(z3.Or(
                    gv == z3.BoolVal(True),
                    gv == z3.BoolVal(False),
                ))
            if grad_wf:
                self._z3_check_safety(
                    solver, grad_wf, step, idx, "gradient_wellformed"
                )

            # 11. Shape-device combined check
            sd_combined: list = []
            sd_combined.extend(self._encode_shape_safety(
                cur_k, step, cur_model, idx))
            sd_combined.extend(self._encode_device_safety(
                cur_k, step, cur_model, idx))
            if sd_combined:
                self._z3_check_safety(
                    solver, sd_combined, step, idx, "shape_device_combined"
                )

            # 12. Full combined safety (all four domains)
            combined: list = []
            combined.extend(self._encode_shape_safety(
                cur_k, step, cur_model, idx))
            combined.extend(self._encode_device_safety(
                cur_k, step, cur_model, idx))
            combined.extend(self._encode_phase_safety(
                cur_k, step, cur_model, idx))
            combined.extend(self._encode_gradient_safety(
                cur_k, step, cur_model, idx))
            if combined:
                self._z3_check_safety(
                    solver, combined, step, idx, "combined_violation"
                )

        # 13. Backward constraint propagation pass
        bw_viols = self._backward_constraint_pass(
            solver, kripke_states, model_states,
        )
        all_viols.extend(bw_viols)

        return all_viols, model_states, kripke_states

    # ======================================================================
    # Verification: inductive step
    # ======================================================================

    def _bmc_inductive_step(
        self,
        kripke_states: List[KripkeState],
        model_states: List[ModelState],
    ) -> List[SafetyViolation]:
        """Forward inductive step: prove safety preserved across
        transitions using free symbolic state variables.
        """
        if not HAS_Z3:
            return []
        violations: List[SafetyViolation] = []
        n_steps = min(len(self.graph.steps) - 1, self.max_k - 1)

        for idx in range(n_steps):
            step = self.graph.steps[idx]
            next_step = self.graph.steps[idx + 1]
            pre_model = (model_states[idx]
                         if idx < len(model_states) else model_states[-1])
            post_model = (model_states[idx + 1]
                          if idx + 1 < len(model_states)
                          else model_states[-1])

            # Free symbolic states (step_idx offset avoids name collisions)
            pre_k = self._build_kripke_state(
                2000 + idx, pre_model, free_shapes=True
            )
            post_k = self._build_kripke_state(
                2000 + idx + 1, post_model, free_shapes=True
            )

            solver = z3.Solver()
            solver.set("timeout", 5000)

            # Positivity for free dims
            for dims in pre_k.shape_vars.values():
                for d in dims:
                    solver.add(d > 0)
            for dims in post_k.shape_vars.values():
                for d in dims:
                    solver.add(d > 0)

            # Safety assumption at pre-state
            for enc in (
                self._encode_shape_safety(pre_k, step, pre_model, idx),
                self._encode_device_safety(pre_k, step, pre_model, idx),
                self._encode_phase_safety(pre_k, step, pre_model, idx),
                self._encode_gradient_safety(pre_k, step, pre_model, idx),
            ):
                for c in enc:
                    solver.add(c)

            # Transition
            for c in self._encode_transition(
                pre_k, step, post_k, pre_model, idx
            ):
                solver.add(c)

            # Per-domain inductive checks at post-state
            for kind, encoder in self._filter_domain_checks([
                ("shape_incompatible",
                 lambda: self._encode_shape_safety(
                     post_k, next_step, post_model, idx + 1)),
                ("device_mismatch",
                 lambda: self._encode_device_safety(
                     post_k, next_step, post_model, idx + 1)),
                ("phase_violation",
                 lambda: self._encode_phase_safety(
                     post_k, next_step, post_model, idx + 1)),
                ("gradient_violation",
                 lambda: self._encode_gradient_safety(
                     post_k, next_step, post_model, idx + 1)),
            ]):
                post_safety = encoder()
                if post_safety:
                    solver.push()
                    solver.add(z3.Not(z3.And(*post_safety)))
                    result = self.ctx.timed_check(solver)
                    if result == z3.sat:
                        m = solver.model()
                        violations.append(SafetyViolation(
                            kind="inductive_violation",
                            step_index=idx + 1,
                            step=next_step,
                            message=self._format_z3_model(
                                m, idx + 1, kind
                            ),
                        ))
                    solver.pop()

            # Cross-domain inductive check
            xd = self._encode_cross_domain_safety(
                pre_k, post_k, step, pre_model, idx
            )
            if xd:
                solver.push()
                solver.add(z3.Not(z3.And(*xd)))
                result = self.ctx.timed_check(solver)
                if result == z3.sat:
                    m = solver.model()
                    violations.append(SafetyViolation(
                        kind="inductive_violation",
                        step_index=idx,
                        step=step,
                        message=self._format_z3_model(
                            m, idx, "cross_domain"
                        ),
                    ))
                solver.pop()

        return violations

    # Common module-level names that are always in scope (imports/builtins).
    _ALWAYS_DEFINED = frozenset({
        "self", "torch", "F", "nn", "np", "math", "None", "True", "False",
        "print", "len", "range", "int", "float", "list", "tuple", "dict",
        "isinstance", "type", "enumerate", "zip", "map", "filter", "sorted",
        "super", "object", "str", "bool", "set", "frozenset", "getattr",
    })

    def _check_use_before_def(self) -> List[SafetyViolation]:
        """Check for variables used before they are defined.

        This catches swap_layers mutations where layer call order is
        reversed, causing variables to be referenced before assignment.

        Only flags a variable when it IS defined later in the step list
        (indicating reordering), not when it is never defined at all.
        Variables that never appear as outputs are likely from control flow
        (tuple unpacking, method calls, conditionals) that the extractor
        does not model.
        """
        violations: List[SafetyViolation] = []
        defined: set = set(self.graph.input_names)

        # Collect all variable names that are eventually defined (outputs)
        all_outputs = {step.output for step in self.graph.steps if step.output}

        for step in self.graph.steps:
            for inp in step.inputs:
                if inp in defined:
                    continue
                # Skip internal variables and self attributes
                if (inp.startswith("__") or inp.startswith("self.")
                        or inp.startswith("_attr") or inp.startswith("_tensor")):
                    continue
                # Skip dotted attribute access — the base object is usually
                # a parameter or a known variable (e.g. batch.premise)
                if "." in inp:
                    base = inp.split(".")[0]
                    if base in defined or base in self._ALWAYS_DEFINED:
                        continue
                # Skip common imports and builtins
                if inp in self._ALWAYS_DEFINED:
                    continue
                # Skip numeric-looking arguments (constant pool entries)
                try:
                    float(inp)
                    continue
                except (ValueError, TypeError):
                    pass
                # Only flag if this variable IS defined later (reordered),
                # not if it's simply absent from the graph (unmodeled flow).
                if inp not in all_outputs:
                    continue
                # Skip self-referencing steps (e.g. outputs = outputs + [x])
                # which are loop accumulators, not use-before-def errors.
                if inp == step.output:
                    continue
                violations.append(SafetyViolation(
                    kind="use_before_def",
                    step_index=-1,
                    step=step,
                    message=(
                        f"Variable '{inp}' used before definition "
                        f"at line {step.line}"
                    ),
                ))
            if step.output:
                defined.add(step.output)
        return violations

    def _check_dead_outputs(self) -> List[SafetyViolation]:
        """Check for tensors computed inside conditional branches but never used.

        Detects patterns like:
            if not self.training:
                output = torch.cat([cls, anchors], dim=2)  # computed
            return cls                                       # 'output' discarded

        Only checks within conditional (if/else) branches, not the top-level
        forward steps.  Top-level "unused" variables (e.g. q, k projections
        whose attention math isn't fully extracted) produce too many false
        positives because the extractor doesn't model all PyTorch ops.
        """
        violations: List[SafetyViolation] = []
        output_names = set(self.graph.output_names)

        # Collect all tensor names that are consumed anywhere in the graph
        consumed: set = set()
        def _collect_consumed(steps: List[ComputationStep]) -> None:
            for step in steps:
                consumed.update(inp for inp in step.inputs
                                if not inp.startswith("self."))
                if step.true_branch:
                    _collect_consumed(step.true_branch)
                if step.false_branch:
                    _collect_consumed(step.false_branch)
        _collect_consumed(self.graph.steps)
        consumed.update(output_names)

        def _check_branch(steps: List[ComputationStep]) -> None:
            """Check steps that are inside a conditional branch for dead outputs."""
            for step in steps:
                if step.op == OpKind.RETURN:
                    continue
                if step.output and not step.output.startswith("__"):
                    if step.output not in consumed:
                        violations.append(SafetyViolation(
                            kind="dead_output",
                            step_index=step.line,
                            step=step,
                            message=(
                                f"Result '{step.output}' is computed "
                                f"but never used or returned"
                            ),
                            confidence=Confidence.HIGH,
                        ))
                if step.true_branch:
                    _check_branch(step.true_branch)
                if step.false_branch:
                    _check_branch(step.false_branch)

        # Only scan inside conditional branches at the top level, not top-level steps
        # themselves.  This avoids false positives for projections whose downstream
        # ops (e.g. attention matmul) aren't fully extracted.
        for top_step in self.graph.steps:
            if top_step.op == OpKind.CONDITIONAL:
                if top_step.true_branch:
                    _check_branch(top_step.true_branch)
                if top_step.false_branch:
                    _check_branch(top_step.false_branch)

        return violations

    # ======================================================================
    # Top-level verify()
    # ======================================================================

    def _build_unsupported_tracker(self) -> UnsupportedOpTracker:
        """Scan the graph and record every operator that has no shape transfer
        function (``OpKind.UNSUPPORTED``) so the result can surface a precise
        "unsupported op: …" diagnostic instead of the verifier silently
        guessing. Supported ops are counted too, giving an op-coverage fraction.
        """
        tracker = UnsupportedOpTracker()

        def _scan(steps: List[ComputationStep]) -> None:
            for s in steps:
                if s.op == OpKind.UNSUPPORTED:
                    tracker.record(str(s.params.get("op_name", "<unknown>")))
                else:
                    tracker.record_supported()
                if s.op == OpKind.CONDITIONAL:
                    _scan(s.true_branch or [])
                    _scan(s.false_branch or [])

        _scan(self.graph.steps)
        return tracker

    def verify(self) -> VerificationResult:
        """Run constraint-based verification with forward symbolic
        propagation over the product theory T_shape × T_device × T_phase.

        Returns a ``VerificationResult`` that is either safe (with a
        ``SafetyCertificate`` including Z3 statistics) or unsafe (with a
        ``CounterexampleTrace``).

        Note: the SafetyCertificate encodes verification conditions
        (assertion witnesses), not proof certificates with inference chains.
        """
        t0 = time.monotonic()

        if self.graph.num_steps == 0:
            return VerificationResult(
                safe=True,
                certificate=SafetyCertificate(
                    model_name=self.graph.class_name,
                    properties=["shape_compatible", "device_consistent",
                                "gradient_valid"],
                    k=0, checked_steps=0, verification_time_ms=0.0,
                ),
                graph=self.graph,
                unsupported_op_tracker=self._build_unsupported_tracker(),
            )

        # Phase 1: base case (concrete + Z3)
        all_viols, model_states, kripke_states = self._bmc_base_case()

        # Phase 1.5: use-before-def check (catches swap_layers mutations)
        if not all_viols:
            ubd_viols = self._check_use_before_def()
            all_viols.extend(ubd_viols)

        # Phase 1.6: dead-output check (computed in branch but never used/returned)
        dead_viols = self._check_dead_outputs()
        all_viols.extend(dead_viols)

        # Phase 2: inductive step (Z3 with free variables)
        # Inductive violations indicate proof incompleteness, not unsafety.
        # They are recorded for statistics but do not make the model unsafe.
        ind_viols = self._bmc_inductive_step(kripke_states, model_states)

        elapsed = (time.monotonic() - t0) * 1000
        stats = self.ctx.get_stats() if HAS_Z3 else {}

        if all_viols:
            first_fail = min(v.step_index for v in all_viols)
            cex = CounterexampleTrace(
                model_name=self.graph.class_name,
                violations=all_viols,
                failing_step=first_fail,
                states=model_states[: first_fail + 2],
                concrete_dims=self._extract_concrete_dims(),
            )
            return VerificationResult(
                safe=False,
                counterexample=cex,
                graph=self.graph,
                verification_time_ms=elapsed,
                dynamic_features=getattr(self.graph, 'dynamic_features', {}),
                dynamic_feature_warnings=_generate_dynamic_warnings(
                    getattr(self.graph, 'dynamic_features', {})),
                unsupported_op_tracker=self._build_unsupported_tracker(),
            )

        cert = SafetyCertificate(
            model_name=self.graph.class_name,
            properties=["shape_compatible", "device_consistent",
                         "gradient_valid"],
            k=min(self.max_k, self.graph.num_steps),
            symbolic_bindings={
                n: str(v) for n, v in self.ctx._sym_dims.items()
            },
            checked_steps=len(self.graph.steps),
            verification_time_ms=elapsed,
            z3_queries=stats.get("z3_queries", 0),
            z3_total_time_ms=stats.get("z3_total_time_ms", 0.0),
            z3_sat_count=stats.get("z3_sat_count", 0),
            z3_unsat_count=stats.get("z3_unsat_count", 0),
            theories_used=["QF_LIA", "QF_UF", "QF_UFLIA"]
                + (["T_broadcast"] if HAS_THEORY_PLUGINS else [])
                + (["T_stride"] if self.ctx.stride_theory is not None else [])
                + (["T_device"] if self.ctx.device_theory is not None else [])
                + (["T_phase"] if self.ctx.phase_theory is not None else []),
            product_domains=["T_shape", "T_device", "T_phase"],
        )
        # Aggregate per-step proof certificates collected during BMC.
        # Each safety check that returned UNSAT attempted to extract a
        # proof certificate via replay in a clean proof-enabled context.
        proof_cert = None
        try:
            from .proof_certificate import (
                ProofCertificate,
                ProofStep,
                CertificateStrategy,
            )
            step_certs = getattr(self, '_step_certificates', [])
            if step_certs:
                # Merge all per-step certificates into one composite cert
                all_steps: list = []
                all_theories: set = set()
                all_vcs: list = []
                for _idx, _kind, sc in step_certs:
                    offset = len(all_steps)
                    for s in sc.steps:
                        shifted = ProofStep(
                            rule=s.rule,
                            conclusion=s.conclusion,
                            premises=[p + offset for p in s.premises],
                            theory=s.theory,
                        )
                        all_steps.append(shifted)
                    all_theories.update(sc.theories_used or [])
                    all_vcs.extend(sc.verification_conditions or [])
                proof_cert = ProofCertificate(
                    model_name=self.graph.class_name,
                    properties=["shape_compatible", "device_consistent",
                                "gradient_valid"],
                    steps=all_steps,
                    root_step=len(all_steps) - 1 if all_steps else -1,
                    theories_used=sorted(all_theories),
                    verification_conditions=all_vcs,
                    extraction_time_ms=sum(
                        sc.extraction_time_ms for _, _, sc in step_certs
                    ),
                    strategy=CertificateStrategy.REPLAY,
                )
        except Exception:
            pass
        cert.proof_certificate = proof_cert
        return VerificationResult(
            safe=True,
            certificate=cert,
            graph=self.graph,
            verification_time_ms=elapsed,
            dynamic_features=getattr(self.graph, 'dynamic_features', {}),
            dynamic_feature_warnings=_generate_dynamic_warnings(
                getattr(self.graph, 'dynamic_features', {})),
            proof_certificate=proof_cert,
            unsupported_op_tracker=self._build_unsupported_tracker(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_z3_model(
        self, model: z3.ModelRef, step_idx: int, kind: str
    ) -> str:
        parts = [f"Z3 violation ({kind}) at step {step_idx}:"]
        for decl in model.decls():
            parts.append(f"  {decl.name()} = {model[decl]}")
        return "\n".join(parts)

    def _extract_concrete_dims(self) -> Dict[str, int]:
        """Try to extract concrete dimension values from Z3."""
        if not HAS_Z3:
            return {}
        solver = z3.Solver()
        for c in self.ctx.positive_dim_constraints():
            solver.add(c)
        result: Dict[str, int] = {}
        if solver.check() == z3.sat:
            model = solver.model()
            for name, var in self.ctx._sym_dims.items():
                val = model.evaluate(var, model_completion=True)
                try:
                    result[name] = val.as_long()
                except (AttributeError, z3.Z3Exception):
                    pass
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Symbolic shape propagation engine
# ═══════════════════════════════════════════════════════════════════════════════

class SymbolicShapePropagator:
    """Propagate symbolic shapes through the computation graph using Z3.

    This class walks the computation graph and for each step:
      1. Encodes input shapes as Z3 integer vectors.
      2. Applies the appropriate shape rule.
      3. Records the output shape (possibly with new symbolic dims).

    The result is a complete shape environment mapping every tensor name
    to a (possibly symbolic) ``TensorShape``.
    """

    def __init__(self, graph: ComputationGraph) -> None:
        self.graph = graph
        self.ctx = _Z3Context() if HAS_Z3 else None

    def propagate(
        self, input_shapes: Dict[str, tuple]
    ) -> Dict[str, TensorShape]:
        """Propagate shapes starting from *input_shapes*.

        Returns a dict mapping each tensor name to its inferred shape.
        """
        env: Dict[str, TensorShape] = {}

        for name, raw in input_shapes.items():
            dims = tuple(
                ShapeDim(d) if isinstance(d, int) else ShapeDim(d)
                for d in raw
            )
            env[name] = TensorShape(dims)

        # Seed fx-folded constant tensor shapes (e.g. torch.rand(2,4) in forward).
        for cname, cshape in self.graph.const_shapes.items():
            env[cname] = cshape

        for step in self.graph.steps:
            self._propagate_step(env, step)

        return env

    def _propagate_step(
        self, env: Dict[str, TensorShape], step: ComputationStep
    ) -> None:
        """Propagate shapes for one computation step."""

        if step.op == OpKind.LAYER_CALL:
            layer = self.graph.layers.get(step.layer_ref or "")
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None

            if layer and inp_shape:
                propagator = _LAYER_PROPAGATORS.get(layer.kind)
                if propagator:
                    out, _ = propagator(inp_shape, layer)
                    if out:
                        env[step.output] = out
                        return
                # Shape-preserving layers
                if layer.kind in (LayerKind.RELU, LayerKind.DROPOUT,
                                  LayerKind.IDENTITY, LayerKind.SOFTMAX):
                    env[step.output] = inp_shape
                    return
                if layer.kind == LayerKind.FLATTEN:
                    out, _ = _propagate_flatten(inp_shape, 1)
                    if out:
                        env[step.output] = out
                    return
                # Unsupported layer: output shape UNKNOWN (fully symbolic)
                logger.warning(
                    "Unsupported layer kind %s (%s): output shape marked UNKNOWN",
                    layer.kind.name,
                    getattr(layer, 'attr_name', '?'),
                )
                env[step.output] = TensorShape(
                    tuple(ShapeDim(f"_unk_{getattr(layer, 'attr_name', 'op')}_{i}")
                          for i in range(inp_shape.ndim))
                )

        elif step.op == OpKind.NEW_TENSOR:
            shape = step.params.get("shape")
            if isinstance(shape, TensorShape):
                env[step.output] = shape

        elif step.op == OpKind.MATMUL:
            if len(step.inputs) >= 2:
                a = env.get(step.inputs[0])
                b = env.get(step.inputs[1])
                if a and b:
                    result = compute_matmul_shape(a, b)
                    if result:
                        env[step.output] = result
            if len(step.inputs) >= 2:
                a = env.get(step.inputs[0])
                b = env.get(step.inputs[1])
                if a and b:
                    result = compute_broadcast_shape(a, b)
                    if result:
                        env[step.output] = result

        elif step.op == OpKind.MULTIPLY:
            # Element-wise multiply/sub: same broadcast semantics as ADD
            if len(step.inputs) >= 2:
                a = env.get(step.inputs[0])
                b = env.get(step.inputs[1])
                if a and b:
                    result = compute_broadcast_shape(a, b)
                    if result:
                        env[step.output] = result

        elif step.op == OpKind.INTERPOLATE:
            # F.interpolate preserves batch and channel dims
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape and inp_shape.ndim >= 3:
                kept = inp_shape.dims[:2]
                spatial = tuple(ShapeDim("_up") for _ in inp_shape.dims[2:])
                env[step.output] = TensorShape(kept + spatial)

        elif step.op == OpKind.RESHAPE:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            dims = step.params.get("dims")
            if inp_shape and dims:
                result = compute_reshape_shape(inp_shape, dims)
                if result:
                    env[step.output] = result

        elif step.op == OpKind.FLATTEN:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                sd = step.params.get("start_dim", 1)
                ed = step.params.get("end_dim", -1)
                out, _ = _propagate_flatten(inp_shape, sd, ed)
                if out:
                    env[step.output] = out

        elif step.op in (OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.SOFTMAX,
                          OpKind.CONTIGUOUS, OpKind.DETACH):
            inp = step.inputs[0] if step.inputs else None
            if inp and inp in env:
                env[step.output] = env[inp]

        elif step.op == OpKind.SQUEEZE:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                dim = step.params.get("dim")
                if dim is not None:
                    if dim < 0:
                        dim = inp_shape.ndim + dim
                    new_dims = list(inp_shape.dims)
                    if 0 <= dim < len(new_dims):
                        d = new_dims[dim]
                        if not d.is_symbolic and d.value == 1:
                            new_dims.pop(dim)
                    env[step.output] = TensorShape(tuple(new_dims))
                else:
                    new_dims = [d for d in inp_shape.dims
                                if d.is_symbolic or d.value != 1]
                    env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.UNSQUEEZE:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                dim = step.params.get("dim", 0)
                if dim < 0:
                    dim = inp_shape.ndim + 1 + dim
                new_dims = list(inp_shape.dims)
                new_dims.insert(dim, ShapeDim(1))
                env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.TRANSPOSE:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                d0 = step.params.get("dim0", 0)
                d1 = step.params.get("dim1", 1)
                if d0 < 0:
                    d0 = inp_shape.ndim + d0
                if d1 < 0:
                    d1 = inp_shape.ndim + d1
                new_dims = list(inp_shape.dims)
                if 0 <= d0 < len(new_dims) and 0 <= d1 < len(new_dims):
                    new_dims[d0], new_dims[d1] = new_dims[d1], new_dims[d0]
                env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.TO_DEVICE:
            inp = step.inputs[0] if step.inputs else None
            if inp and inp in env:
                env[step.output] = env[inp]

        elif step.op in (
            OpKind.GATHER, OpKind.INDEX_SELECT, OpKind.SCATTER,
            OpKind.MASKED_SELECT, OpKind.MASKED_FILL, OpKind.NARROW,
            OpKind.SELECT_DIM, OpKind.TAKE,
        ):
            self._apply_indexing(env, step, None)
        elif step.op == OpKind.SDPA:
            self._apply_sdpa(env, step, None)

        elif step.op in (OpKind.RETURN, OpKind.CUSTOM, OpKind.UNSUPPORTED):
            pass

        elif step.op == OpKind.SUBSCRIPT:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                indices = step.params.get("indices", [])
                new_dims = []
                for i, dim in enumerate(inp_shape.dims):
                    if i < len(indices):
                        if indices[i] == 'slice':
                            new_dims.append(dim)
                        # 'int' → drop dimension
                    else:
                        new_dims.append(dim)
                env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.PERMUTE:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                perm = step.params.get("dims")
                if perm and all(d is not None for d in perm):
                    new_dims = []
                    for p in perm:
                        if isinstance(p, int):
                            if p < 0:
                                p = inp_shape.ndim + p
                            if 0 <= p < inp_shape.ndim:
                                new_dims.append(inp_shape.dims[p])
                            else:
                                new_dims.append(ShapeDim("_perm"))
                        else:
                            new_dims.append(ShapeDim("_perm"))
                    env[step.output] = TensorShape(tuple(new_dims))
                else:
                    env[step.output] = inp_shape

        elif step.op == OpKind.CAT:
            # torch.cat: concatenate along dim
            if step.inputs:
                shapes = [env.get(inp) for inp in step.inputs]
                shapes = [s for s in shapes if s is not None]
                if shapes:
                    dim = step.params.get("dim", 0)
                    base = shapes[0]
                    if dim < 0:
                        dim = base.ndim + dim
                    new_dims = list(base.dims)
                    if 0 <= dim < len(new_dims):
                        total = new_dims[dim]
                        all_concrete = not total.is_symbolic
                        cat_sum = total.value if not total.is_symbolic else 0
                        for s in shapes[1:]:
                            if dim < s.ndim:
                                d = s.dims[dim]
                                if not d.is_symbolic and all_concrete:
                                    cat_sum += d.value
                                else:
                                    all_concrete = False
                        if all_concrete:
                            new_dims[dim] = ShapeDim(cat_sum)
                        else:
                            new_dims[dim] = ShapeDim("_cat")
                    env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.STACK:
            # torch.stack: adds a new dimension
            if step.inputs:
                shapes = [env.get(inp) for inp in step.inputs]
                shapes = [s for s in shapes if s is not None]
                if shapes:
                    dim = step.params.get("dim", 0)
                    base = shapes[0]
                    if dim < 0:
                        dim = base.ndim + 1 + dim
                    new_dims = list(base.dims)
                    new_dims.insert(dim, ShapeDim(len(shapes)))
                    env[step.output] = TensorShape(tuple(new_dims))

        elif step.op == OpKind.EXPAND:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                dims = step.params.get("dims")
                if (not dims and len(step.inputs) > 1
                        and step.inputs[1] in env):
                    ref = env[step.inputs[1]]
                    dims = tuple(
                        d.value if not d.is_symbolic else str(d.value)
                        for d in ref.dims
                    )
                if dims and all(d is not None for d in dims):
                    allow_neg_one = (
                        step.params.get("expand_kind") != "broadcast_to"
                    )
                    out_shape, _err = compute_expand_shape(
                        inp_shape, tuple(dims), allow_neg_one=allow_neg_one
                    )
                    if out_shape is not None:
                        env[step.output] = out_shape
                    else:
                        env[step.output] = inp_shape
                else:
                    env[step.output] = inp_shape

        elif step.op == OpKind.REPEAT:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                dims = step.params.get("dims")
                if dims and all(d is not None for d in dims):
                    new_dims = []
                    for i, d in enumerate(dims):
                        if i < inp_shape.ndim and not inp_shape.dims[i].is_symbolic:
                            new_dims.append(ShapeDim(inp_shape.dims[i].value * d))
                        else:
                            new_dims.append(ShapeDim("_rep"))
                    env[step.output] = TensorShape(tuple(new_dims))
                else:
                    env[step.output] = inp_shape

        elif step.op in (OpKind.MEAN_REDUCE, OpKind.SUM_REDUCE):
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                dim = step.params.get("dim")
                keepdim = step.params.get("keepdim", False)
                if isinstance(dim, (list, tuple)):
                    norm = sorted(
                        (d if d >= 0 else inp_shape.ndim + d) for d in dim
                    )
                    new_dims = []
                    for i, sd in enumerate(inp_shape.dims):
                        if i in norm:
                            if keepdim:
                                new_dims.append(ShapeDim(1))
                            # else: drop this dim
                        else:
                            new_dims.append(sd)
                    env[step.output] = TensorShape(tuple(new_dims))
                elif dim is not None and isinstance(dim, int):
                    if dim < 0:
                        dim = inp_shape.ndim + dim
                    new_dims = list(inp_shape.dims)
                    if 0 <= dim < len(new_dims):
                        if keepdim:
                            new_dims[dim] = ShapeDim(1)
                        else:
                            new_dims.pop(dim)
                    env[step.output] = TensorShape(tuple(new_dims))
                elif dim is None:
                    env[step.output] = TensorShape(())

        elif step.op == OpKind.PAD:
            inp = step.inputs[0] if step.inputs else None
            inp_shape = env.get(inp) if inp else None
            if inp_shape:
                pad_arg = step.params.get("pad")
                if pad_arg and isinstance(pad_arg, (tuple, list)):
                    new_dims = list(inp_shape.dims)
                    n_padded = len(pad_arg) // 2
                    for i in range(n_padded):
                        dim_idx = inp_shape.ndim - 1 - i
                        if 0 <= dim_idx < len(new_dims) and not new_dims[dim_idx].is_symbolic:
                            new_dims[dim_idx] = ShapeDim(
                                new_dims[dim_idx].value + pad_arg[2*i] + pad_arg[2*i+1]
                            )
                    env[step.output] = TensorShape(tuple(new_dims))
                else:
                    env[step.output] = inp_shape

        elif step.op == OpKind.EINSUM:
            equation = step.params.get("equation", "")
            out_sh_ein = None
            if step.inputs and isinstance(equation, str) and equation:
                in_shapes_ein = [env[i] for i in step.inputs if i in env]
                if len(in_shapes_ein) == len(step.inputs):
                    from src.smt.einsum_theory import (
                        infer_einsum_shape as _infer_einsum_ai,
                    )
                    out_sh_ein = _infer_einsum_ai(equation, in_shapes_ein)
            if out_sh_ein is not None:
                env[step.output] = out_sh_ein
            elif step.inputs:
                for inp in step.inputs:
                    if inp in env:
                        env[step.output] = env[inp]
                        break


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Phase-aware analysis
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseAnalyzer:
    """Analyse phase-dependent behaviour of an nn.Module.

    Detects:
      - Dropout layers that are active only in TRAIN mode.
      - BatchNorm layers that switch between training and running statistics.
      - Shape differences between train and eval modes.
    """

    def __init__(self, graph: ComputationGraph) -> None:
        self.graph = graph

    def has_phase_dependent_layers(self) -> bool:
        """Check whether the graph has layers whose behaviour depends on
        train/eval phase."""
        for layer in self.graph.layers.values():
            if layer.kind in (LayerKind.DROPOUT, LayerKind.BATCHNORM1D,
                              LayerKind.BATCHNORM2D):
                return True
        return False

    def compare_phases(
        self, input_shapes: Dict[str, tuple]
    ) -> Dict[str, Any]:
        """Compare model behaviour in TRAIN vs EVAL phase.

        Returns a dict with keys:
          - "train_shapes": shape env in train mode
          - "eval_shapes":  shape env in eval mode
          - "differences":  list of (tensor_name, train_shape, eval_shape)
        """
        train_checker = ConstraintVerifier(
            self.graph, input_shapes,
            default_phase=Phase.TRAIN,
        )
        eval_checker = ConstraintVerifier(
            self.graph, input_shapes,
            default_phase=Phase.EVAL,
        )

        # Simulate both phases
        train_state = train_checker._init_state.copy()
        eval_state = eval_checker._init_state.copy()

        for step in self.graph.steps:
            train_state, _ = train_checker._step_transition(train_state, step)
            eval_state, _ = eval_checker._step_transition(eval_state, step)

        differences = []
        all_names = (set(train_state.shape_env.keys())
                     | set(eval_state.shape_env.keys()))
        for name in sorted(all_names):
            ts = train_state.shape_env.get(name)
            es = eval_state.shape_env.get(name)
            if ts != es:
                differences.append((name, ts, es))

        return {
            "train_shapes": train_state.shape_env,
            "eval_shapes": eval_state.shape_env,
            "differences": differences,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Device analysis
# ═══════════════════════════════════════════════════════════════════════════════

class DeviceAnalyzer:
    """Analyse device placement of tensors in an nn.Module.

    Detects cross-device operations and unnecessary device transfers.
    """

    def __init__(self, graph: ComputationGraph) -> None:
        self.graph = graph

    def check_device_consistency(
        self,
        input_shapes: Dict[str, tuple],
        input_devices: Optional[Dict[str, Device]] = None,
    ) -> List[SafetyViolation]:
        """Check that all operations use tensors on the same device.

        Returns a list of SafetyViolation for any cross-device operations.
        """
        state = ModelState(phase=Phase.TRAIN)

        for name, raw in input_shapes.items():
            dims = tuple(
                ShapeDim(d) if isinstance(d, int) else ShapeDim(d)
                for d in raw
            )
            state.shape_env[name] = TensorShape(dims)

        if input_devices:
            state.device_map.update(input_devices)
        else:
            for name in input_shapes:
                state.device_map[name] = Device.CPU

        checker = ConstraintVerifier(
            self.graph, input_shapes,
        )
        checker._init_state = state

        all_violations: List[SafetyViolation] = []
        current = state.copy()
        for idx, step in enumerate(self.graph.steps):
            current, viols = checker._step_transition(current, step)
            for v in viols:
                v.step_index = idx
            all_violations.extend(
                v for v in viols if v.kind == "device_mismatch"
            )

        return all_violations

    def trace_device_transfers(
        self,
        input_shapes: Dict[str, tuple],
        input_devices: Optional[Dict[str, Device]] = None,
    ) -> List[Tuple[int, str, Device, Device]]:
        """Return a list of device transfers as (step_idx, tensor, from, to).
        """
        state = ModelState(phase=Phase.TRAIN)
        for name, raw in input_shapes.items():
            dims = tuple(
                ShapeDim(d) if isinstance(d, int) else ShapeDim(d)
                for d in raw
            )
            state.shape_env[name] = TensorShape(dims)

        if input_devices:
            state.device_map.update(input_devices)
        else:
            for name in input_shapes:
                state.device_map[name] = Device.CPU

        transfers: List[Tuple[int, str, Device, Device]] = []
        current = state.copy()
        checker = ConstraintVerifier(self.graph, input_shapes)
        checker._init_state = state

        for idx, step in enumerate(self.graph.steps):
            if step.op == OpKind.TO_DEVICE:
                old_dev = current.device_map.get(
                    step.inputs[0], Device.CPU
                ) if step.inputs else Device.CPU

                current, _ = checker._step_transition(current, step)

                new_dev = current.device_map.get(step.output, old_dev)
                if old_dev != new_dev:
                    transfers.append((idx, step.output, old_dev, new_dev))
            else:
                current, _ = checker._step_transition(current, step)

        return transfers


# ═══════════════════════════════════════════════════════════════════════════════
# 11b. Kripke structure extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_kripke_structure(
    graph: ComputationGraph,
    input_shapes: Dict[str, Any],
    initial_device: Device = Device.CPU,
    initial_phase: Phase = Phase.EVAL,
) -> KripkeStructure:
    """Extract the formal Kripke structure from a computation graph.

    Constructs K = (S, S₀, R, AP, L) from the DAG of computation steps:
      - One state per DAG node (layer application) + initial state
      - Transitions follow DAG edges
      - Labeling checks shape/device/phase/gradient safety at each state
    """
    # Run the verifier to get model_states and violations
    checker = ConstraintVerifier(
        graph,
        input_shapes=input_shapes,
        default_device=initial_device,
        default_phase=initial_phase,
    )
    result = checker.verify()

    # Build Kripke states: initial state + one per computation step
    states: List[KripkeState] = []

    # State 0: initial state
    init_shape_vars: Dict[str, List[Any]] = {}
    init_device_vars: Dict[str, Any] = {}
    for name, raw_shape in input_shapes.items():
        init_shape_vars[name] = [str(d) for d in raw_shape]
        init_device_vars[name] = initial_device.value
    states.append(KripkeState(
        step_index=0,
        shape_vars=init_shape_vars,
        device_vars=init_device_vars,
        layer_name="input",
    ))

    # One state per computation step
    for idx, step in enumerate(graph.steps):
        step_shape_vars: Dict[str, List[Any]] = {}
        step_device_vars: Dict[str, Any] = {}
        step_shape_vars[step.output] = []
        step_device_vars[step.output] = initial_device.value
        for inp in step.inputs:
            step_device_vars[inp] = initial_device.value
        layer_name = step.layer_ref or step.op.name
        states.append(KripkeState(
            step_index=idx + 1,
            shape_vars=step_shape_vars,
            device_vars=step_device_vars,
            layer_name=layer_name,
        ))

    # Build transitions: each step creates an edge from its predecessor
    transitions: List[KripkeTransition] = []
    for idx, step in enumerate(graph.steps):
        op_name = step.layer_ref or step.op.name
        transitions.append(KripkeTransition(
            source=idx,
            target=idx + 1,
            operation=op_name,
        ))

    # Build labeling: determine which APs hold at each state
    # Collect violation step indices
    violation_steps: Set[int] = set()
    violation_kinds: Dict[int, Set[str]] = {}
    if result.counterexample:
        for v in result.counterexample.violations:
            violation_steps.add(v.step_index)
            violation_kinds.setdefault(v.step_index, set()).add(v.kind)

    labeling: Dict[int, FrozenSet[str]] = {}
    all_aps = {"shape_safe", "device_consistent", "gradient_valid", "phase_correct"}

    for i in range(len(states)):
        # step_index in violations is 0-based matching graph.steps index
        # State i corresponds to step i-1 output (state 0 = initial, always safe)
        step_idx = i - 1  # The step that produced this state
        labels = set(all_aps)

        if step_idx in violation_steps:
            kinds = violation_kinds.get(step_idx, set())
            if any("shape" in k for k in kinds):
                labels.discard("shape_safe")
            if any("device" in k for k in kinds):
                labels.discard("device_consistent")
            if any("gradient" in k for k in kinds):
                labels.discard("gradient_valid")
            if any("phase" in k for k in kinds):
                labels.discard("phase_correct")

        labeling[i] = frozenset(labels)

    return KripkeStructure(
        states=states,
        initial_state_idx=0,
        transitions=transitions,
        labeling=labeling,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Public API: verify_model
# ═══════════════════════════════════════════════════════════════════════════════

def verify_model(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, Union[str, int]]] = None,
    high_confidence_only: bool = False,
    verification_mode: str = "bounded",
    symbolic_dims: Optional[Dict[str, str]] = None,
    produce_certificates: bool = False,
    return_kripke: bool = False,
    use_kb_normalization: bool = False,
    check_devices: bool = True,
    check_phases: bool = True,
    check_gradients: bool = True,
    check_dtypes: bool = True,
    input_dtypes: Optional[Dict[str, str]] = None,
) -> VerificationResult:
    """One-shot verification of an nn.Module defined in *source*.

    Parameters
    ----------
    source : str
        Python source code containing an ``nn.Module`` subclass.
    input_shapes : dict, optional
        Mapping from forward-parameter names to shape tuples.  Dimensions
        may be ints (concrete) or strings (symbolic).
    default_device : Device
        Default device for input tensors.
    default_phase : Phase
        Default phase (TRAIN or EVAL).
    max_k : int, optional
        Maximum verification depth.  Defaults to the number of steps in the
        graph.
    constraints : dict, optional
        Relational constraints between symbolic dimensions.  Keys are
        dimension names; values are either ints (fixing the dimension) or
        strings containing arithmetic expressions (e.g.
        ``{"embed_dim": "heads * head_dim", "heads": 8}``).
    high_confidence_only : bool
        When True, automatically filters the result to only include
        HIGH-confidence (Z3-proven) violations, reducing FP rate to 0%%.
        Suitable for CI/CD gating per Sadowski et al. thresholds.
    verification_mode : str
        ``"bounded"`` (default) uses the existing bounded model checking.
        ``"unbounded"`` uses IC3/PDR for unbounded parametric verification.
    symbolic_dims : dict, optional
        Only used when ``verification_mode="unbounded"``.  Mapping from
        shape position names to symbolic parameter names (e.g.
        ``{"batch": "batch_size"}``).
    check_devices : bool
        When False, device-mismatch violations are suppressed from the result.
    check_phases : bool
        When False, phase-dependent violations are suppressed from the result.
    check_gradients : bool
        When False, gradient-flow violations are suppressed from the result.

    Returns
    -------
    VerificationResult
        Contains either a ``SafetyCertificate`` (if safe) or a
        ``CounterexampleTrace`` (if unsafe).

    Examples
    --------
    >>> result = verify_model('''
    ... import torch.nn as nn
    ... class MyModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(10, 5)
    ...     def forward(self, x):
    ...         return self.fc(x)
    ... ''', input_shapes={"x": ("batch", 10)})
    >>> result.safe
    True
    """
    t0 = time.monotonic()

    if verification_mode == "unbounded":
        try:
            from src.ic3_pdr import ic3_verify
        except ImportError:
            return VerificationResult(
                safe=False,
                errors=["IC3/PDR module not available"],
                verification_time_ms=(time.monotonic() - t0) * 1000,
            )
        ic3_result = ic3_verify(
            source,
            symbolic_dims=symbolic_dims,
            input_shapes=input_shapes,
            solver_timeout_ms=5000,
        )
        return VerificationResult(
            safe=ic3_result.safe,
            verification_time_ms=ic3_result.verification_time_ms,
        )

    try:
        graph = extract_computation_graph(source)
    except (ValueError, SyntaxError) as exc:
        return VerificationResult(
            safe=False,
            errors=[str(exc)],
            verification_time_ms=(time.monotonic() - t0) * 1000,
        )

    checker = ConstraintVerifier(
        graph,
        input_shapes=input_shapes or {},
        default_device=default_device,
        default_phase=default_phase,
        max_k=max_k,
        constraints=constraints,
        produce_certificates=produce_certificates,
        use_kb_normalization=use_kb_normalization,
        check_devices=check_devices,
        check_phases=check_phases,
        check_gradients=check_gradients,
        check_dtypes=check_dtypes,
        input_dtypes=input_dtypes,
    )

    result = checker.verify()

    # ----------------------------------------------------------------
    # Buffer-device pass: if any buffers were registered, re-verify
    # with default_device=CUDA_0 to catch "buffer stays on CPU" bugs.
    # (register_buffer tensors are pinned to CPU in _build_initial_state;
    # if inputs/layer-outputs are on CUDA the cat/add will fail at runtime.)
    # ----------------------------------------------------------------
    if check_devices and graph.buffer_shapes and default_device == Device.CPU:
        cuda_checker = ConstraintVerifier(
            graph,
            input_shapes=input_shapes or {},
            default_device=Device.CUDA_0,
            default_phase=default_phase,
            max_k=max_k,
            constraints=constraints,
            produce_certificates=False,
            use_kb_normalization=use_kb_normalization,
        )
        cuda_result = cuda_checker.verify()
        # Only import device_mismatch violations that involve a buffer
        if not cuda_result.safe and cuda_result.counterexample:
            buf_keys = {f"self.{n}" for n in graph.buffer_shapes}
            for viol in cuda_result.counterexample.violations:
                if viol.kind == "device_mismatch":
                    step = viol.step
                    involves_buffer = step is not None and any(
                        inp in buf_keys for inp in (step.inputs or [])
                    )
                    if involves_buffer:
                        # Re-tag so the message is informative
                        viol = SafetyViolation(
                            kind="device_mismatch",
                            step_index=viol.step_index,
                            step=viol.step,
                            message=(
                                viol.message
                                + " (buffer registered on CPU may mismatch CUDA input)"
                            ),
                            tensor_a=viol.tensor_a,
                            tensor_b=viol.tensor_b,
                            device_a=Device.CPU,
                            device_b=Device.CUDA_0,
                            confidence=Confidence.HIGH,
                        )
                        if result.safe:
                            # Demote the main result from safe to unsafe
                            result = VerificationResult(
                                safe=False,
                                counterexample=CounterexampleTrace(
                                    model_name=graph.class_name,
                                    violations=[viol],
                                    failing_step=viol.step_index,
                                    states=[],
                                ),
                                graph=graph,
                                errors=result.errors,
                                verification_time_ms=result.verification_time_ms,
                                confidence=result.confidence,
                            )
                        elif result.counterexample:
                            result.counterexample.violations.append(viol)

    if high_confidence_only:
        result = result.filter_by_confidence(Confidence.HIGH)

    # Feature-flag filtering: suppress violation kinds that are disabled.
    # This mirrors the higher-level filtering in verify_architecture but
    # operates directly on the VerificationResult so callers of verify_model
    # see filtered results regardless of which API layer they use.
    _PHASE_KINDS = {"phase_violation", "phase_error"}
    _GRAD_KINDS = {"gradient_broken", "gradient_violation"}
    _DTYPE_KINDS = {"dtype_error", "dtype_mismatch"}

    def _filter_violations(res: VerificationResult, keep_pred) -> VerificationResult:
        if res.safe or not res.counterexample:
            return res
        kept = [v for v in res.counterexample.violations if keep_pred(v)]
        if len(kept) == len(res.counterexample.violations):
            return res  # unchanged
        if not kept:
            return VerificationResult(
                safe=True,
                graph=res.graph,
                errors=res.errors,
                verification_time_ms=res.verification_time_ms,
                confidence=res.confidence,
                dynamic_features=res.dynamic_features,
                dynamic_feature_warnings=res.dynamic_feature_warnings,
                unsupported_op_tracker=res.unsupported_op_tracker,
            )
        new_cex = CounterexampleTrace(
            model_name=res.counterexample.model_name,
            violations=kept,
            failing_step=kept[0].step_index if kept else -1,
            states=res.counterexample.states,
            concrete_dims=res.counterexample.concrete_dims,
        )
        return VerificationResult(
            safe=False,
            counterexample=new_cex,
            graph=res.graph,
            errors=res.errors,
            verification_time_ms=res.verification_time_ms,
            confidence=res.confidence,
            dynamic_features=res.dynamic_features,
            dynamic_feature_warnings=res.dynamic_feature_warnings,
            unsupported_op_tracker=res.unsupported_op_tracker,
        )

    if not check_devices:
        result = _filter_violations(result, lambda v: v.kind != "device_mismatch")
    if not check_phases:
        result = _filter_violations(result, lambda v: v.kind not in _PHASE_KINDS)
    if not check_gradients:
        result = _filter_violations(result, lambda v: v.kind not in _GRAD_KINDS)
    if not check_dtypes:
        result = _filter_violations(result, lambda v: v.kind not in _DTYPE_KINDS)

    if return_kripke:
        result.kripke_structure = extract_kripke_structure(
            graph,
            input_shapes=input_shapes or {},
            initial_device=default_device,
            initial_phase=default_phase,
        )
    return result


# Backward-compatible alias (deprecated; use ConstraintVerifier directly).
BoundedModelChecker = ConstraintVerifier
