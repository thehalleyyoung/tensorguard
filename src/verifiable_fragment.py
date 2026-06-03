"""
Formal syntactic characterization of TensorGuard's verifiable fragment.

Defines the subset of PyTorch ``nn.Module`` code that TensorGuard can analyze
via ``torch.fx.symbolic_trace``.  The verifiable fragment is characterized as
a formal grammar over module definitions and forward-method statements, and
a pre-verification ``check_traceability`` function reports whether a given
module lies within this fragment.

Grammar (Verifiable Fragment V_TG)
===================================

A module M is in V_TG iff:

  Module      ::=  class C(nn.Module):
                       __init__(self, <params>): <InitBody>
                       forward(self, <inputs>): <FwdBody>

  InitBody    ::=  (self.<attr> = <LayerExpr>)*
  LayerExpr   ::=  nn.<SupportedLayer>(<literal>*)
                |  nn.Sequential(<LayerExpr>*)
                |  nn.ModuleList([<LayerExpr>*])

  FwdBody     ::=  <Stmt>*; return <Expr>

  Stmt        ::=  <var> = <Expr>
                |  <var> = self.<attr>(<Expr>*)          # submodule call
                |  for <var> in self.<modulelist>:       # static iteration
                       <Stmt>*

  Expr        ::=  <var>
                |  self.<attr>(<Expr>*)                  # nn.Module call
                |  <SupportedFunc>(<Expr>*)              # torch.* / F.*
                |  <Expr>.<SupportedMethod>(<literal>*)  # tensor method
                |  <Expr> <BinOp> <Expr>                 # +, *, @
                |  <literal>

  SupportedFunc  ::=  torch.{cat,stack,matmul,mm,bmm,flatten,relu,...}
                   |  F.{relu,gelu,silu,sigmoid,...,linear,conv2d,...}
                   |  operator.{add,mul,getitem}

  SupportedMethod ::= view | reshape | flatten | squeeze | unsqueeze
                    |  transpose | permute | contiguous | detach
                    |  to | cuda | cpu | mean | sum | expand | repeat | tile
                    |  repeat_interleave
                    |  chunk | split | softmax | relu | sigmoid | tanh

  SupportedLayer  ::= Linear | Conv{1,2,3}d | ConvTranspose{1,2,3}d
                    |  BatchNorm{1,2,3}d | LayerNorm | GroupNorm
                    |  InstanceNorm{1,2,3}d | SyncBatchNorm
                    |  Dropout | Dropout{2,3}d | AlphaDropout
                    |  ReLU | GELU | SiLU | Tanh | Sigmoid | ...
                    |  MaxPool{1,2,3}d | AvgPool{1,2}d
                    |  Adaptive{Avg,Max}Pool{1,2}d
                    |  Embedding | LSTM | GRU | RNN
                    |  MultiheadAttention
                    |  TransformerEncoder{,Layer}
                    |  TransformerDecoder{,Layer}
                    |  Flatten | Identity | Upsample | PixelShuffle
                    |  ...  (full list in SUPPORTED_LAYER_TYPES)

Excluded (outside V_TG):
  - Data-dependent control flow:  if <tensor>.cond: ... / while ...
  - Data-dependent iteration:     for _ in range(int(x.item())): ...
  - Dynamic assertions:           assert <tensor_expr>
  - .item() / .tolist() calls:    converts tensor to Python scalar
  - Custom autograd Functions:    torch.autograd.Function subclasses
  - In-place mutation *of inputs*: x += y  (in-place on non-leaf is OK in fx)
  - torch.jit.script modules
  - External library calls that are opaque to torch.fx
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Type

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.fx
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════════════
# Unsupported-construct taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

class UnsupportedCategory(Enum):
    """Taxonomy of constructs outside the verifiable fragment."""
    DATA_DEPENDENT_CONTROL_FLOW = auto()
    DATA_DEPENDENT_ITERATION = auto()
    DYNAMIC_ASSERTION = auto()
    TENSOR_TO_SCALAR = auto()
    CUSTOM_AUTOGRAD = auto()
    INPLACE_MUTATION = auto()
    JIT_SCRIPT = auto()
    OPAQUE_EXTERNAL_CALL = auto()
    DYNAMIC_MODULE_CONSTRUCTION = auto()
    UNSUPPORTED_BUILTIN = auto()
    OTHER = auto()


@dataclass
class UnsupportedConstruct:
    """A single construct that puts a module outside the verifiable fragment."""
    category: UnsupportedCategory
    description: str
    location: Optional[str] = None  # e.g. "forward:line 12"
    severity: str = "blocking"      # "blocking" or "warning"


# ═══════════════════════════════════════════════════════════════════════════════
# Traceability report
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TraceabilityReport:
    """Result of checking whether a module is in the verifiable fragment."""
    module_name: str
    in_verifiable_fragment: bool
    fx_traceable: bool
    fx_trace_error: Optional[str] = None
    unsupported_constructs: List[UnsupportedConstruct] = field(default_factory=list)
    supported_layers: List[str] = field(default_factory=list)
    unsupported_layers: List[str] = field(default_factory=list)
    num_parameters: int = 0
    num_submodules: int = 0
    static_warnings: List[str] = field(default_factory=list)

    @property
    def blocking_issues(self) -> List[UnsupportedConstruct]:
        return [c for c in self.unsupported_constructs if c.severity == "blocking"]

    def summary(self) -> str:
        status = "IN fragment" if self.in_verifiable_fragment else "OUTSIDE fragment"
        lines = [f"{self.module_name}: {status}"]
        if self.fx_trace_error:
            lines.append(f"  FX trace error: {self.fx_trace_error}")
        for c in self.unsupported_constructs:
            lines.append(f"  [{c.category.name}] {c.description}")
        if self.unsupported_layers:
            lines.append(f"  Unsupported layers: {', '.join(self.unsupported_layers)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Formal grammar AST node types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GrammarNode:
    """Base class for verifiable-fragment grammar AST nodes."""
    pass


@dataclass
class ModuleDef(GrammarNode):
    """Module ::= class C(nn.Module): __init__; forward"""
    name: str
    layers: List[LayerDecl] = field(default_factory=list)
    forward_stmts: List[FwdStmt] = field(default_factory=list)


@dataclass
class LayerDecl(GrammarNode):
    """InitBody entry: self.<attr> = nn.<Layer>(...)"""
    attr_name: str
    layer_type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FwdStmt(GrammarNode):
    """A single statement in the forward body."""
    pass


@dataclass
class SubmoduleCall(FwdStmt):
    """self.<attr>(<args>)"""
    attr_name: str
    args: List[str] = field(default_factory=list)


@dataclass
class FunctionCall(FwdStmt):
    """torch.<func>(<args>) or F.<func>(<args>)"""
    function_name: str
    args: List[str] = field(default_factory=list)


@dataclass
class MethodCall(FwdStmt):
    """<tensor>.<method>(<args>)"""
    method_name: str
    receiver: str = ""
    args: List[str] = field(default_factory=list)


@dataclass
class BinaryOp(FwdStmt):
    """<expr> <op> <expr>"""
    op: str
    left: str = ""
    right: str = ""


@dataclass
class ReturnStmt(FwdStmt):
    """return <expr>"""
    expr: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Supported constructs registry
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_LAYER_TYPES: Set[str] = {
    "Linear", "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm",
    "LayerNorm", "GroupNorm",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
    "Dropout", "Dropout2d", "Dropout3d", "AlphaDropout",
    "ReLU", "ReLU6", "GELU", "SiLU", "Tanh", "Sigmoid",
    "LeakyReLU", "ELU", "PReLU", "SELU", "Mish", "Hardswish", "Hardsigmoid",
    "Softmax", "LogSoftmax",
    "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "AvgPool1d", "AvgPool2d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d",
    "AdaptiveMaxPool1d", "AdaptiveMaxPool2d",
    "LPPool2d", "FractionalMaxPool2d",
    "Embedding", "LSTM", "GRU", "RNN",
    "MultiheadAttention",
    "TransformerEncoder", "TransformerDecoder",
    "TransformerEncoderLayer", "TransformerDecoderLayer",
    "Flatten", "Unflatten", "Identity",
    "Upsample", "PixelShuffle", "PixelUnshuffle",
    "Unfold", "Fold",
    "ReflectionPad2d", "ReplicationPad2d", "ZeroPad2d", "ConstantPad2d",
    "Sequential", "ModuleList", "ModuleDict",
}

SUPPORTED_TENSOR_METHODS: Set[str] = {
    "view", "reshape", "flatten", "squeeze", "unsqueeze",
    "movedim", "moveaxis", "transpose", "swapaxes", "swapdims",
    "permute", "roll", "rot90", "flip", "contiguous", "detach",
    "to", "cuda", "cpu", "float", "half", "double",
    "mean", "sum", "expand", "repeat", "tile", "repeat_interleave",
    "chunk", "split", "softmax", "relu", "sigmoid", "tanh",
    "size", "shape", "dim", "numel",
    "clone", "gather", "index_select", "scatter", "scatter_", "scatter_add",
    "scatter_add_", "narrow", "select",
    "add", "mul", "matmul", "bmm", "mm",
    "add_", "mul_",  # in-place on intermediates is OK
}

SUPPORTED_TORCH_FUNCTIONS: Set[str] = {
    "torch.cat", "torch.stack", "torch.hstack", "torch.vstack",
    "torch.dstack", "torch.column_stack", "torch.row_stack",
    "torch.squeeze", "torch.unsqueeze", "torch.movedim", "torch.moveaxis",
    "torch.swapaxes", "torch.swapdims", "torch.roll", "torch.rot90",
    "torch.flip", "torch.repeat_interleave", "torch.tile",
    "torch.broadcast_tensors", "torch.broadcast_shapes",
    "torch.matmul", "torch.mm", "torch.bmm",
    "torch.flatten", "torch.relu", "torch.sigmoid", "torch.tanh",
    "torch.softmax", "torch.dropout", "torch.where",
    "torch.chunk", "torch.split", "torch.einsum",
    "torch.gather", "torch.index_select", "torch.scatter", "torch.scatter_add",
    "torch.add", "torch.mul",
    "torch.zeros", "torch.ones", "torch.zeros_like", "torch.ones_like",
    "torch.arange", "torch.linspace",
}

SUPPORTED_F_FUNCTIONS: Set[str] = {
    "F.relu", "F.gelu", "F.silu", "F.sigmoid", "F.tanh",
    "F.leaky_relu", "F.elu", "F.softmax", "F.log_softmax",
    "F.dropout", "F.linear", "F.conv2d", "F.conv1d",
    "F.batch_norm", "F.layer_norm", "F.group_norm",
    "F.max_pool2d", "F.avg_pool2d", "F.adaptive_avg_pool2d",
    "F.interpolate", "F.pad",
}

# Patterns indicating data-dependent control flow in source
_DATA_DEPENDENT_PATTERNS = {
    ".item()", ".tolist()", ".numpy()",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Static source-level analysis
# ═══════════════════════════════════════════════════════════════════════════════

class _ForwardAnalyzer(ast.NodeVisitor):
    """AST visitor that detects constructs outside the verifiable fragment."""

    def __init__(self):
        self.issues: List[UnsupportedConstruct] = []
        self.warnings: List[str] = []
        self._in_forward = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "forward":
            self._in_forward = True
            self.generic_visit(node)
            self._in_forward = False
        else:
            self.generic_visit(node)

    def visit_If(self, node: ast.If):
        if self._in_forward and self._is_tensor_dependent(node.test):
            # self.training is a known-safe pattern (torch.fx handles it)
            if not self._is_training_guard(node.test):
                self.issues.append(UnsupportedConstruct(
                    category=UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW,
                    description="if-branch conditioned on tensor value",
                    location=f"forward:line {node.lineno}",
                ))
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        if self._in_forward and self._is_tensor_dependent(node.test):
            self.issues.append(UnsupportedConstruct(
                category=UnsupportedCategory.DATA_DEPENDENT_ITERATION,
                description="while-loop with tensor-dependent condition",
                location=f"forward:line {node.lineno}",
            ))
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if self._in_forward and self._is_data_dependent_range(node):
            self.issues.append(UnsupportedConstruct(
                category=UnsupportedCategory.DATA_DEPENDENT_ITERATION,
                description="for-loop with data-dependent iteration count",
                location=f"forward:line {node.lineno}",
            ))
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        if self._in_forward:
            self.issues.append(UnsupportedConstruct(
                category=UnsupportedCategory.DYNAMIC_ASSERTION,
                description="assert statement in forward (may reference tensors)",
                location=f"forward:line {node.lineno}",
            ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if self._in_forward:
            func_str = self._call_name(node)
            # .item() / .tolist() / .numpy()
            if func_str and any(p.lstrip(".") == func_str.split(".")[-1]
                                for p in (".item", ".tolist", ".numpy")
                                if func_str.endswith(p.lstrip("."))):
                self.issues.append(UnsupportedConstruct(
                    category=UnsupportedCategory.TENSOR_TO_SCALAR,
                    description=f"call to {func_str} converts tensor to Python value",
                    location=f"forward:line {node.lineno}",
                ))
        self.generic_visit(node)

    # ── helpers ──

    def _is_tensor_dependent(self, node: ast.AST) -> bool:
        """Heuristic: does the expression likely depend on a tensor?

        We exclude ``self.<attr>`` lookups (boolean flags like ``self.training``,
        ``self.aux_logits``) and ``is None`` / ``is not None`` comparisons,
        which are safe for torch.fx.
        """
        # self.training is handled by torch.fx (leaf module attribute)
        if self._is_training_guard(node):
            return False
        # self.<attr> plain attribute access is safe (boolean flags)
        if self._is_self_attr(node):
            return False
        # ``x is None`` / ``x is not None`` comparisons are safe
        if self._is_none_check(node):
            return False
        # Calls to known-safe utility functions
        if isinstance(node, ast.Call) and self._is_safe_utility_call(node):
            return False
        # UnaryOp on a non-tensor (e.g. ``not aux_defined``) is safe
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Name):
            return False
        # Calls on variables (x.sum(), x.max(), etc.) are likely tensor ops
        if isinstance(node, ast.Call):
            return True
        if isinstance(node, ast.Compare):
            for v in ast.walk(node):
                if isinstance(v, ast.Call):
                    # Skip safe utility calls inside comparisons
                    if self._is_safe_utility_call(v):
                        continue
                    return True
                if isinstance(v, ast.Attribute) and v.attr in (
                    "sum", "max", "min", "mean", "item", "numel",
                ):
                    return True
        return False

    def _is_training_guard(self, node: ast.AST) -> bool:
        """Check if this is 'self.training' — a safe pattern for torch.fx."""
        if isinstance(node, ast.Attribute):
            if (node.attr == "training" and isinstance(node.value, ast.Name)
                    and node.value.id == "self"):
                return True
        return False

    def _is_self_attr(self, node: ast.AST) -> bool:
        """Check if this is ``self.<attr>`` — a boolean attribute flag."""
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return True
        return False

    def _is_none_check(self, node: ast.AST) -> bool:
        """Check if this is ``x is None`` or ``x is not None``."""
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.Is, ast.IsNot)):
                    for comp in node.comparators:
                        if isinstance(comp, ast.Constant) and comp.value is None:
                            return True
        return False

    def _is_safe_utility_call(self, node: ast.Call) -> bool:
        """Check if a call is a known-safe utility (not tensor-dependent).

        Examples: ``torch.jit.is_scripting()``, ``isinstance()``, ``len()``.
        """
        name = self._call_name(node)
        if name in (
            "torch.jit.is_scripting", "torch.jit.is_tracing",
            "isinstance", "type", "len", "bool", "int", "float",
            "hasattr", "getattr", "print",
        ):
            return True
        return False

    def _is_data_dependent_range(self, node: ast.For) -> bool:
        """Check if a for-loop's range is data-dependent."""
        if isinstance(node.iter, ast.Call):
            func_str = self._call_name(node.iter)
            if func_str == "range":
                for arg in node.iter.args:
                    if isinstance(arg, ast.Call):
                        return True
        return False

    def _call_name(self, node: ast.Call) -> Optional[str]:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None


def _analyze_source(source: str) -> Tuple[List[UnsupportedConstruct], List[str]]:
    """Statically analyze module source for unsupported constructs."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return [], ["Could not parse source"]
    analyzer = _ForwardAnalyzer()
    analyzer.visit(tree)
    return analyzer.issues, analyzer.warnings


def analyze_source(source: str) -> List[UnsupportedConstruct]:
    """Public, instance-free fragment check over module *source*.

    Returns the list of blocking constructs (those that place the module
    OUTSIDE the verifiable fragment). An empty list means no statically
    detectable out-of-fragment construct was found.

    This is the explicit "unsupported → reported, never silently passed"
    fallback: callers (e.g. ``verify_architecture`` in ``sound`` mode) use it to
    turn a would-be silent ``SAFE`` into an honest ``UNKNOWN``. Unlike
    :func:`check_traceability`, it needs no instantiated module and performs no
    ``torch.fx`` tracing, so it is cheap and side-effect free.
    """
    issues, _warnings = _analyze_source(source)
    return [c for c in issues if c.severity == "blocking"]



# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic (torch.fx) traceability check
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_trace_error(error_msg: str) -> UnsupportedCategory:
    """Classify a torch.fx trace error into our taxonomy."""
    err = error_msg.lower()
    if "control flow" in err or "conditional" in err:
        return UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW
    if "item" in err:
        return UnsupportedCategory.TENSOR_TO_SCALAR
    if "proxy" in err or "tracer" in err:
        return UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW
    if "is not defined" in err:
        return UnsupportedCategory.OPAQUE_EXTERNAL_CALL
    if "inplace" in err or "in-place" in err:
        return UnsupportedCategory.INPLACE_MUTATION
    if "autograd" in err or "custom" in err:
        return UnsupportedCategory.CUSTOM_AUTOGRAD
    if "script" in err or "jit" in err:
        return UnsupportedCategory.JIT_SCRIPT
    if "assert" in err:
        return UnsupportedCategory.DYNAMIC_ASSERTION
    return UnsupportedCategory.OTHER


def _check_submodule_support(module: "nn.Module") -> Tuple[List[str], List[str]]:
    """Check which submodules are in the supported set."""
    supported = []
    unsupported = []
    for name, child in module.named_modules():
        if name == "":
            continue
        cls_name = type(child).__name__
        if cls_name in SUPPORTED_LAYER_TYPES:
            supported.append(f"{name} ({cls_name})")
        else:
            # Check by inheritance
            found = False
            if HAS_TORCH:
                from src.fx_extractor import _MODULE_KIND_MAP, _init_module_kind_map
                _init_module_kind_map()
                for cls in _MODULE_KIND_MAP:
                    if isinstance(child, cls):
                        found = True
                        break
            if found:
                supported.append(f"{name} ({cls_name})")
            else:
                unsupported.append(f"{name} ({cls_name})")
    return supported, unsupported


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def check_traceability(module: "nn.Module") -> TraceabilityReport:
    """Analyze a module BEFORE verification to determine if it's in the
    verifiable fragment.

    Performs both static source analysis (AST) and dynamic traceability
    testing (torch.fx.symbolic_trace).

    Parameters
    ----------
    module : nn.Module
        The module to analyze.

    Returns
    -------
    TraceabilityReport
        Detailed report on traceability, including any unsupported constructs.
    """
    if not HAS_TORCH:
        return TraceabilityReport(
            module_name="unknown",
            in_verifiable_fragment=False,
            fx_traceable=False,
            fx_trace_error="PyTorch not available",
        )

    module_name = type(module).__name__
    report = TraceabilityReport(
        module_name=module_name,
        in_verifiable_fragment=False,
        fx_traceable=False,
    )

    # Count parameters and submodules
    report.num_parameters = sum(p.numel() for p in module.parameters())
    report.num_submodules = sum(1 for _ in module.named_modules()) - 1

    # Phase 1: Check submodule support
    supported, unsupported = _check_submodule_support(module)
    report.supported_layers = supported
    report.unsupported_layers = unsupported

    # Phase 2: Static source analysis
    try:
        source = inspect.getsource(type(module))
        static_issues, static_warnings = _analyze_source(source)
        report.unsupported_constructs.extend(static_issues)
        report.static_warnings.extend(static_warnings)
    except (OSError, TypeError):
        report.static_warnings.append(
            "Could not retrieve source (dynamically defined or built-in)"
        )

    # Phase 3: Dynamic torch.fx traceability
    try:
        module.eval()
        _traced = torch.fx.symbolic_trace(module)
        report.fx_traceable = True
    except Exception as exc:
        report.fx_traceable = False
        report.fx_trace_error = str(exc)
        category = _classify_trace_error(str(exc))
        report.unsupported_constructs.append(UnsupportedConstruct(
            category=category,
            description=f"torch.fx.symbolic_trace failed: {str(exc)[:200]}",
            severity="blocking",
        ))

    # Verdict: in fragment iff traceable AND no blocking issues
    blocking = report.blocking_issues
    report.in_verifiable_fragment = report.fx_traceable and len(blocking) == 0

    return report


def extract_grammar(module: "nn.Module") -> Optional[ModuleDef]:
    """Extract the formal grammar AST for a module in the verifiable fragment.

    Returns None if the module is not traceable.
    """
    if not HAS_TORCH:
        return None

    try:
        module.eval()
        traced = torch.fx.symbolic_trace(module)
    except Exception:
        return None

    module_def = ModuleDef(name=type(module).__name__)

    # Extract layer declarations from named_modules
    for name, child in module.named_modules():
        if name == "":
            continue
        cls_name = type(child).__name__
        params: Dict[str, Any] = {}
        if hasattr(child, "in_features"):
            params["in_features"] = child.in_features
        if hasattr(child, "out_features"):
            params["out_features"] = child.out_features
        if hasattr(child, "in_channels"):
            params["in_channels"] = child.in_channels
        if hasattr(child, "out_channels"):
            params["out_channels"] = child.out_channels
        module_def.layers.append(LayerDecl(
            attr_name=name,
            layer_type=cls_name,
            params=params,
        ))

    # Extract forward-body statements from the fx graph
    for node in traced.graph.nodes:
        if node.op == "call_module":
            module_def.forward_stmts.append(SubmoduleCall(
                attr_name=str(node.target),
                args=[str(a) for a in node.args if isinstance(a, torch.fx.Node)],
            ))
        elif node.op == "call_function":
            fname = getattr(node.target, "__name__", str(node.target))
            module_def.forward_stmts.append(FunctionCall(
                function_name=fname,
                args=[str(a) for a in node.args if isinstance(a, torch.fx.Node)],
            ))
        elif node.op == "call_method":
            module_def.forward_stmts.append(MethodCall(
                method_name=str(node.target),
                args=[str(a) for a in node.args if isinstance(a, torch.fx.Node)],
            ))
        elif node.op == "output":
            module_def.forward_stmts.append(ReturnStmt(
                expr=str(node.args),
            ))

    return module_def


# ═══════════════════════════════════════════════════════════════════════════════
# Formal specification document (Step 8)
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical grammar of the verifiable fragment V_TG. Single source of truth for
# the generated spec document (see ``render_spec_markdown``). Kept in sync with
# the module docstring and the SUPPORTED_* tables below.
VERIFIABLE_FRAGMENT_GRAMMAR = """\
Module      ::=  class C(nn.Module):
                     __init__(self, <params>): <InitBody>
                     forward(self, <inputs>): <FwdBody>

InitBody    ::=  (self.<attr> = <LayerExpr>)*
LayerExpr   ::=  nn.<SupportedLayer>(<literal>*)
              |  nn.Sequential(<LayerExpr>*)
              |  nn.ModuleList([<LayerExpr>*])

FwdBody     ::=  <Stmt>*; return <Expr>

Stmt        ::=  <var> = <Expr>
              |  <var> = self.<attr>(<Expr>*)          # submodule call
              |  for <var> in self.<modulelist>:       # static iteration
                     <Stmt>*

Expr        ::=  <var>
              |  self.<attr>(<Expr>*)                  # nn.Module call
              |  <SupportedFunc>(<Expr>*)              # torch.* / F.*
              |  <Expr>.<SupportedMethod>(<literal>*)  # tensor method
              |  <Expr> <BinOp> <Expr>                 # +, *, @
              |  <literal>
"""

# Human-readable description + detection mechanism for every out-of-fragment
# category. ``detected_by`` is one of "static" (AST source scan, instance-free),
# "fx" (torch.fx trace error classification), or "static+fx".
UNSUPPORTED_CATEGORY_INFO: Dict[UnsupportedCategory, Dict[str, str]] = {
    UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW: {
        "description": "Branch (if/while) whose condition depends on a tensor value.",
        "detected_by": "static+fx",
    },
    UnsupportedCategory.DATA_DEPENDENT_ITERATION: {
        "description": "Loop whose trip count depends on runtime data (e.g. range(int(x.item()))).",
        "detected_by": "static+fx",
    },
    UnsupportedCategory.DYNAMIC_ASSERTION: {
        "description": "assert statement in forward (may reference tensor values).",
        "detected_by": "static+fx",
    },
    UnsupportedCategory.TENSOR_TO_SCALAR: {
        "description": ".item() / .tolist() / .numpy() converts a tensor to a Python value.",
        "detected_by": "static+fx",
    },
    UnsupportedCategory.CUSTOM_AUTOGRAD: {
        "description": "Custom torch.autograd.Function subclass with opaque shape behaviour.",
        "detected_by": "fx",
    },
    UnsupportedCategory.INPLACE_MUTATION: {
        "description": "In-place mutation that torch.fx cannot trace soundly.",
        "detected_by": "fx",
    },
    UnsupportedCategory.JIT_SCRIPT: {
        "description": "torch.jit.script / scripted submodule opaque to torch.fx.",
        "detected_by": "fx",
    },
    UnsupportedCategory.OPAQUE_EXTERNAL_CALL: {
        "description": "Call into an external/undefined symbol opaque to the tracer.",
        "detected_by": "fx",
    },
    UnsupportedCategory.DYNAMIC_MODULE_CONSTRUCTION: {
        "description": "Submodules constructed dynamically at forward time.",
        "detected_by": "fx",
    },
    UnsupportedCategory.UNSUPPORTED_BUILTIN: {
        "description": "Python builtin not modelled by the shape semantics.",
        "detected_by": "fx",
    },
    UnsupportedCategory.OTHER: {
        "description": "Any other torch.fx trace failure not otherwise classified.",
        "detected_by": "fx",
    },
}


def _md_table_block(title: str, items: Set[str]) -> str:
    ordered = sorted(items)
    lines = [f"### {title} ({len(ordered)})", ""]
    lines.append("```")
    # wrap at a reasonable width for readability
    row: List[str] = []
    width = 0
    for it in ordered:
        if width + len(it) + 2 > 76 and row:
            lines.append(", ".join(row) + ",")
            row, width = [], 0
        row.append(it)
        width += len(it) + 2
    if row:
        lines.append(", ".join(row))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_spec_markdown() -> str:
    """Render the formal verifiable-fragment specification as Markdown.

    Generated from this module's single source of truth (the grammar constant,
    the SUPPORTED_* tables and the UnsupportedCategory taxonomy), so the doc can
    never silently drift from the code. Regenerate with::

        python -m src.verifiable_fragment > VERIFIABLE_FRAGMENT.md
    """
    static_cats = sorted(
        c.name for c, info in UNSUPPORTED_CATEGORY_INFO.items()
        if "static" in info["detected_by"]
    )
    parts: List[str] = []
    parts.append("# TensorGuard Verifiable Fragment (V_TG)")
    parts.append("")
    parts.append(
        "> **Generated file — do not edit by hand.** Regenerate with "
        "`python -m src.verifiable_fragment > VERIFIABLE_FRAGMENT.md`. "
        "Single source of truth: `src/verifiable_fragment.py`."
    )
    parts.append("")
    parts.append(
        "This document formally characterizes the subset of PyTorch "
        "`nn.Module` code that TensorGuard can analyze. A module inside the "
        "fragment is amenable to sound shape/device/gradient verification; a "
        "module outside it is **reported as `UNKNOWN`, never silently passed** "
        "(see the fallback policy below)."
    )
    parts.append("")
    parts.append("## Grammar")
    parts.append("")
    parts.append("```")
    parts.append(VERIFIABLE_FRAGMENT_GRAMMAR.rstrip())
    parts.append("```")
    parts.append("")
    parts.append("## Supported constructs")
    parts.append("")
    parts.append(_md_table_block("Layer types", SUPPORTED_LAYER_TYPES))
    parts.append(_md_table_block("Tensor methods", SUPPORTED_TENSOR_METHODS))
    parts.append(_md_table_block("torch.* functions", SUPPORTED_TORCH_FUNCTIONS))
    parts.append(_md_table_block("torch.nn.functional (F.*) functions", SUPPORTED_F_FUNCTIONS))
    parts.append("## Excluded constructs (outside V_TG)")
    parts.append("")
    parts.append("| Category | Description | Detected by |")
    parts.append("| --- | --- | --- |")
    for cat in UnsupportedCategory:
        info = UNSUPPORTED_CATEGORY_INFO.get(cat)
        if not info:
            continue
        parts.append(f"| `{cat.name}` | {info['description']} | {info['detected_by']} |")
    parts.append("")
    parts.append("*Detected by:* `static` = instance-free AST scan "
                 "(`analyze_source`); `fx` = `torch.fx` trace-error "
                 "classification during `check_traceability`; `static+fx` = both.")
    parts.append("")
    parts.append("## Fallback policy: unsupported → `UNKNOWN`, never a silent pass")
    parts.append("")
    parts.append(
        "When a module contains any construct above, TensorGuard does **not** "
        "emit a confident `SAFE`. Two complementary mechanisms enforce this:"
    )
    parts.append("")
    parts.append(
        "1. **Pre-verification** — `check_traceability(module)` returns "
        "`in_verifiable_fragment=False` with the offending "
        "`UnsupportedConstruct`s. The instance-free `analyze_source(source)` "
        "exposes the statically detectable subset "
        f"({', '.join('`' + c + '`' for c in static_cats)})."
    )
    parts.append(
        "2. **During verification** — in `--soundness-mode sound`, "
        "`verify_architecture` folds these signals (plus opaque "
        "out-of-fragment layers and heuristic-tagged operators) into "
        "abstention, yielding `verdict=UNKNOWN` with `unknown_reasons` rather "
        "than a silent `SAFE`. See `SOUNDNESS_CONTRACT.md`."
    )
    parts.append("")
    return "\n".join(parts) + "\n"


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys
    _sys.stdout.write(render_spec_markdown())
