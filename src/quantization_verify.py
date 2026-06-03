"""Instance-level quantization placement gates for PyTorch modules.

The source-level ``quant_export_checks`` analyzer catches syntactic hazards
before importing a model.  This module complements it with checks over real
prepared/converted modules and ``torch.fx`` graphs: observer calibration,
activation qscheme placement, quant/dequant boundaries, and quantized graph
dataflow.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "QuantizationIssue",
    "QuantizationVerdict",
    "verify_quantization",
    "verify_quantization_eager",
    "verify_quantization_fx",
]

_FLOAT = "float"
_QUANTIZED = "quantized"
_UNKNOWN = "unknown"
_State = str


@dataclass(frozen=True)
class QuantizationIssue:
    """One statically visible quantization contract violation."""

    kind: str
    message: str
    location: str = "module"


@dataclass(frozen=True)
class QuantizationVerdict:
    """Result of checking eager or FX quantization metadata."""

    ok: bool
    issues: Tuple[QuantizationIssue, ...] = ()
    warnings: Tuple[str, ...] = ()
    mode: str = "eager"

    def has_issue(self, kind: str) -> bool:
        return any(issue.kind == kind for issue in self.issues)


def verify_quantization(
    model: Any,
    *,
    mode: str = "auto",
    require_calibrated: bool = True,
    require_stub_boundaries: bool = True,
    require_float_output: bool = True,
    quantized_inputs: Optional[Iterable[str]] = None,
) -> QuantizationVerdict:
    """Check quantization metadata on an eager module or ``torch.fx`` graph.

    ``mode="auto"`` dispatches to FX graph analysis when ``model`` has a
    ``graph`` attribute, otherwise to eager-module checks.
    """

    if mode not in {"auto", "eager", "fx"}:
        raise ValueError("mode must be one of 'auto', 'eager', or 'fx'")
    if mode == "fx" or (mode == "auto" and hasattr(model, "graph")):
        return verify_quantization_fx(
            model,
            require_calibrated=require_calibrated,
            require_float_output=require_float_output,
            quantized_inputs=quantized_inputs,
        )
    return verify_quantization_eager(
        model,
        require_calibrated=require_calibrated,
        require_stub_boundaries=require_stub_boundaries,
        require_float_output=require_float_output,
    )


def verify_quantization_eager(
    model: Any,
    *,
    require_calibrated: bool = True,
    require_stub_boundaries: bool = True,
    require_float_output: bool = True,
) -> QuantizationVerdict:
    """Check eager-mode prepared or converted quantization placement.

    Prepared modules are checked for observer state and qscheme placement.
    Converted modules are checked for the quant/dequant boundaries required by
    quantized kernels that still receive normal float inputs at the public API.
    """

    issues: List[QuantizationIssue] = []
    warnings: List[str] = []
    issues.extend(_module_metadata_issues(model, require_calibrated=require_calibrated))

    modules = tuple(_named_modules(model))
    has_quantized_compute = any(_is_quantized_compute_module(mod) for _, mod in modules)
    has_quantize = any(_is_quantize_module(mod) for _, mod in modules)
    has_dequantize = any(_is_dequantize_module(mod) for _, mod in modules)

    if require_stub_boundaries and has_quantized_compute:
        if not has_quantize:
            issues.append(
                QuantizationIssue(
                    "missing_quantstub",
                    "converted quantized module has no Quantize/QuantStub entry boundary; "
                    "float inputs will reach quantized kernels",
                )
            )
        if require_float_output and not has_dequantize:
            issues.append(
                QuantizationIssue(
                    "missing_dequantstub",
                    "converted quantized module has no DeQuantize/DeQuantStub exit boundary; "
                    "quantized tensors escape the module output",
                )
            )

    if require_stub_boundaries and has_quantized_compute:
        try:
            from torch.fx import symbolic_trace  # type: ignore

            graph_verdict = _verify_fx_graph(
                symbolic_trace(model),
                require_float_output=require_float_output,
                quantized_inputs=None,
                include_module_metadata=False,
            )
            issues.extend(graph_verdict.issues)
            warnings.extend(graph_verdict.warnings)
        except Exception as exc:  # pragma: no cover - version/model dependent
            warnings.append(f"fx placement refinement skipped: {type(exc).__name__}: {exc}")

    final_issues = _dedupe_issues(issues)
    return QuantizationVerdict(not final_issues, final_issues, tuple(warnings), "eager")


def verify_quantization_fx(
    graph_module: Any,
    *,
    require_calibrated: bool = True,
    require_float_output: bool = True,
    quantized_inputs: Optional[Iterable[str]] = None,
) -> QuantizationVerdict:
    """Check an FX graph for quantization boundary/dataflow mistakes."""

    return _verify_fx_graph(
        graph_module,
        require_float_output=require_float_output,
        quantized_inputs=quantized_inputs,
        include_module_metadata=True,
        require_calibrated=require_calibrated,
    )


def _verify_fx_graph(
    graph_module: Any,
    *,
    require_float_output: bool,
    quantized_inputs: Optional[Iterable[str]],
    include_module_metadata: bool,
    require_calibrated: bool = True,
) -> QuantizationVerdict:
    issues: List[QuantizationIssue] = []
    warnings: List[str] = []
    if include_module_metadata:
        issues.extend(_module_metadata_issues(graph_module, require_calibrated=require_calibrated))

    q_inputs = set(quantized_inputs or ())
    states: Dict[Any, _State] = {}

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            states[node] = _QUANTIZED if str(node.target) in q_inputs or node.name in q_inputs else _FLOAT
            continue

        if node.op == "get_attr":
            states[node] = _UNKNOWN
            continue

        if node.op == "call_module":
            submod = graph_module.get_submodule(str(node.target))
            input_state = _first_tensor_state(node.args, states)
            location = f"fx:{node.name}"
            if _is_quantize_module(submod):
                issues.extend(_quantize_qparam_issues(submod, location))
                states[node] = _QUANTIZED
            elif _is_dequantize_module(submod):
                if input_state == _FLOAT:
                    issues.append(
                        QuantizationIssue(
                            "dequantize_float_input",
                            "DeQuantize received a float tensor; the graph is missing an upstream quantized value",
                            location,
                        )
                    )
                states[node] = _FLOAT
            elif _is_quantized_compute_module(submod):
                if input_state == _FLOAT:
                    issues.append(
                        QuantizationIssue(
                            "missing_quantstub",
                            f"quantized module {node.target!s} receives a float tensor; "
                            "insert Quantize/torch.quantize_per_tensor before it",
                            location,
                        )
                    )
                states[node] = _QUANTIZED
            else:
                states[node] = input_state
            continue

        if node.op == "call_method":
            target = str(node.target)
            input_state = _first_tensor_state(node.args, states)
            location = f"fx:{node.name}"
            if target == "dequantize":
                if input_state == _FLOAT:
                    issues.append(
                        QuantizationIssue(
                            "dequantize_float_input",
                            "Tensor.dequantize() received a float tensor",
                            location,
                        )
                    )
                states[node] = _FLOAT
            else:
                states[node] = input_state
            continue

        if node.op == "call_function":
            location = f"fx:{node.name}"
            target_name = _target_name(node.target)
            if _is_quantize_function(node.target):
                issues.extend(_quantize_function_issues(node.args, node.kwargs, location))
                states[node] = _QUANTIZED
            elif target_name.endswith(".dequantize") or target_name == "dequantize":
                states[node] = _FLOAT
            elif _is_quantized_function(node.target):
                if _first_tensor_state(node.args, states) == _FLOAT:
                    issues.append(
                        QuantizationIssue(
                            "missing_quantstub",
                            f"quantized function {target_name} receives a float tensor",
                            location,
                        )
                    )
                states[node] = _QUANTIZED
            elif node.target in {operator.add, operator.sub, operator.mul, operator.truediv}:
                if _contains_state(node.args, states, _QUANTIZED):
                    issues.append(
                        QuantizationIssue(
                            "quantized_arithmetic_without_floatfunctional",
                            "raw tensor arithmetic on quantized tensors has no general QuantizedCPU kernel; "
                            "use torch.ao.nn.quantized.FloatFunctional or dequantize first",
                            location,
                        )
                    )
                states[node] = _first_tensor_state(node.args, states)
            else:
                states[node] = _first_tensor_state(node.args, states)
            continue

        if node.op == "output":
            output_state = _state_of(node.args[0] if node.args else None, states)
            if require_float_output and output_state == _QUANTIZED:
                issues.append(
                    QuantizationIssue(
                        "missing_dequantstub",
                        "FX graph returns a quantized tensor where a float public output was required",
                        f"fx:{node.name}",
                    )
                )
            states[node] = output_state

    final_issues = _dedupe_issues(issues)
    return QuantizationVerdict(not final_issues, final_issues, tuple(warnings), "fx")


def _module_metadata_issues(model: Any, *, require_calibrated: bool) -> List[QuantizationIssue]:
    issues: List[QuantizationIssue] = []
    for name, module in _named_modules(model):
        location = name or "module"
        if _is_observer_or_fake_quant(module):
            qscheme = getattr(module, "qscheme", None)
            if _is_activation_observer_name(name) and _is_per_channel_qscheme(qscheme):
                issues.append(
                    QuantizationIssue(
                        "qscheme_placement",
                        f"activation observer uses per-channel qscheme {qscheme}; "
                        "PyTorch activation quantization expects per-tensor qparams",
                        location,
                    )
                )
            if require_calibrated and _observer_is_uncalibrated(module):
                issues.append(
                    QuantizationIssue(
                        "calibration_state",
                        "observer has not seen calibration data (min/max are still initial or empty)",
                        location,
                    )
                )
        qconfig = getattr(module, "qconfig", None)
        activation_factory = getattr(qconfig, "activation", None)
        if activation_factory is not None:
            qscheme = _factory_qscheme(activation_factory)
            if _is_per_channel_qscheme(qscheme):
                issues.append(
                    QuantizationIssue(
                        "qscheme_placement",
                        f"qconfig activation observer uses per-channel qscheme {qscheme}; "
                        "activation observers must be per-tensor",
                        location,
                    )
                )
    return _dedupe_issues(issues)


def _named_modules(model: Any) -> Iterable[Tuple[str, Any]]:
    named = getattr(model, "named_modules", None)
    if named is None:
        return (("", model),)
    return named()


def _dedupe_issues(issues: Sequence[QuantizationIssue]) -> Tuple[QuantizationIssue, ...]:
    seen = set()
    out: List[QuantizationIssue] = []
    for issue in issues:
        key = (issue.kind, issue.location, issue.message)
        if key not in seen:
            seen.add(key)
            out.append(issue)
    return tuple(out)


def _type_module(obj: Any) -> str:
    return type(obj).__module__


def _type_name(obj: Any) -> str:
    return type(obj).__qualname__


def _is_quantize_module(module: Any) -> bool:
    return _type_name(module) == "Quantize" and "quantized" in _type_module(module)


def _is_dequantize_module(module: Any) -> bool:
    return _type_name(module) == "DeQuantize" and "quantized" in _type_module(module)


def _is_quantized_compute_module(module: Any) -> bool:
    namespace = _type_module(module)
    if not (
        namespace.startswith("torch.ao.nn.quantized")
        or namespace.startswith("torch.nn.quantized")
        or ".quantized." in namespace
    ):
        return False
    if _is_quantize_module(module) or _is_dequantize_module(module):
        return False
    name = _type_name(module)
    return name not in {"FloatFunctional", "QFunctional", "LinearPackedParams"}


def _is_observer_or_fake_quant(module: Any) -> bool:
    namespace = _type_module(module)
    name = _type_name(module)
    if namespace.startswith("torch.ao.quantization.observer"):
        return True
    if namespace.startswith("torch.ao.quantization.fake_quantize"):
        return True
    return name.endswith("Observer") or "FakeQuantize" in name


def _is_activation_observer_name(name: str) -> bool:
    lowered = name.lower()
    return not any(token in lowered for token in ("weight_fake_quant", "weight_observer", "weight_post_process"))


def _factory_qscheme(factory: Any) -> Optional[Any]:
    try:
        observer = factory()
    except Exception:
        return None
    return getattr(observer, "qscheme", None)


def _is_per_channel_qscheme(qscheme: Any) -> bool:
    return qscheme is not None and "per_channel" in str(qscheme)


def _observer_is_uncalibrated(module: Any) -> bool:
    min_val = getattr(module, "min_val", None)
    max_val = getattr(module, "max_val", None)
    if min_val is None and max_val is None:
        return False
    if _is_empty(min_val) or _is_empty(max_val):
        return True
    if _all_inf(min_val, positive=True) or _all_inf(max_val, positive=False):
        return True
    if _any_nonfinite(min_val) or _any_nonfinite(max_val):
        return True
    return False


def _is_empty(value: Any) -> bool:
    try:
        return value is not None and hasattr(value, "numel") and int(value.numel()) == 0
    except Exception:
        return False


def _all_inf(value: Any, *, positive: bool) -> bool:
    if value is None:
        return False
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel") and int(value.numel()) == 0:
            return True
        if hasattr(value, "isinf"):
            mask = value.isinf()
            sign = value > 0 if positive else value < 0
            return bool((mask & sign).all().item())
        scalar = float(value)
        return math.isinf(scalar) and ((scalar > 0) == positive)
    except Exception:
        return False


def _any_nonfinite(value: Any) -> bool:
    if value is None:
        return False
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel") and int(value.numel()) == 0:
            return True
        if hasattr(value, "isfinite"):
            return not bool(value.isfinite().all().item())
        return not math.isfinite(float(value))
    except Exception:
        return False


def _quantize_qparam_issues(module: Any, location: str) -> List[QuantizationIssue]:
    issues: List[QuantizationIssue] = []
    scale = getattr(module, "scale", None)
    if scale is not None and _known_nonpositive(scale):
        issues.append(QuantizationIssue("qparams", "Quantize scale must be positive", location))
    return issues


def _quantize_function_issues(args: Sequence[Any], kwargs: Dict[str, Any], location: str) -> Tuple[QuantizationIssue, ...]:
    scale = kwargs.get("scale")
    if scale is None and len(args) >= 2:
        scale = args[1]
    if scale is not None and _known_nonpositive(scale):
        return (QuantizationIssue("qparams", "torch.quantize_per_tensor scale must be positive", location),)
    return []


def _known_nonpositive(value: Any) -> bool:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel"):
            if int(value.numel()) == 0:
                return False
            return bool((value <= 0).any().item())
        return float(value) <= 0
    except Exception:
        return False


def _target_name(target: Any) -> str:
    module = getattr(target, "__module__", "")
    name = getattr(target, "__name__", repr(target))
    return f"{module}.{name}" if module else name


def _is_quantize_function(target: Any) -> bool:
    name = _target_name(target)
    return name.endswith("quantize_per_tensor") or name.endswith("quantize_per_channel")


def _is_quantized_function(target: Any) -> bool:
    name = _target_name(target)
    return "quantized" in name or name.startswith("quantized::")


def _state_of(obj: Any, states: Dict[Any, _State]) -> _State:
    if isinstance(obj, (tuple, list)):
        child_states = [_state_of(item, states) for item in obj]
        if not child_states:
            return _UNKNOWN
        if all(state == _FLOAT for state in child_states):
            return _FLOAT
        if all(state == _QUANTIZED for state in child_states):
            return _QUANTIZED
        if any(state == _QUANTIZED for state in child_states):
            return _QUANTIZED
        return _UNKNOWN
    if isinstance(obj, dict):
        return _state_of(tuple(obj.values()), states)
    try:
        if obj in states:
            return states[obj]
    except TypeError:
        return _UNKNOWN
    return _UNKNOWN


def _first_tensor_state(args: Sequence[Any], states: Dict[Any, _State]) -> _State:
    for arg in args:
        state = _state_of(arg, states)
        if state != _UNKNOWN:
            return state
    return _UNKNOWN


def _contains_state(args: Sequence[Any], states: Dict[Any, _State], expected: _State) -> bool:
    return _state_of(tuple(args), states) == expected or any(_state_of(arg, states) == expected for arg in args)
