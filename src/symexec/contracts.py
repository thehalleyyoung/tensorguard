"""Inferred shape contracts → jaxtyping ``.pyi`` stubs (even_more Tier 3 #8).

TensorGuard already *consumes* jaxtyping/torchtyping annotations to seed an
analysis (``_infer_from_annotation``).  This module closes the loop: it
*produces* shape contracts the rest of the ecosystem can consume.

For each top-level function and each ``forward``/``__call__`` method it runs the
symbolic executor with annotation-seeded parameters (the same sound seeding the
engine uses) and reads off two things:

* the **input contract** assumed for each tensor parameter — exactly the
  abstraction derived from its annotation (so the emitted stub never claims more
  about an input than the annotation already guaranteed); and
* the **output contract** — the rank/shape/dtype abstraction the analysis
  *derived* for the function's joined return value.

These are rendered as jaxtyping annotations (``Float[Tensor, "b c h w"]``) in a
``.pyi`` stub.  The contract is *advisory and inferential*: it summarises facts
the analysis assumed (inputs) or proved (outputs); a dimension is only given a
concrete size or a shared symbolic name when the analysis tracked it, otherwise
it gets an anonymous per-axis name.  Unknown-rank tensors degrade to a plain
``Tensor`` and non-tensor results are annotated only when their type is known.

The module is torch-free and pure: it only constructs an interpreter and reads
already-computed abstractions; it never emits a diagnostic.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .interpreter import Interpreter, _infer_from_annotation
from .values import AbstractValue, NoneVal, TensorVal
from .symdim import SymDim

__all__ = [
    "TensorSpec",
    "ParamContract",
    "FunctionContract",
    "infer_contracts",
    "to_pyi",
    "contracts_to_pyi",
]


# Map a recorded dtype leaf name to a jaxtyping dtype class.  Unknown dtypes use
# jaxtyping's any-dtype ``Shaped`` so the contract stays honest.
_FLOAT_DTYPES = frozenset({
    "float16", "float32", "float64", "half", "float", "double", "bfloat16",
})
_INT_DTYPES = frozenset({
    "uint8", "int8", "int16", "int32", "int64", "long", "short", "int",
    "uint16", "uint32", "uint64", "byte", "char",
})
_COMPLEX_DTYPES = frozenset({"complex64", "complex128", "cfloat", "cdouble"})


def _jaxtyping_dtype(dtype: Optional[str]) -> str:
    if dtype is None:
        return "Shaped"
    if dtype in _FLOAT_DTYPES:
        return "Float"
    if dtype in _INT_DTYPES:
        return "Int"
    if dtype in _COMPLEX_DTYPES:
        return "Complex"
    if dtype == "bool":
        return "Bool"
    return "Shaped"


@dataclass(frozen=True)
class TensorSpec:
    """A rendered tensor shape contract: a jaxtyping dtype class, an optional
    rank, and per-axis tokens (a concrete size, a shared symbolic-dim name, or an
    anonymous ``d{i}`` axis)."""

    dtype_class: str
    rank: Optional[int]
    axes: Optional[Tuple[str, ...]]

    def render(self) -> str:
        if self.rank is None or self.axes is None:
            return "Tensor"
        spec = " ".join(self.axes)
        return f"{self.dtype_class}[Tensor, \"{spec}\"]"


@dataclass(frozen=True)
class ParamContract:
    name: str
    spec: Optional[TensorSpec]          # tensor contract, when inferable
    scalar_type: Optional[str] = None   # 'int'/'float'/'bool'/'str' when known

    def render(self) -> str:
        if self.spec is not None:
            return f"{self.name}: {self.spec.render()}"
        if self.scalar_type is not None:
            return f"{self.name}: {self.scalar_type}"
        return self.name


@dataclass
class FunctionContract:
    name: str
    params: List[ParamContract] = field(default_factory=list)
    ret: Optional[TensorSpec] = None
    ret_is_none: bool = False
    class_name: Optional[str] = None    # set for methods (forward/__call__)
    is_method: bool = False

    def render_def(self, indent: str = "") -> str:
        head = "self, " if self.is_method else ""
        args = ", ".join(p.render() for p in self.params)
        sig = head + args
        if self.ret is not None:
            ret = " -> " + self.ret.render()
        elif self.ret_is_none:
            ret = " -> None"
        else:
            ret = ""
        return f"{indent}def {self.name}({sig}){ret}: ..."


def _axis_token(d: Optional[SymDim], i: int) -> str:
    """Render one shape axis.  Constant dims become their integer; a *pure*
    single-variable dim becomes that variable's name (so equal symbolic axes
    across inputs/outputs share a name); everything else is an anonymous axis."""
    if d is None:
        return f"d{i}"
    if d.is_const and d.value is not None:
        return str(d.value)
    if d.const == 0 and len(d.terms) == 1 and d.terms[0][1] == 1:
        return d.terms[0][0]
    return f"d{i}"


def _spec_from_value(v: AbstractValue) -> Optional[TensorSpec]:
    if not isinstance(v, TensorVal):
        return None
    dtype_class = _jaxtyping_dtype(v.dtype)
    if v.rank is None:
        return TensorSpec(dtype_class=dtype_class, rank=None, axes=None)
    if v.shape is None:
        axes = tuple(f"d{i}" for i in range(v.rank))
    else:
        axes = tuple(_axis_token(v.shape[i] if i < len(v.shape) else None, i)
                     for i in range(v.rank))
    return TensorSpec(dtype_class=dtype_class, rank=v.rank, axes=axes)


def _scalar_type(ann) -> Optional[str]:
    if isinstance(ann, ast.Name) and ann.id in ("int", "float", "bool", "str"):
        return ann.id
    return None


# jaxtyping dtype-class names we can echo straight back from an input annotation.
_JAXTYPING_CLASS_NAMES = frozenset({
    "Float", "Int", "Bool", "Complex", "Shaped", "Num", "Inexact", "Integer",
    "Float32", "Float64", "Float16", "BFloat16", "Int8", "Int16", "Int32",
    "Int64", "UInt8", "UInt", "Key",
})


def _annotation_dtype_class(ann) -> Optional[str]:
    """If ``ann`` is a jaxtyping subscript (``Float[Tensor, "b c"]``), return the
    base dtype-class name so an echoed input contract keeps the user's declared
    dtype rather than degrading to ``Shaped``."""
    if not isinstance(ann, ast.Subscript):
        return None
    base = ann.value
    bname = (
        base.id if isinstance(base, ast.Name)
        else (base.attr if isinstance(base, ast.Attribute) else None)
    )
    if bname in _JAXTYPING_CLASS_NAMES:
        # Normalise the precise-width aliases to a renderable base class.
        if bname.startswith("Float") or bname == "BFloat16":
            return "Float"
        if bname.startswith("Int") or bname.startswith("UInt"):
            return "Int"
        return bname
    return None


def _param_contracts(func: ast.FunctionDef, *, skip_self: bool) -> List[ParamContract]:
    out: List[ParamContract] = []
    arg_objs = list(func.args.args)
    if skip_self and arg_objs and arg_objs[0].arg == "self":
        arg_objs = arg_objs[1:]
    for a in arg_objs:
        seeded = _infer_from_annotation(a.annotation)
        spec = _spec_from_value(seeded) if isinstance(seeded, TensorVal) else None
        # Preserve the declared jaxtyping dtype class on the echoed input.
        if spec is not None and spec.rank is not None:
            declared = _annotation_dtype_class(a.annotation)
            if declared is not None and declared != spec.dtype_class:
                spec = TensorSpec(
                    dtype_class=declared, rank=spec.rank, axes=spec.axes
                )
        out.append(ParamContract(
            name=a.arg, spec=spec, scalar_type=_scalar_type(a.annotation)
        ))
    return out


def infer_contracts(
    source: str, *, config: "object | None" = None, filename: str = "<contracts>"
) -> List[FunctionContract]:
    """Infer shape contracts for every top-level function and ``forward`` /
    ``__call__`` method in ``source``.  Returns an empty list on a syntax error.
    """
    try:
        module = ast.parse(source)
    except SyntaxError:
        return []
    interp = Interpreter(module, filename=filename, config=config)

    contracts: List[FunctionContract] = []

    # Top-level free functions.
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            try:
                ret = interp.run_function(node, args={}, self_val=None)
            except Exception:
                ret = None
            contracts.append(_build_contract(node, ret, class_name=None,
                                              is_method=False))

    # forward / __call__ methods of every class (mirrors engine pass 3 seeding).
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for mname in ("forward", "__call__"):
            method = next(
                (m for m in node.body
                 if isinstance(m, ast.FunctionDef) and m.name == mname),
                None,
            )
            if method is None:
                continue
            try:
                self_val = interp._instantiate(node, [], {}, None)
                seed: dict = {}
                posargs = [a for a in method.args.args if a.arg != "self"]
                if mname == "forward" and posargs and posargs[0].annotation is None:
                    seed[posargs[0].arg] = TensorVal(rank=None)
                ret = interp.run_function(method, args=seed, self_val=self_val)
            except Exception:
                ret = None
            contracts.append(_build_contract(method, ret, class_name=node.name,
                                              is_method=True))
            break  # forward takes precedence over __call__
    return contracts


def _build_contract(
    func: ast.FunctionDef, ret: Optional[AbstractValue], *,
    class_name: Optional[str], is_method: bool,
) -> FunctionContract:
    params = _param_contracts(func, skip_self=is_method)
    spec = _spec_from_value(ret) if isinstance(ret, TensorVal) else None
    return FunctionContract(
        name=func.name,
        params=params,
        ret=spec,
        ret_is_none=isinstance(ret, NoneVal),
        class_name=class_name,
        is_method=is_method,
    )


def _used_dtype_classes(contracts: List[FunctionContract]) -> List[str]:
    used = set()
    for c in contracts:
        for p in c.params:
            if p.spec is not None and p.spec.rank is not None:
                used.add(p.spec.dtype_class)
        if c.ret is not None and c.ret.rank is not None:
            used.add(c.ret.dtype_class)
    return sorted(used)


def _needs_tensor_import(contracts: List[FunctionContract]) -> bool:
    for c in contracts:
        if c.ret is not None:
            return True
        for p in c.params:
            if p.spec is not None:
                return True
    return False


def to_pyi(contracts: List[FunctionContract]) -> str:
    """Render inferred contracts as a jaxtyping ``.pyi`` stub."""
    dtypes = _used_dtype_classes(contracts)
    lines: List[str] = [
        "# Auto-generated shape contracts (TensorGuard). Do not edit by hand.",
    ]
    if dtypes:
        lines.append(f"from jaxtyping import {', '.join(dtypes)}")
        lines.append("from torch import Tensor")
    elif _needs_tensor_import(contracts):
        lines.append("from torch import Tensor")
    lines.append("")

    free = [c for c in contracts if not c.is_method]
    for c in free:
        lines.append(c.render_def())

    methods_by_class: dict = {}
    for c in contracts:
        if c.is_method:
            methods_by_class.setdefault(c.class_name, []).append(c)
    first_block = not free
    for cls_name, methods in methods_by_class.items():
        if not first_block:
            lines.append("")
        first_block = False
        lines.append(f"class {cls_name}:")
        for m in methods:
            lines.append(m.render_def(indent="    "))

    return "\n".join(lines).rstrip() + "\n"


def contracts_to_pyi(source: str, *, config: "object | None" = None) -> str:
    """Convenience: infer contracts from ``source`` and render the ``.pyi``."""
    return to_pyi(infer_contracts(source, config=config))
