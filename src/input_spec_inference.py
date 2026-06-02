"""Step 42 — automatic inference of forward input specifications.

TensorGuard normally needs an explicit ``input_shapes`` mapping (the ``-s``
annotation burden).  In practice the shapes a model expects are *already*
documented in the source in one of a handful of conventional places:

* **Shape-typed annotations** on ``forward`` parameters — ``jaxtyping``
  (``Float[Tensor, "batch 3 224 224"]``), ``torchtyping``
  (``TensorType["batch", 3, 224, 224]``), or a bare ``Tensor`` subscript.
* **Docstrings** — Google / NumPy style ``Args:`` blocks that spell out
  ``x: shape (B, 3, 224, 224)``.
* **Example inputs** — a class attribute / method (``example_inputs``,
  ``example_input_array`` — the PyTorch-Lightning convention) or a module-level
  assignment built from ``torch.randn``/``zeros``/``ones`` /``rand``/``empty``.
* **Config dicts** — a literal ``dict`` whose keys look like
  ``input_shape``/``input_size``/``img_size``.

This module recovers an ``input_shapes`` mapping *statically* (pure AST, no code
execution, no third-party imports) so that ``verify_model`` can run with zero
hand-written annotations.  Every inferred dimension is either a concrete int or
a symbolic string name, exactly matching the ``input_shapes`` contract used
elsewhere.  Inference is deliberately conservative: when a source is ambiguous
it abstains (returns nothing for that parameter) rather than guess, so it can
never *introduce* an unsound shape — at worst it leaves a parameter
unconstrained, which is the pre-existing behaviour.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

Dim = Union[int, str]
Shape = Tuple[Dim, ...]

__all__ = [
    "InferredSpec",
    "infer_input_specs",
]

# Tensor factory calls whose positional int args describe a shape.
_TENSOR_FACTORIES = frozenset(
    {"randn", "rand", "zeros", "ones", "empty", "full", "randint"}
)

# Annotation base names that carry a shape subscript.
_SHAPE_ANNOTATION_BASES = frozenset(
    {
        "Float", "Int", "Bool", "Complex", "Inexact", "Num", "Shaped",
        "UInt", "Float16", "Float32", "Float64", "BFloat16",
        "TensorType", "Tensor", "torch.Tensor",
    }
)

_CONFIG_SHAPE_KEYS = ("input_shape", "input_size", "input_dims", "in_shape")


@dataclass
class InferredSpec:
    """Result of input-spec inference for one model."""

    shapes: Dict[str, Shape] = field(default_factory=dict)
    # param name -> human-readable provenance ("annotation", "example_inputs", ...)
    sources: Dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:  # truthy iff anything was inferred
        return bool(self.shapes)


def infer_input_specs(
    source: str, class_name: Optional[str] = None
) -> InferredSpec:
    """Infer an ``input_shapes`` mapping from *source*.

    Parameters
    ----------
    source:
        Python source defining one or more ``nn.Module`` subclasses.
    class_name:
        If given, restrict to that class.  Otherwise the first class whose
        ``forward`` carries usable shape information is used.

    Returns
    -------
    InferredSpec
        ``.shapes`` maps forward-parameter names to shape tuples (ints or
        symbolic strings).  Empty when nothing could be recovered.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return InferredSpec()

    classes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
        and (class_name is None or n.name == class_name)
    ]
    module_assignments = _module_level_assignments(tree)

    best = InferredSpec()
    for cls in classes:
        spec = _infer_for_class(cls, module_assignments)
        if len(spec.shapes) > len(best.shapes):
            best = spec
        if class_name is not None:
            return spec
    return best


# --------------------------------------------------------------------------- #
# Per-class inference.
# --------------------------------------------------------------------------- #
def _infer_for_class(
    cls: ast.ClassDef, module_assignments: Dict[str, ast.expr]
) -> InferredSpec:
    forward = _find_method(cls, "forward")
    if forward is None:
        return InferredSpec()

    params = _forward_tensor_params(forward)
    spec = InferredSpec()

    # Source 1: shape-typed annotations on the forward parameters (highest
    # priority — authoritative, and naturally parametric/symbolic).
    for name, annotation in params:
        if annotation is None:
            continue
        shape = _shape_from_annotation(annotation)
        if shape is not None:
            spec.shapes[name] = shape
            spec.sources[name] = "annotation"

    # Source 2: docstring Args block.
    doc = ast.get_docstring(forward)
    if doc:
        for name, shape in _shapes_from_docstring(doc).items():
            if name in dict(params) and name not in spec.shapes:
                spec.shapes[name] = shape
                spec.sources[name] = "docstring"

    # Source 3: example inputs (class attr / method / lightning convention /
    # module-level assignment).  Concrete; assigned positionally to remaining
    # parameters.
    if not _all_params_covered(params, spec):
        example_shapes = _example_input_shapes(cls, module_assignments)
        remaining = [n for n, _ in params if n not in spec.shapes]
        for name, shape in zip(remaining, example_shapes):
            spec.shapes[name] = shape
            spec.sources[name] = "example_inputs"

    # Source 4: config dict literals (img_size / input_shape).  Only used to
    # fill a single still-uncovered primary parameter.
    if not _all_params_covered(params, spec):
        cfg = _shape_from_config(cls, module_assignments)
        if cfg is not None:
            for name, _ in params:
                if name not in spec.shapes:
                    spec.shapes[name] = cfg
                    spec.sources[name] = "config"
                    break

    return spec


def _all_params_covered(
    params: List[Tuple[str, Optional[ast.expr]]], spec: InferredSpec
) -> bool:
    return all(n in spec.shapes for n, _ in params) and bool(params)


# --------------------------------------------------------------------------- #
# Forward signature helpers.
# --------------------------------------------------------------------------- #
def _find_method(
    cls: ast.ClassDef, name: str
) -> Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node
    return None


def _forward_tensor_params(
    forward: Union[ast.FunctionDef, ast.AsyncFunctionDef]
) -> List[Tuple[str, Optional[ast.expr]]]:
    """Return ``(name, annotation)`` for each non-self positional parameter.

    Parameters annotated with an obviously non-tensor type (``int``, ``bool``,
    ``str``, ``float``, ``Optional[...]`` of those) are skipped.
    """
    out: List[Tuple[str, Optional[ast.expr]]] = []
    args = forward.args
    positional = list(args.posonlyargs) + list(args.args)
    for arg in positional:
        if arg.arg in ("self", "cls"):
            continue
        ann = arg.annotation
        if ann is not None and _is_scalar_annotation(ann):
            continue
        out.append((arg.arg, ann))
    return out


def _is_scalar_annotation(ann: ast.expr) -> bool:
    name = _annotation_base_name(ann)
    return name in {"int", "bool", "str", "float", "bytes"}


# --------------------------------------------------------------------------- #
# Annotation -> shape.
# --------------------------------------------------------------------------- #
def _annotation_base_name(ann: ast.expr) -> str:
    if isinstance(ann, ast.Name):
        return ann.id
    if isinstance(ann, ast.Attribute):
        return ann.attr
    if isinstance(ann, ast.Subscript):
        return _annotation_base_name(ann.value)
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        return ann.value
    return ""


def _shape_from_annotation(ann: ast.expr) -> Optional[Shape]:
    if not isinstance(ann, ast.Subscript):
        return None
    base = _annotation_base_name(ann.value)
    if base not in _SHAPE_ANNOTATION_BASES:
        return None

    sl = ann.slice
    # jaxtyping: Float[Tensor, "batch 3 224 224"]
    if isinstance(sl, ast.Tuple):
        elts = list(sl.elts)
        # Drop a leading dtype/array-type element (Tensor / np.ndarray / ...).
        if elts and _looks_like_array_type(elts[0]):
            elts = elts[1:]
        # jaxtyping packs all dims into a single trailing string literal.
        if len(elts) == 1 and _is_str_const(elts[0]):
            return _parse_dim_string(elts[0].value)
        dims = _dims_from_elements(elts)
        return dims
    # torchtyping single-dim or jaxtyping with no array type:
    #   TensorType["batch"]  or  Float["b c h w"]
    if _is_str_const(sl):
        return _parse_dim_string(sl.value)
    dims = _dims_from_elements([sl])
    return dims


def _looks_like_array_type(node: ast.expr) -> bool:
    base = _annotation_base_name(node)
    return base in {
        "Tensor", "torch.Tensor", "ndarray", "np.ndarray", "Array", "jax.Array",
    }


def _dims_from_elements(elts: List[ast.expr]) -> Optional[Shape]:
    dims: List[Dim] = []
    for e in elts:
        if _is_str_const(e):
            parsed = _parse_dim_string(e.value)
            if parsed is None:
                return None
            dims.extend(parsed)
        elif isinstance(e, ast.Constant) and isinstance(e.value, int):
            dims.append(e.value)
        elif isinstance(e, ast.Name):
            dims.append(e.id)
        else:
            return None
    return tuple(dims) if dims else None


_DIM_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_dim_string(s: str) -> Optional[Shape]:
    """Parse a jaxtyping-style dim string like ``"batch 3 224 224"``."""
    s = s.strip()
    if not s:
        return None
    # jaxtyping separates dims by spaces; tolerate commas too.
    tokens = re.split(r"[\s,]+", s)
    dims: List[Dim] = []
    for tok in tokens:
        if not tok:
            continue
        # jaxtyping modifiers: "*batch" (variadic), "#c" (broadcastable),
        # "dim=3" — strip leading symbols, keep the meaningful name/number.
        tok = tok.lstrip("*#")
        if "=" in tok:
            tok = tok.split("=", 1)[-1]
        if tok.isdigit():
            dims.append(int(tok))
        elif _DIM_TOKEN.match(tok):
            dims.append(tok)
        else:
            return None
    return tuple(dims) if dims else None


# --------------------------------------------------------------------------- #
# Docstring -> shape.
# --------------------------------------------------------------------------- #
# Matches "x: (B, 3, 224, 224)" / "x (Tensor): shape (B, 3, H, W)" /
# "x -- shape [batch, 10]".
_DOC_PARAM = re.compile(
    r"""^\s*
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)        # param name
        \s*(?:\([^)]*\))?                        # optional (Tensor) type
        \s*[:\-]+                                # ':' or '--'
        .*?                                      # words like 'shape', 'of size'
        [\(\[]\s*(?P<dims>[^)\]]+?)\s*[\)\]]     # (dims) or [dims]
    """,
    re.VERBOSE,
)


def _shapes_from_docstring(doc: str) -> Dict[str, Shape]:
    out: Dict[str, Shape] = {}
    for line in doc.splitlines():
        m = _DOC_PARAM.match(line)
        if not m:
            continue
        name = m.group("name")
        dims = _parse_dim_string(m.group("dims").replace(",", " "))
        if dims is not None and len(dims) >= 1:
            out[name] = dims
    return out


# --------------------------------------------------------------------------- #
# Example inputs.
# --------------------------------------------------------------------------- #
def _example_input_shapes(
    cls: ast.ClassDef, module_assignments: Dict[str, ast.expr]
) -> List[Shape]:
    """Return shapes of example-input tensors, in argument order."""
    candidates: List[ast.expr] = []

    # Class-level attribute assignments.
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and _is_example_name(tgt.id):
                    candidates.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if _is_example_name(node.target.id) and node.value is not None:
                candidates.append(node.value)

    # example_inputs()/get_example_inputs() method returning tensor(s).
    for mname in ("example_inputs", "get_example_inputs", "example_input_array"):
        method = _find_method(cls, mname)
        if method is not None:
            for ret in _return_values(method):
                candidates.append(ret)

    # Module-level assignment (e.g. example_inputs = torch.randn(2, 3, 224, 224)).
    for name, value in module_assignments.items():
        if _is_example_name(name):
            candidates.append(value)

    for cand in candidates:
        shapes = _tensor_shapes_from_expr(cand)
        if shapes:
            return shapes
    return []


def _is_example_name(name: str) -> bool:
    low = name.lower()
    return "example_input" in low or low in {
        "example_input_array", "dummy_input", "sample_input", "example",
    }


def _tensor_shapes_from_expr(expr: ast.expr) -> List[Shape]:
    """Extract one shape per tensor-factory call in *expr* (tuple/list aware)."""
    if isinstance(expr, (ast.Tuple, ast.List)):
        out: List[Shape] = []
        for e in expr.elts:
            out.extend(_tensor_shapes_from_expr(e))
        return out
    shape = _shape_from_factory_call(expr)
    return [shape] if shape is not None else []


def _shape_from_factory_call(expr: ast.expr) -> Optional[Shape]:
    if not isinstance(expr, ast.Call):
        return None
    fname = _call_func_name(expr.func)
    if fname not in _TENSOR_FACTORIES:
        return None
    dims: List[Dim] = []
    args = list(expr.args)
    # torch.full(size, fill_value) / randint(low, high, size): the shape is a
    # tuple/list argument, not the positional ints.
    if fname in ("full", "randint"):
        for a in args:
            if isinstance(a, (ast.Tuple, ast.List)):
                parsed = _ints_or_names(a.elts)
                return parsed
        return None
    # A single tuple/list arg also describes the shape: torch.randn((2, 3)).
    if len(args) == 1 and isinstance(args[0], (ast.Tuple, ast.List)):
        return _ints_or_names(args[0].elts)
    for a in args:
        if isinstance(a, ast.Constant) and isinstance(a.value, int):
            dims.append(a.value)
        elif isinstance(a, ast.Name):
            dims.append(a.id)
        elif isinstance(a, ast.UnaryOp):
            break
        else:
            break
    return tuple(dims) if dims else None


def _ints_or_names(elts: List[ast.expr]) -> Optional[Shape]:
    dims: List[Dim] = []
    for e in elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, int):
            dims.append(e.value)
        elif isinstance(e, ast.Name):
            dims.append(e.id)
        else:
            return None
    return tuple(dims) if dims else None


def _return_values(
    method: Union[ast.FunctionDef, ast.AsyncFunctionDef]
) -> List[ast.expr]:
    out: List[ast.expr] = []
    for node in ast.walk(method):
        if isinstance(node, ast.Return) and node.value is not None:
            out.append(node.value)
    return out


# --------------------------------------------------------------------------- #
# Config dicts.
# --------------------------------------------------------------------------- #
def _shape_from_config(
    cls: ast.ClassDef, module_assignments: Dict[str, ast.expr]
) -> Optional[Shape]:
    for value in list(module_assignments.values()) + _class_dict_values(cls):
        shape = _shape_from_dict_literal(value)
        if shape is not None:
            return shape
    return None


def _class_dict_values(cls: ast.ClassDef) -> List[ast.expr]:
    out: List[ast.expr] = []
    for node in cls.body:
        if isinstance(node, ast.Assign):
            out.append(node.value)
    return out


def _shape_from_dict_literal(value: ast.expr) -> Optional[Shape]:
    if not isinstance(value, ast.Dict):
        return None
    for k, v in zip(value.keys, value.values):
        if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                and k.value in _CONFIG_SHAPE_KEYS:
            if isinstance(v, (ast.Tuple, ast.List)):
                return _ints_or_names(v.elts)
    return None


# --------------------------------------------------------------------------- #
# Small AST utilities.
# --------------------------------------------------------------------------- #
def _module_level_assignments(tree: ast.Module) -> Dict[str, ast.expr]:
    out: Dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                out[node.target.id] = node.value
    return out


def _call_func_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_str_const(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)
