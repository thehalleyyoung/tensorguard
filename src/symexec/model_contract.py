"""Derive a **weights contract from model code** — the bridge that lets a
checkpoint be certified against the model that will load it, with no reference
checkpoint required (even_more.md quantum leap, weights layer).

Given the *source* of an ``nn.Module`` and a concrete *construction* (e.g.
``GPT(n_layer=12, n_embd=768)``), we drive the symbolic-execution engine to run
the model's ``__init__`` — resolving every layer's integer hyper-parameters by
abstract interpretation — and read off the resulting submodule tree.  For each
standard ``nn`` layer we emit the parameters PyTorch is *guaranteed* to register,
named exactly as ``state_dict`` / ``load_state_dict(strict=True)`` would (dotted
submodule paths, ``0/1/...`` for ``nn.Sequential``).

Soundness
---------
The contract is deliberately **partial** and **shape-only**:

* We only emit a parameter when its existence and full shape are *forced* by the
  resolved hyper-parameters and statically-known constructor flags (``bias=``,
  ``elementwise_affine=``, ``affine=``, ``track_running_stats=``).  When a flag or
  dimension is not statically known, or a container (``nn.ModuleList``, a
  comprehension) cannot be enumerated, we **abstain** rather than guess.
* Because we cannot see ``register_parameter`` / ``register_buffer`` / raw
  ``nn.Parameter`` attributes or dynamically-built submodules, the contract never
  claims to be exhaustive: a checkpoint may legitimately carry tensors the
  contract does not mention.  It is therefore checked with ``contract_partial`` —
  only *positive* obligations (a derived tensor must be present, with the derived
  shape) are asserted; a missing derived tensor or a shape mismatch is a genuine
  ``load_state_dict(strict=True)`` failure.

Dtypes are intentionally absent (a model's parameter dtype is decided at load /
``.to(...)`` time, not by its code), so only shapes are constrained.

Torch-free; standard library only.
"""

from __future__ import annotations

import ast
import enum
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from .interpreter import Interpreter
from .state import State
from .values import AbstractValue, ModuleVal

__all__ = [
    "AbstainCode",
    "Abstention",
    "CoverageReport",
    "ModelContract",
    "derive_model_contract",
    "model_contract_to_expected",
    "certify_weights_against_model",
]

# nn layers whose parameter shapes we can emit soundly from resolved meta.
_NN_LEAF = {
    "Linear",
    "Conv1d", "Conv2d", "Conv3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "Embedding",
    "LayerNorm",
}


# --------------------------------------------------------------------------- #
# Abstention taxonomy.                                                          #
# --------------------------------------------------------------------------- #
class AbstainCode(enum.Enum):
    """A *closed* set of machine-readable reasons the deriver declined to emit a
    parameter (or to descend into a subtree).  Every abstention carries exactly
    one of these codes, so contract coverage is measurable and the deriver can
    never silently guess.  The string value is the stable serialized form."""

    #: A required integer dimension / hyper-parameter is not statically known
    #: (e.g. ``Linear`` in/out_features, ``Conv`` channels / kernel size,
    #: ``Embedding`` num/dim, ``LayerNorm`` normalized_shape, ``BatchNorm``
    #: num_features).  PyTorch *would* register the tensor, but we cannot force
    #: its shape, so we abstain rather than guess.
    NON_CONSTANT_DIM = "non_constant_dim"

    #: A boolean construction flag that decides a parameter's *existence* is not
    #: statically known (``bias=``, ``elementwise_affine=``, ``affine=``,
    #: ``track_running_stats=``).
    UNRESOLVED_FLAG = "unresolved_flag"

    #: The construction is statically known but contradictory / degenerate, so no
    #: sound shape exists (e.g. ``in_channels`` not divisible by ``groups``, or
    #: ``groups == 0``).
    INVALID_LAYER_CONFIG = "invalid_layer_config"

    #: A container (``nn.ModuleList`` / a comprehension, or a tuple of unknown
    #: length) whose elements cannot be statically enumerated.
    UNENUMERABLE_CONTAINER = "unenumerable_container"

    #: A submodule whose *existence* depends on a construction-time branch
    #: (``if cfg.use_x: self.proj = nn.Linear(...)``) whose guard is not
    #: statically resolved.  PyTorch may or may not register the submodule, so we
    #: abstain on that subtree only — never guessing it is always present (which
    #: would invent params) nor always absent (which would silently drop them).
    CONDITIONAL_SUBMODULE = "conditional_submodule"

    #: A module or container that refers back to itself (a cycle in the value
    #: graph); we abstain on the back-edge to stay total.
    CYCLIC_REFERENCE = "cyclic_reference"

    #: The module-tree recursion depth ceiling was reached (``_MAX_WALK_DEPTH``);
    #: we abstain rather than recurse without bound.
    MAX_DEPTH_EXCEEDED = "max_depth_exceeded"


@dataclass(frozen=True)
class Abstention:
    """One typed abstention: *where* (``path``), *why* (``code``), and a free-text
    ``detail`` for humans.  Frozen + hashable so contracts can dedupe/sort them."""

    path: str
    code: AbstainCode
    detail: str

    def _sort_key(self) -> Tuple[str, str, str]:
        return (self.path, self.code.value, self.detail)


@dataclass(frozen=True)
class ModelContract:
    """A partial, shape-only ``name -> shape`` contract derived from model code.

    ``params`` maps a ``state_dict`` parameter name to its forced shape;
    ``abstained`` records a typed :class:`Abstention` for every layer/container we
    could not resolve, so the boundary of the guarantee is explicit and
    *measurable*.  ``partial`` is always ``True`` (see the module docstring)."""

    model_class: str
    construction: str
    params: Dict[str, Tuple[int, ...]]
    abstained: Tuple[Abstention, ...]
    resolved_layers: int
    partial: bool = True

    @property
    def abstain_codes(self) -> Tuple[AbstainCode, ...]:
        """The distinct abstention codes present, sorted by value."""
        return tuple(sorted({a.code for a in self.abstained}, key=lambda c: c.value))

    def coverage(self, oracle: Mapping[str, Tuple[int, ...]]) -> "CoverageReport":
        """Measure this *partial* contract against an authoritative ``oracle``
        map of ``state_dict`` ``name -> shape`` (e.g. the real torch module's
        ``state_dict()`` in a test).

        Coverage is the fraction of the parameters PyTorch actually registers
        that this contract emitted **soundly** (present *and* shape-identical).
        The method is pure and torch-free: the caller supplies the oracle, so
        torch never enters the trust path.

        The returned :class:`CoverageReport` also surfaces:

        * ``unsound`` — emitted names that are absent from the oracle or carry a
          wrong shape.  For a *sound* partial contract this **must** be empty;
          any entry is a latent false-positive bug in the deriver.
        * ``missing`` — oracle names the contract did not emit (the legitimately
          abstained / not-yet-resolved boundary).
        """
        oracle = {str(k): tuple(int(d) for d in v) for k, v in oracle.items()}
        correct: List[str] = []
        unsound: List[str] = []
        for name, shape in self.params.items():
            shape = tuple(int(d) for d in shape)
            if name in oracle and oracle[name] == shape:
                correct.append(name)
            else:
                unsound.append(name)
        missing = [n for n in oracle if n not in self.params]
        return CoverageReport(
            emitted=len(self.params),
            registered=len(oracle),
            correct=tuple(sorted(correct)),
            unsound=tuple(sorted(unsound)),
            missing=tuple(sorted(missing)),
        )


@dataclass(frozen=True)
class CoverageReport:
    """The result of measuring a :class:`ModelContract` against a torch
    ``state_dict`` oracle (see :meth:`ModelContract.coverage`).

    ``fraction`` = ``len(correct) / registered`` is the headline coverage number;
    ``is_sound`` is the hard gate (no emitted param may be wrong/absent)."""

    emitted: int
    registered: int
    correct: Tuple[str, ...]
    unsound: Tuple[str, ...]
    missing: Tuple[str, ...]

    @property
    def num_correct(self) -> int:
        return len(self.correct)

    @property
    def fraction(self) -> float:
        """Soundly-emitted params / params PyTorch registers.

        A model that registers *no* parameters is vacuously fully covered
        (``1.0``) so the metric never penalises an empty oracle."""
        if self.registered == 0:
            return 1.0
        return len(self.correct) / self.registered

    @property
    def is_sound(self) -> bool:
        """``True`` iff every emitted parameter matched the oracle (no false
        positives).  This is the invariant CI must never let regress."""
        return len(self.unsound) == 0


# --------------------------------------------------------------------------- #
# Driving the engine to instantiate a model.                                    #
# --------------------------------------------------------------------------- #
def _instantiate_model(
    source: str, construction: str, *, filename: str
) -> Tuple[ModuleVal, str]:
    """Symbolically instantiate ``construction`` against ``source`` and return the
    root :class:`ModuleVal` plus the model class name."""
    module = ast.parse(source, filename=filename)
    interp = Interpreter(module, filename=filename)

    expr = ast.parse(construction, mode="eval").body
    if not isinstance(expr, ast.Call):
        raise ValueError(f"construction {construction!r} is not a constructor call")

    if isinstance(expr.func, ast.Name):
        class_name = expr.func.id
    elif isinstance(expr.func, ast.Attribute):
        class_name = expr.func.attr
    else:
        raise ValueError(f"cannot resolve the class in {construction!r}")

    cls = interp.classes.get(class_name)
    if cls is None:
        raise ValueError(f"class {class_name!r} is not defined in the model source")

    state = State()
    pos: List[AbstractValue] = [interp.eval_expr(a, state) for a in expr.args]
    kw: Dict[str, AbstractValue] = {
        k.arg: interp.eval_expr(k.value, state) for k in expr.keywords if k.arg
    }
    root = interp._instantiate(cls, pos, kw, expr)
    if not isinstance(root, ModuleVal):
        raise ValueError(f"{class_name!r} did not instantiate to a module")
    return root, class_name


# --------------------------------------------------------------------------- #
# Per-layer sound parameter emission.                                           #
# --------------------------------------------------------------------------- #
def _emit_layer(
    mv: ModuleVal, prefix: str, params: Dict[str, Tuple[int, ...]],
    abstained: List[Abstention],
) -> None:
    """Emit the parameters a standard nn layer is *guaranteed* to register, or
    record a typed abstention if a required dimension/flag is unresolved."""
    def key(p: str) -> str:
        return p if prefix == "" else f"{prefix}.{p}"

    cls = mv.class_name

    def flag(name: str) -> Optional[int]:
        v = mv.get_meta(name)
        if v is None or v < 0:
            return None  # absent or statically-unknown -> abstain on it
        return v

    if cls == "Linear":
        i, o = mv.get_meta("in_features"), mv.get_meta("out_features")
        if i is None or o is None:
            abstained.append(Abstention(prefix, AbstainCode.NON_CONSTANT_DIM,
                                        "Linear with unresolved in/out_features"))
            return
        params[key("weight")] = (o, i)
        b = flag("bias")
        if b is None:
            abstained.append(Abstention(key("bias"), AbstainCode.UNRESOLVED_FLAG,
                                        "Linear bias flag not statically known"))
        elif b == 1:
            params[key("bias")] = (o,)
        return

    if cls == "Embedding":
        n, d = mv.get_meta("num_embeddings"), mv.get_meta("embedding_dim")
        if n is None or d is None:
            abstained.append(Abstention(prefix, AbstainCode.NON_CONSTANT_DIM,
                                        "Embedding with unresolved num/dim"))
            return
        params[key("weight")] = (n, d)
        return

    if cls in ("Conv1d", "Conv2d", "Conv3d"):
        i, o = mv.get_meta("in_channels"), mv.get_meta("out_channels")
        groups = mv.get_meta("groups")
        klen = mv.get_meta("k_len")
        if i is None or o is None or groups is None or klen is None:
            abstained.append(Abstention(prefix, AbstainCode.NON_CONSTANT_DIM,
                                        f"{cls} with unresolved channels/groups/kernel"))
            return
        kernel = tuple(mv.get_meta(f"k{j}") for j in range(klen))
        if any(k is None for k in kernel):
            abstained.append(Abstention(prefix, AbstainCode.NON_CONSTANT_DIM,
                                        f"{cls} with unresolved kernel size"))
            return
        if groups == 0 or i % groups != 0:
            abstained.append(Abstention(prefix, AbstainCode.INVALID_LAYER_CONFIG,
                                        f"{cls} in_channels not divisible by groups (or groups==0)"))
            return
        params[key("weight")] = (o, i // groups, *kernel)
        b = flag("bias")
        if b is None:
            abstained.append(Abstention(key("bias"), AbstainCode.UNRESOLVED_FLAG,
                                        f"{cls} bias flag not statically known"))
        elif b == 1:
            params[key("bias")] = (o,)
        return

    if cls == "LayerNorm":
        nlen = mv.get_meta("ns_len")
        affine = flag("elementwise_affine")
        if nlen is None:
            abstained.append(Abstention(prefix, AbstainCode.NON_CONSTANT_DIM,
                                        "LayerNorm with unresolved normalized_shape"))
            return
        if affine is None:
            abstained.append(Abstention(prefix, AbstainCode.UNRESOLVED_FLAG,
                                        "LayerNorm elementwise_affine not statically known"))
            return
        if affine == 1:
            ns = tuple(mv.get_meta(f"ns{j}") for j in range(nlen))
            if any(s is None for s in ns):
                abstained.append(Abstention(prefix, AbstainCode.NON_CONSTANT_DIM,
                                            "LayerNorm with unresolved normalized_shape"))
                return
            params[key("weight")] = ns
            params[key("bias")] = ns
        return

    if cls in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d"):
        nf = mv.get_meta("num_features")
        affine = flag("affine")
        trs = flag("track_running_stats")
        if nf is None:
            abstained.append(Abstention(prefix, AbstainCode.NON_CONSTANT_DIM,
                                        f"{cls} with unresolved num_features"))
            return
        if affine is None or trs is None:
            abstained.append(Abstention(prefix, AbstainCode.UNRESOLVED_FLAG,
                                        f"{cls} affine/track_running_stats not statically known"))
            return
        if affine == 1:
            params[key("weight")] = (nf,)
            params[key("bias")] = (nf,)
        if trs == 1:
            params[key("running_mean")] = (nf,)
            params[key("running_var")] = (nf,)
            params[key("num_batches_tracked")] = ()
        return


# --------------------------------------------------------------------------- #
# Walking the module tree.                                                      #
# --------------------------------------------------------------------------- #
# Hard ceiling on module-tree recursion.  The interpreter already bounds the
# depth at which it will *instantiate* nested user modules, so legitimate models
# never approach this; the guard exists purely to make ``_walk`` a *total*
# function — it must terminate and abstain (never hang or raise ``RecursionError``)
# on any value graph it is handed, including hand-built or pathologically deep /
# cyclic ones.  Kept well below Python's default recursion limit (1000) so the
# guard fires before the interpreter stack would overflow.
_MAX_WALK_DEPTH = 200


def _walk(
    val: AbstractValue, prefix: str, params: Dict[str, Tuple[int, ...]],
    abstained: List[Abstention], resolved: List[int], seen: set,
    depth: int = 0,
) -> None:
    # Depth guard: abstain (with a typed reason) rather than recurse without bound.
    if depth > _MAX_WALK_DEPTH:
        abstained.append(Abstention(prefix, AbstainCode.MAX_DEPTH_EXCEEDED,
                                    "max module-tree depth exceeded"))
        return

    if isinstance(val, ModuleVal):
        # A submodule registered only on one side of an unresolved construction
        # branch: PyTorch may or may not register it, so abstain on it (and its
        # whole subtree) rather than treat it as unconditionally present/absent.
        # Checked before the leaf/container dispatch so it applies to any class.
        if val.get_meta("__conditional__") == 1:
            abstained.append(Abstention(prefix, AbstainCode.CONDITIONAL_SUBMODULE,
                                        f"submodule {prefix or '<root>'} is registered "
                                        f"under an unresolved construction-time "
                                        f"condition"))
            return
        if val.class_name in _NN_LEAF:
            before = len(params)
            _emit_layer(val, prefix, params, abstained)
            if len(params) > before:
                resolved[0] += 1
            return
        # A registered container (nn.ModuleList/ModuleDict) whose contents could
        # not be statically enumerated: abstain on the subtree, never guess.
        if val.get_meta("__opaque_container__") == 1:
            abstained.append(Abstention(prefix, AbstainCode.UNENUMERABLE_CONTAINER,
                                        f"{val.class_name} contents not statically "
                                        f"enumerable"))
            return
        # User module or nn.Sequential/ModuleList/ModuleDict: recurse into its
        # (possibly index-/key-keyed) registered submodule attributes.
        if id(val) in seen:  # a module instance reached twice -> cycle / sharing
            abstained.append(Abstention(prefix, AbstainCode.CYCLIC_REFERENCE,
                                        "cyclic module reference"))
            return
        seen.add(id(val))
        for name, child in val.attrs:
            child_prefix = name if prefix == "" else f"{prefix}.{name}"
            _walk(child, child_prefix, params, abstained, resolved, seen, depth + 1)
        return

    # A plain ``list``/``tuple`` attribute is NOT registered by PyTorch: its
    # module children never appear in ``state_dict`` (a classic silent-bug source
    # in real code).  We therefore emit nothing for it — matching torch exactly —
    # rather than inventing ``attr.0.*`` params (which would be a false positive).
    # Registered module containers are modelled as ``ModuleVal`` above.



# --------------------------------------------------------------------------- #
# Public API.                                                                   #
# --------------------------------------------------------------------------- #
def derive_model_contract(
    source: str, construction: str, *, filename: str = "<model>"
) -> ModelContract:
    """Derive a partial, shape-only weights contract from model ``source`` and a
    concrete ``construction`` expression (e.g. ``"GPT(n_layer=12, n_embd=768)"``)."""
    root, class_name = _instantiate_model(source, construction, filename=filename)
    params: Dict[str, Tuple[int, ...]] = {}
    abstained: List[Abstention] = []
    resolved = [0]
    _walk(root, "", params, abstained, resolved, set())
    return ModelContract(
        model_class=class_name,
        construction=construction,
        params=dict(sorted(params.items())),
        abstained=tuple(sorted(set(abstained), key=lambda a: a._sort_key())),
        resolved_layers=resolved[0],
    )


def model_contract_to_expected(
    contract: ModelContract,
) -> Dict[str, Tuple[Optional[str], Tuple[int, ...]]]:
    """Adapt a :class:`ModelContract` to the ``expected`` map
    :func:`~src.symexec.weights.certify_weights_file` consumes (dtype ``None``,
    i.e. shape-only)."""
    return {name: (None, shape) for name, shape in contract.params.items()}


def certify_weights_against_model(
    checkpoint_path: str,
    model_source: str,
    construction: str,
    *,
    check_finite: bool = True,
    filename: str = "<model>",
):
    """Certify a safetensors ``checkpoint_path`` against a contract *derived from
    model code* — the no-reference bridge.

    Returns ``(certificate, contract)``."""
    from .weights import certify_weights_file

    contract = derive_model_contract(model_source, construction, filename=filename)
    cert = certify_weights_file(
        checkpoint_path,
        check_finite=check_finite,
        expected=model_contract_to_expected(contract),
        contract_partial=True,
    )
    return cert, contract
