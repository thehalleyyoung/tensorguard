"""v5 / Track-C — Symbolic config attribute binding.

Modern HF/timm Transformer modules read shape-determining hyperparameters
from ``self.config`` rather than passing them as constructor positional
arguments.  Example::

    class BertSelfAttention(nn.Module):
        def __init__(self, config):
            self.num_heads = config.num_attention_heads
            self.head_dim  = config.hidden_size // config.num_attention_heads
            self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        def forward(self, x):
            B, T, _ = x.shape
            qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
            ...

To verify such a module *generically* (i.e. without binding to a specific
config instance) we treat each config attribute as a fresh symbolic
integer ``SymInt('hidden_size')`` and propagate it through the analyzer.

Public API
----------
* :class:`SymInt`            — symbolic positive integer with light arithmetic.
* :func:`symbolic_config`    — decorator that attaches an attrs contract.
* :func:`bind_symbolic_attrs`— produces a stand-in config object that the
                              analyzer can read attributes from.
* :func:`verify_against_instance` — post-hoc check that an actual config
                              instance is consistent with the contract.

The contract is intentionally lightweight: it stores nothing more than the
expected attribute names and (optionally) a divisibility requirement
between them (e.g. ``hidden_size % num_heads == 0``).  This is enough to
let downstream modules in :mod:`src.v5.qkv_unpacking` and
:mod:`src.v5.reshape_neg1` reason about Transformer-shaped tensors.

This module *does not* edit any existing src/ file — it only imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Re-export the existing TensorShape pieces so callers can stay within v5.
from src.tensor_shapes import ShapeDim, TensorShape  # noqa: F401


# ────────────────────────────────────────────────────────────────────────────
# Symbolic integer
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SymInt:
    """Symbolic positive integer used as a stand-in for a config attribute.

    Equality and hashing are *by name* — two ``SymInt('h')`` are the same
    symbol and will be unified by the underlying SMT encoding.
    """

    name: str
    positive: bool = True
    divisible_by: Tuple[Union[int, "SymInt"], ...] = ()

    # ── light arithmetic so callers can write e.g.  3 * h, h // n ─────────
    def __mul__(self, other: Any) -> "SymExpr":
        return SymExpr("*", (self, other))

    __rmul__ = __mul__

    def __add__(self, other: Any) -> "SymExpr":
        return SymExpr("+", (self, other))

    __radd__ = __add__

    def __floordiv__(self, other: Any) -> "SymExpr":
        return SymExpr("//", (self, other))

    def __mod__(self, other: Any) -> "SymExpr":
        return SymExpr("%", (self, other))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SymInt({self.name!r})"

    # Allow using a SymInt as a tensor dim — TensorShape stores str|int.
    def as_dim(self) -> ShapeDim:
        return ShapeDim(self.name)


@dataclass(frozen=True)
class SymExpr:
    """Tiny arithmetic AST over :class:`SymInt`s and ints."""
    op: str
    args: Tuple[Any, ...]

    def __repr__(self) -> str:
        a, b = self.args
        return f"({a!r}{self.op}{b!r})"

    def as_dim(self) -> ShapeDim:
        return ShapeDim(repr(self))


# ────────────────────────────────────────────────────────────────────────────
# Contract + decorator
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolicConfigContract:
    """Per-class declaration of which config attributes should be bound
    to fresh symbolic integers when analyzing the class's ``forward`` body.
    """

    attrs: Dict[str, SymInt]
    invariants: List[Callable[[Dict[str, int]], bool]] = field(default_factory=list)

    def attr_names(self) -> List[str]:
        return list(self.attrs.keys())


# A registry keyed by the *qualified* class name (``module.ClassName``).
# Using qualified names lets the analyzer look up a contract from a
# string when source code is parsed without the class object handy.
_REGISTRY: Dict[str, SymbolicConfigContract] = {}


def _qualname(cls_or_name: Union[type, str]) -> str:
    if isinstance(cls_or_name, str):
        return cls_or_name
    mod = getattr(cls_or_name, "__module__", "")
    name = cls_or_name.__name__
    return f"{mod}.{name}" if mod else name


def symbolic_config(
    cls: Optional[type] = None,
    *,
    attrs: Optional[Dict[str, SymInt]] = None,
    invariants: Optional[List[Callable[[Dict[str, int]], bool]]] = None,
) -> Any:
    """Attach a symbolic-config contract to ``cls``.

    Usage::

        @symbolic_config(attrs={
            'hidden_size':         SymInt('h'),
            'num_attention_heads': SymInt('n'),
        }, invariants=[lambda c: c['hidden_size'] % c['num_attention_heads'] == 0])
        class BertSelfAttention(nn.Module):
            ...

    Or imperatively::

        symbolic_config(BertSelfAttention, attrs={'hidden_size': SymInt('h')})
    """
    if attrs is None:
        attrs = {}
    contract = SymbolicConfigContract(attrs=dict(attrs),
                                      invariants=list(invariants or []))

    def _register(target: type) -> type:
        _REGISTRY[_qualname(target)] = contract
        # Also store on the class so it survives pickling.
        setattr(target, "__tensorguard_sym_config__", contract)
        return target

    if cls is not None:
        return _register(cls)
    return _register


# A namespace under :mod:`tensorguard.contract` is requested in the spec.
# We provide it as an attribute lookup so that
# ``tensorguard.contract.symbolic_config`` works as soon as anything in the
# v5 package has been imported.
class _ContractNamespace:
    symbolic_config = staticmethod(symbolic_config)
    SymInt = SymInt
    SymExpr = SymExpr
    SymbolicConfigContract = SymbolicConfigContract

    @staticmethod
    def get(cls_or_name: Union[type, str]) -> Optional[SymbolicConfigContract]:
        return _REGISTRY.get(_qualname(cls_or_name))

    @staticmethod
    def all_registered() -> Dict[str, SymbolicConfigContract]:
        return dict(_REGISTRY)


def _install_contract_namespace() -> None:
    """Register ``tensorguard.contract`` if a ``tensorguard`` shim module is
    importable.  This keeps the public spelling requested by the contract.
    """
    import sys
    import types

    tg = sys.modules.get("tensorguard")
    if tg is None:
        tg = types.ModuleType("tensorguard")
        sys.modules["tensorguard"] = tg
    if not hasattr(tg, "contract"):
        tg.contract = _ContractNamespace()


_install_contract_namespace()


# ────────────────────────────────────────────────────────────────────────────
# Symbolic stand-in config
# ────────────────────────────────────────────────────────────────────────────

class _SymbolicConfig:
    """An object that returns the *symbolic* binding for any attribute
    declared in the contract, raising AttributeError otherwise.
    """

    def __init__(self, contract: SymbolicConfigContract):
        self._contract = contract
        for name, sym in contract.attrs.items():
            object.__setattr__(self, name, sym)

    def __repr__(self) -> str:  # pragma: no cover
        items = ", ".join(f"{k}={v.name}" for k, v in self._contract.attrs.items())
        return f"<SymbolicConfig {items}>"


def bind_symbolic_attrs(cls_or_contract: Union[type, str, SymbolicConfigContract]
                       ) -> _SymbolicConfig:
    """Return an object usable as a stand-in for ``self.config`` whose
    attribute reads yield :class:`SymInt`s as declared by the contract.
    """
    if isinstance(cls_or_contract, SymbolicConfigContract):
        contract = cls_or_contract
    else:
        contract = _REGISTRY.get(_qualname(cls_or_contract))
        if contract is None:
            raise KeyError(f"No symbolic config registered for {cls_or_contract!r}")
    return _SymbolicConfig(contract)


# ────────────────────────────────────────────────────────────────────────────
# Post-hoc verification against a real config instance
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationReport:
    ok: bool
    missing_attrs: List[str] = field(default_factory=list)
    bad_invariants: List[int] = field(default_factory=list)
    detail: str = ""

    def __bool__(self) -> bool:  # noqa: D401
        return self.ok


def verify_against_instance(cls_or_contract: Union[type, str, SymbolicConfigContract],
                            config_instance: Any) -> VerificationReport:
    """Check that ``config_instance`` defines every attribute declared in
    the contract, and that every invariant holds when those attributes are
    resolved to concrete ints.
    """
    if isinstance(cls_or_contract, SymbolicConfigContract):
        contract = cls_or_contract
    else:
        contract = _REGISTRY.get(_qualname(cls_or_contract))
        if contract is None:
            return VerificationReport(False, detail=f"no contract for {cls_or_contract!r}")

    missing: List[str] = []
    bound: Dict[str, int] = {}
    for name in contract.attrs:
        if not hasattr(config_instance, name):
            missing.append(name)
        else:
            v = getattr(config_instance, name)
            if isinstance(v, int):
                bound[name] = v
    if missing:
        return VerificationReport(False, missing_attrs=missing,
                                  detail=f"config missing attrs: {missing}")

    bad: List[int] = []
    for i, inv in enumerate(contract.invariants):
        try:
            if not inv(bound):
                bad.append(i)
        except Exception:
            bad.append(i)
    if bad:
        return VerificationReport(False, bad_invariants=bad,
                                  detail=f"invariants {bad} failed")

    return VerificationReport(True, detail="ok")


# ────────────────────────────────────────────────────────────────────────────
# Shape helpers for use by other v5 modules
# ────────────────────────────────────────────────────────────────────────────

def sym_to_dim(value: Any) -> ShapeDim:
    """Coerce ints, str, SymInt, SymExpr, or ShapeDim to a ShapeDim."""
    if isinstance(value, ShapeDim):
        return value
    if isinstance(value, (SymInt, SymExpr)):
        return value.as_dim()
    if isinstance(value, int):
        return ShapeDim(value)
    if isinstance(value, str):
        return ShapeDim(value)
    raise TypeError(f"cannot coerce {value!r} ({type(value).__name__}) to ShapeDim")


def shape_with_config(shape_template: Iterable[Any]) -> TensorShape:
    """Build a :class:`TensorShape` from a mix of ints / SymInts / strings."""
    return TensorShape(tuple(sym_to_dim(d) for d in shape_template))


__all__ = [
    "SymInt",
    "SymExpr",
    "SymbolicConfigContract",
    "symbolic_config",
    "bind_symbolic_attrs",
    "verify_against_instance",
    "VerificationReport",
    "sym_to_dim",
    "shape_with_config",
]
