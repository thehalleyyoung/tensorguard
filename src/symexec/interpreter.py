"""AST symbolic interpreter for the TensorGuard symbolic executor.

This is the heart of the engine.  It walks the AST of a function body, threading
an abstract :class:`~src.symexec.state.State` and emitting
:class:`~src.symexec.bugs.SymBug` records for forced runtime failures.

Modeled (sound, no-false-positive) checks implemented here:

* **Tuple-unpacking arity** (``a, b = rhs``) — fires when ``rhs`` is provably not
  unpackable into the requested arity (``None``, a 0-d tensor, a tuple of the
  wrong fixed length, or — via interprocedural resolution — a callee that
  returns a single non-tuple value on the only feasible path).  This is the
  *titans-pytorch #60* class.
* **Rank-dependent indexing** (``x[-1, :, :]``) — fires when a subscript uses
  more index dimensions than the receiver's known rank.  This is the
  *OpenStrawberry #113* class.

Anything outside the modeled fragment evaluates to ``TOP`` and never reports.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .bugs import SymBug, SymBugKind
from .serialize import dumps, to_json
from .state import State
from .transfer import tensor_method
from .symdim import SymDim
from . import smt_bridge
from . import cegar
from . import trace_min
from . import stubs
from . import confidence as confidence_mod
from .confidence import ConfidenceSignals
from .abstain import AbstainCategory, AbstainLedger, AbstainReason
from .coverage import CoverageMeter
from .config import SymConfig, DEFAULT_CONFIG
from .disjunctive import DisjunctiveState
from .values import (
    AbstractValue,
    BoolVal,
    CallableVal,
    DictVal,
    FloatVal,
    IntVal,
    ListVal,
    ModuleVal,
    NONE,
    NoneVal,
    SetVal,
    StrVal,
    TensorVal,
    TOP,
    TupleVal,
    int_const,
    int_range,
    join_many,
)

_MAX_DEPTH = 6

# Loop fixpoint controls (Steps 17/38): unroll the body precisely a few times
# (so certain early-iteration bugs are caught at their exact states), then drive
# an interval-widening fixpoint to a sound loop-head invariant.
_LOOP_UNROLL = 2
_LOOP_FIX_MAX = 8
_LOOP_NARROW_MAX = 4

# Bounded precise unrolling of a constant-trip ``for`` loop during module
# construction (Step 12).  A GPT-style stack is typically 12-96 blocks; we cap
# well above that.  A constant range longer than this (or a symbolic one) is not
# unrolled — the container it builds is marked opaque so the deriver abstains.
_MAX_CONSTRUCT_UNROLL = 512


def _parse_einsum_eq(eq: str):
    """Parse an einsum equation string into (operand_subscripts, out_subscript).

    Returns ``None`` (abstain) when the equation contains an ellipsis or any
    non-alphabetic subscript character, which we don't model precisely.
    ``out_subscript`` is ``None`` for implicit-output equations (no ``->``)."""
    eq = eq.replace(" ", "")
    if not eq or "..." in eq:
        return None
    if "->" in eq:
        lhs, out_sub = eq.split("->", 1)
    else:
        lhs, out_sub = eq, None
    subs = lhs.split(",")
    for s in subs:
        if s and not s.isalpha():
            return None
    if out_sub and not out_sub.isalpha():
        return None
    return subs, out_sub


def _einsum_implicit_out(subs) -> str:
    """Implicit einsum output: indices occurring exactly once across all
    operand subscripts, in alphabetical order."""
    from collections import Counter

    counts = Counter(ch for s in subs for ch in s)
    return "".join(sorted(ch for ch, n in counts.items() if n == 1))


def _parse_einops_axes(side: str):
    """Parse one side of an einops pattern into a list of *groups*.

    Each top-level group is a list of members; a member is an axis-name ``str``
    or an ``int`` literal (e.g. ``1``).  A parenthesised composition ``(h w)``
    becomes a multi-member group.  Returns ``None`` (abstain) on an ellipsis,
    nested/unbalanced parens, or any non-identifier/non-integer token — the
    parts of the einops language we don't model precisely."""
    if "..." in side or "\u2026" in side:
        return None
    tokens = side.replace("(", " ( ").replace(")", " ) ").split()
    groups: List[list] = []
    cur: Optional[list] = None
    for tok in tokens:
        if tok == "(":
            if cur is not None:  # nested composition: abstain
                return None
            cur = []
        elif tok == ")":
            if cur is None:  # unbalanced
                return None
            groups.append(cur)
            cur = None
        else:
            if tok.isdigit():
                member = int(tok)
            elif tok.isidentifier():
                member = tok
            else:
                return None
            (cur if cur is not None else groups).append(
                member if cur is not None else [member]
            )
    if cur is not None:  # unclosed paren
        return None
    return groups


def _parse_einops_pattern(pat: str):
    """Split an einops ``'lhs -> rhs'`` pattern into (lhs_groups, rhs_groups),
    or ``None`` to abstain (no arrow, or either side unparseable)."""
    if "->" not in pat:
        return None
    lhs, rhs = pat.split("->", 1)
    lg = _parse_einops_axes(lhs)
    rg = _parse_einops_axes(rhs)
    if lg is None or rg is None:
        return None
    return lg, rg


def _einops_names(groups) -> List[str]:
    """Flatten the string axis names (ignoring integer literals) of a parsed
    einops side, in left-to-right order."""
    return [m for g in groups for m in g if isinstance(m, str)]


# -- Step 48: input-shape inference from type annotations -------------------
# Annotations are a declared *contract*, so seeding an entry parameter from its
# annotation is sound — any failure we then force is real for the declared type.
_TENSOR_TYPE_NAMES = frozenset(
    {
        "Tensor",
        "FloatTensor",
        "DoubleTensor",
        "HalfTensor",
        "BFloat16Tensor",
        "LongTensor",
        "IntTensor",
        "ShortTensor",
        "CharTensor",
        "ByteTensor",
        "BoolTensor",
        "ndarray",
    }
)
# jaxtyping dtype heads: ``Float[Tensor, "b c h w"]`` etc.
_JAXTYPING_DTYPES = frozenset(
    {
        "Float",
        "Int",
        "Bool",
        "Complex",
        "Float32",
        "Float64",
        "Float16",
        "BFloat16",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt",
        "Num",
        "Shaped",
        "Inexact",
        "Integer",
        "Key",
    }
)


def _tensor_from_shape_spec(spec: str):
    """Build a TensorVal from a whitespace-separated jaxtyping shape string.
    Integer tokens become constant dims; identifier tokens become named symbolic
    dims (size unknown → size checks stay sound); a variadic ``*``/``...`` axis
    yields an unknown-rank tensor (still sound: it is a tensor)."""
    tokens = spec.split()
    if not tokens:
        return TensorVal(rank=0, shape=())
    dims: List[Optional[SymDim]] = []
    for tok in tokens:
        if tok.startswith("*") or "..." in tok:
            return TensorVal(rank=None)  # variadic batch → rank unknown
        t = tok.lstrip("#")  # '#' marks a broadcastable axis in jaxtyping
        if t.isdigit():
            dims.append(SymDim.const_dim(int(t)))
        elif t.isidentifier():
            dims.append(SymDim.var(t))
        else:
            dims.append(None)  # unrecognised axis form → unknown size, keep rank
    return TensorVal(rank=len(dims), shape=tuple(dims))


def _tensor_from_torchtyping(sl) -> "AbstractValue":
    """``TensorType["b", "x", "y"]`` / ``TensorType[3, 4]`` → rank from the number
    of axis elements; ellipsis or a complex slice element → unknown rank."""
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
    dims: List[Optional[SymDim]] = []
    for e in elts:
        if isinstance(e, ast.Constant):
            if e.value is Ellipsis:
                return TensorVal(rank=None)
            if isinstance(e.value, int) and not isinstance(e.value, bool):
                dims.append(SymDim.const_dim(e.value))
            elif isinstance(e.value, str) and e.value.isidentifier():
                dims.append(SymDim.var(e.value))
            else:
                dims.append(None)
        elif isinstance(e, ast.Name):
            dims.append(SymDim.var(e.id))
        else:
            return TensorVal(rank=None)  # slices/keywords → don't guess the rank
    return TensorVal(rank=len(dims), shape=tuple(dims))


def _infer_from_annotation(ann) -> "Optional[AbstractValue]":
    """Best-effort sound abstraction of a parameter from its type annotation, or
    ``None`` when the annotation carries no usable information (caller keeps Top).
    Handles plain tensor/array types, jaxtyping/torchtyping shape annotations, and
    the builtin scalar types."""
    if ann is None:
        return None
    if isinstance(ann, ast.Name):
        if ann.id in _TENSOR_TYPE_NAMES:
            return TensorVal(rank=None)
        if ann.id == "int":
            return IntVal()
        if ann.id == "float":
            return FloatVal()
        if ann.id == "bool":
            return BoolVal()
        if ann.id == "str":
            return StrVal()
        return None
    if isinstance(ann, ast.Attribute):
        if ann.attr in _TENSOR_TYPE_NAMES:
            return TensorVal(rank=None)
        return None
    if isinstance(ann, ast.Subscript):
        base = ann.value
        bname = (
            base.id
            if isinstance(base, ast.Name)
            else (base.attr if isinstance(base, ast.Attribute) else None)
        )
        sl = ann.slice
        if isinstance(sl, ast.Index):  # py<3.9 compatibility
            sl = sl.value
        if bname in _JAXTYPING_DTYPES:
            spec = None
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            for e in elts:
                if isinstance(e, ast.Constant) and isinstance(e.value, str):
                    spec = e.value
            return _tensor_from_shape_spec(spec) if spec is not None else TensorVal(rank=None)
        if bname == "TensorType":
            return _tensor_from_torchtyping(sl)
        return None  # Optional/Union/List[...]/etc. → abstain (Top)
    return None


class _Contradiction:
    """Sentinel returned by guard-refinement helpers when a constraint is
    infeasible, so the caller can mark the branch state unreachable."""

    __slots__ = ()


CONTRA = _Contradiction()


@dataclass
class Frame:
    func: ast.FunctionDef
    returns: List[AbstractValue] = field(default_factory=list)
    returned_explicitly: bool = False
    # structural arity of each reachable ``return`` (number of comma-separated
    # values; 1 for a bare ``return expr``, k for ``return a, b, ...``).  Used to
    # detect the return-arity contract bug independently of the abstract value,
    # which is often ``TOP`` for tensors produced by opaque ops.
    return_arities: List[int] = field(default_factory=list)


class Interpreter:
    def __init__(
        self,
        module: ast.Module,
        filename: str = "<unknown>",
        *,
        config: "SymConfig | None" = None,
    ):
        self.module = module
        self.filename = filename
        # Analysis policy (Step 86).  The default is ``balanced`` and is
        # byte-identical to the engine's historic behaviour, so omitting it is a
        # no-op; ``sound`` reports a subset and ``heuristic`` a superset.
        self.config: "SymConfig" = config if config is not None else DEFAULT_CONFIG
        self.bugs: List[SymBug] = []
        self.funcs_by_id: Dict[int, ast.FunctionDef] = {}
        self.classes: Dict[str, ast.ClassDef] = {}
        # Import-alias map (Step 83): bound name -> canonical dotted module/symbol
        # path, used purely to resolve a call's leading name to a third-party
        # library function for stub lookup.  Populated from the module's actual
        # ``import`` statements so a stub only matches the genuine library symbol.
        self._import_aliases: Dict[str, str] = {}
        self._index_module(module)
        self._frames: List[Frame] = []
        # id(ast.Call) -> True when the resolved callee structurally returns a
        # single (non-tuple) value on every reachable path.
        self._call_single: Dict[int, bool] = {}
        self._last_single: Optional[bool] = None
        # Fixpoint cache (Step 20): ``(id(loop_stmt), canonical-entry) ->
        # (entry_state, loop_head_invariant, reporting_bugs)``.  Re-entering a
        # loop with a lattice-identical entry state reuses the converged
        # invariant and re-emits its bugs (engine de-dups), making analysis
        # deterministic and cutting the nested-loop cost of the widening passes.
        self._loop_cache: Dict[tuple, tuple] = {}
        # Function-summary cache (Step 44): a canonical key over
        # ``(callee, frame-depth, arg/self abstraction incl. provenance)`` maps to
        # ``(return_value, transitive_bugs, single_return_flag)``.  Re-calling a
        # function with an identical abstract context replays the cached return
        # and re-emits its bugs (engine de-dups) instead of re-executing the body.
        # Keying on frame depth keeps the cache a *pure* memoization: a call's
        # result depends only on (func, args, self, remaining-depth-budget), so
        # reuse never changes behavior — only cost.
        self._summary_cache: Dict[str, tuple] = {}
        # Feasibility-gated reporting (Step 52): the dim path-constraints in
        # effect for the statement currently executing, consulted by ``_emit`` so
        # a report whose failing condition is unsatisfiable under the path (or a
        # report on a provably-infeasible/dead path) is suppressed.  Sound: only a
        # Z3-*proved* unsat suppresses; missing solver / unknown keeps the report.
        self._cur_dim_facts: tuple = ()
        # CEGAR refinement (Step 55): every time the accumulated symbolic path
        # facts are proved jointly unsatisfiable, the spurious path is pruned
        # (``state.reachable = False``) and the responsible interpolant (minimal
        # unsat core) is recorded here for diagnostics / abstain accounting.
        self._refinements: List["cegar.Refinement"] = []
        # Disjunctive-state width bound (Step 57): the most alternative path
        # states ``exec_block`` keeps apart before collapsing to a sound join.
        self._disj_bound: int = 8
        # Abstain accounting (Step 59): every time a detector leaves the modeled
        # fragment (unknown rank/dim, unrepresentable affine form, ellipsis
        # pattern, theory backend unavailable, unmodeled construct, …) it records
        # a structured reason here instead of returning ``Top`` silently.  This is
        # purely diagnostic — recording never changes whether a bug fires — so it
        # measures coverage without touching soundness.
        self._abstentions: AbstainLedger = AbstainLedger()
        # Statement-coverage metering (Step 77): a purely diagnostic, append-only
        # tally of how many distinct source statements the engine interpreted
        # with a non-``Top`` value (vs. unmodeled node types or bindings that
        # collapsed to ``Top``).  Like the abstain ledger it is never consulted by
        # any detector and is *not* folded into the proof fingerprint, so it
        # measures analytical reach without touching soundness or reproducibility.
        self._coverage: CoverageMeter = CoverageMeter()
        # Bounded for-loop unrolling during module construction (Step 12): a
        # positive counter while an ``nn.Module.__init__`` body is executing.
        # When set, a ``for i in range(<const N>)`` loop whose body builds up a
        # registered container (``self.layers.append(Block(i))``) is unrolled
        # *precisely* (N sequential iterations, ``i`` bound to each concrete int)
        # so the container accumulates N distinct submodules — exactly what the
        # contract deriver needs.  A symbolic / unbounded / too-large range is
        # NOT unrolled; instead any container the loop appends to is marked
        # opaque so the deriver abstains (never guesses a child count).
        self._constructing: int = 0
        # Intent pack (training-loop hygiene): each loop node is structurally
        # scanned at most once for missing-zero_grad / step-without-backward /
        # backward-without-step.  The scan is state-independent, so running it a
        # single time (rather than on every fixpoint/unroll pass over the node)
        # is both correct and avoids redundant work.
        self._training_loops_checked: set = set()
        # Step 16 — lexical class context for ``super()`` resolution.  While a
        # class's ``__init__`` (or any method) body executes we push the class
        # that *defines* that method, so a zero-arg ``super().__init__()`` can be
        # resolved to the base class *of the defining class* (correct under
        # multi-level inheritance, independent of the runtime ``self`` type).
        self._class_stack: List[ast.ClassDef] = []

    @staticmethod
    def _binding_targets(stmt: ast.stmt) -> List[str]:
        """The simple ``Name`` targets a binding statement writes (used only to
        read their post-state value back for the coverage meter — never affects
        analysis)."""
        names: List[str] = []
        targets: List[ast.expr] = []
        if isinstance(stmt, ast.Assign):
            targets = list(stmt.targets)
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
            targets = [stmt.target]
        for t in targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                names.extend(e.id for e in t.elts if isinstance(e, ast.Name))
        return names

    def _record_coverage(self, stmt: ast.stmt, modeled: bool, state: "State") -> None:
        """Classify ``stmt`` for the coverage meter after its transfer ran.

        Reads already-computed post-state values back (no re-evaluation, so the
        abstain ledger and fingerprint are untouched).  A binding statement is
        non-``Top`` when any of its name targets holds a non-``Top`` value; any
        other modeled statement counts as interpreted."""
        is_binding = isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        if not modeled:
            non_top = False
        elif is_binding:
            names = self._binding_targets(stmt)
            non_top = any(not state.get(n).is_top() for n in names) if names else False
        else:
            non_top = True
        self._coverage.record(stmt, modeled=modeled, is_binding=is_binding, non_top=non_top)

    def _abstain(self, category: AbstainCategory, detector: str, node=None, detail: str = ""):
        """Record an abstain decision (Step 59) and return ``None``.

        Side-effect-only: appends an :class:`AbstainReason` to the ledger so the
        site that abstains can write ``return self._abstain(...)`` and keep its
        exact ``None`` return / control flow.  Never affects which bugs report."""
        self._abstentions.record(
            AbstainReason(
                category=category,
                detector=detector,
                detail=detail,
                line=getattr(node, "lineno", 0) if node is not None else 0,
                col=getattr(node, "col_offset", 0) if node is not None else 0,
                function=self._cur_func_name(),
            )
        )
        return None

    # -- indexing --------------------------------------------------------
    def _index_module(self, module: ast.Module) -> None:
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef):
                self.funcs_by_id[id(node)] = node
            if isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # ``import a.b.c`` binds ``a`` -> ``a``; ``import a.b as c``
                    # binds ``c`` -> ``a.b``.
                    if alias.asname:
                        self._import_aliases[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        self._import_aliases[head] = head
            if isinstance(node, ast.ImportFrom):
                if node.level or not node.module:
                    continue  # relative imports are project-local, not stubbed
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    self._import_aliases[bound] = f"{node.module}.{alias.name}"

    def _canonical_callee(self, func) -> Optional[str]:
        """Resolve a call target to a canonical dotted library path via the
        module's import bindings, or ``None`` when it does not resolve to an
        imported third-party symbol.  ``relu`` (``from torch import relu``) ->
        ``torch.relu``; ``F.relu`` (``import torch.nn.functional as F``) ->
        ``torch.nn.functional.relu``; ``torch.softmax`` -> ``torch.softmax``."""
        if isinstance(func, ast.Name):
            return self._import_aliases.get(func.id)
        if isinstance(func, ast.Attribute):
            chain = _attr_chain(func)
            if not chain:
                return None
            parts = chain.split(".")
            base = self._import_aliases.get(parts[0])
            if base is None:
                return None
            rest = parts[1:]
            return base + "." + ".".join(rest) if rest else base
        return None

    def _stub_call(self, func, pos, kw, node) -> Optional[AbstractValue]:
        """Apply a third-party shape summary (Step 83) for an otherwise-unmodeled
        library call.  Returns the result abstraction, or ``None`` to abstain.
        Stubs never emit bugs — they only add forward shape knowledge."""
        canonical = self._canonical_callee(func)
        summary = stubs.lookup(canonical)
        if summary is None:
            return None
        out = summary(pos, kw)
        if out is None:
            return None
        return self._derive(out, node, f"{canonical}(...)")

    def _class_method_own(self, cls: ast.ClassDef, name: str) -> Optional[ast.FunctionDef]:
        """A method defined *directly* in ``cls``' body (no base lookup)."""
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == name:
                return item
        return None

    def _base_classdefs(self, cls: ast.ClassDef) -> List[ast.ClassDef]:
        """The user-defined base classes of ``cls`` (left-to-right) that are known
        in this module.  ``nn.Module`` / external bases are skipped — we can only
        follow ``__init__`` of classes whose source we have."""
        out: List[ast.ClassDef] = []
        for b in cls.bases:
            name = (
                b.id if isinstance(b, ast.Name)
                else b.attr if isinstance(b, ast.Attribute)
                else None
            )
            if name is not None and name in self.classes:
                out.append(self.classes[name])
        return out

    def _resolve_method(
        self, cls: ast.ClassDef, name: str, _seen: Optional[set] = None
    ) -> Optional[Tuple[ast.FunctionDef, ast.ClassDef]]:
        """MRO-style depth-first lookup of ``name`` starting at ``cls``: the class'
        own body first, then each user base left-to-right.  Returns the method and
        the class that *defines* it (needed for ``super()`` resolution), or
        ``None``.  Cycle-guarded so a malformed inheritance graph can't loop."""
        if _seen is None:
            _seen = set()
        if id(cls) in _seen:
            return None
        _seen.add(id(cls))
        own = self._class_method_own(cls, name)
        if own is not None:
            return own, cls
        for base in self._base_classdefs(cls):
            found = self._resolve_method(base, name, _seen)
            if found is not None:
                return found
        return None

    def _class_method(self, cls: ast.ClassDef, name: str) -> Optional[ast.FunctionDef]:
        """Resolve ``name`` on ``cls`` following the (user-class) inheritance
        chain, so an inherited method (e.g. a base ``forward``) is found."""
        found = self._resolve_method(cls, name)
        return found[0] if found is not None else None

    # -- entry -----------------------------------------------------------
    def _summary_key(
        self, func: ast.FunctionDef, args: Dict[str, AbstractValue], self_val: Optional[AbstractValue]
    ) -> str:
        """Canonical, provenance-aware key for the function-summary cache.

        Includes the current frame depth so a cached result is only reused at the
        same remaining-depth budget (keeping the cache a behavior-preserving
        memoization), and folds in each argument's provenance so a reused return
        carries a derivation chain that matches its call context."""
        parts = [str(id(func)), str(len(self._frames))]
        for name in sorted(args):
            v = args[name]
            parts.append(name)
            parts.append(dumps(v))
            parts.append(repr(tuple(getattr(v, "provenance", ()) or ())))
        if self_val is not None:
            parts.append("\x00self")
            if isinstance(self_val, ModuleVal):
                parts.append(self_val.class_name)
                for attr, val in sorted(self_val.attrs):
                    parts.append(attr)
                    parts.append(dumps(val))
                    parts.append(repr(tuple(getattr(val, "provenance", ()) or ())))
            else:
                parts.append(dumps(self_val))
        return "\x01".join(parts)

    def run_function(
        self, func: ast.FunctionDef, args: Dict[str, AbstractValue], self_val: Optional[AbstractValue] = None
    ) -> AbstractValue:
        if len(self._frames) >= _MAX_DEPTH:
            return TOP
        key = self._summary_key(func, args, self_val)
        cached = self._summary_cache.get(key)
        if cached is not None:
            ret, cbugs, single = cached
            self.bugs.extend(cbugs)  # re-emit transitive bugs; engine de-dups
            self._last_single = single
            return ret
        mark = len(self.bugs)
        state = State()
        if self_val is not None:
            state.set("self", self_val)
            if isinstance(self_val, ModuleVal):
                state.store["self"] = {k: v for k, v in self_val.attrs}
        self._bind_params(func, args, state, has_self=self_val is not None)
        frame = Frame(func=func)
        self._frames.append(frame)
        try:
            self.exec_block(func.body, state)
        finally:
            self._frames.pop()
        # structural single-return: at least one reachable return, and every
        # reachable return yields exactly one (non-tuple) value.
        self._last_single = (
            bool(frame.return_arities) and all(a == 1 for a in frame.return_arities)
        )
        if not frame.returns:
            ret: AbstractValue = NONE if not frame.returned_explicitly else TOP
        else:
            ret = join_many(frame.returns)
        self._summary_cache[key] = (ret, list(self.bugs[mark:]), self._last_single)
        return ret

    def _bind_params(
        self, func: ast.FunctionDef, args: Dict[str, AbstractValue], state: State, has_self: bool
    ) -> None:
        pa = func.args
        arg_objs = list(pa.args)
        if has_self and arg_objs and arg_objs[0].arg == "self":
            arg_objs = arg_objs[1:]
        names = [a.arg for a in arg_objs]
        # defaults align to the tail of positional params
        defaults = list(pa.defaults)
        ndef = len(defaults)
        for i, a in enumerate(arg_objs):
            name = a.arg
            if name in args:
                state.set(name, args[name])
            elif i >= len(names) - ndef:
                d = defaults[i - (len(names) - ndef)]
                state.set(name, self.eval_expr(d, state))
            else:
                # Step 48: seed an unbound entry parameter from its annotation
                # (a sound contract) so shape/rank checks engage even when the
                # function has no caller/demo; fall back to Top.
                state.set(name, _infer_from_annotation(a.annotation) or TOP)
        for a in pa.kwonlyargs:
            if a.arg in args:
                state.set(a.arg, args[a.arg])
            else:
                state.set(a.arg, _infer_from_annotation(a.annotation) or TOP)

    # -- statement execution --------------------------------------------
    # Statement kinds across which it is safe to keep branch states *apart* as a
    # disjunction (Step 57): pure straight-line statements with no control-flow
    # escape.  Anything else (returns, loops, nested compounds, …) first
    # collapses the disjunction back to a single state, so the path-sensitive
    # window is confined to the assignments/expressions following a branch.
    _DISJ_CONTINUABLE = (
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.Expr,
        ast.Pass,
    )

    def exec_block(self, stmts: List[ast.stmt], state: State) -> State:
        """Execute a statement block.

        Equivalent to threading a single :class:`State`, but branch states are
        kept apart as a bounded :class:`DisjunctiveState` (Step 57) so the
        straight-line code following an ``if`` is analysed on each path
        precisely; the disjunction collapses (sound join) at the next
        control-flow boundary, on overflow, and at the block's end.  With a
        single path throughout, behaviour is byte-identical to the old loop."""
        disj = DisjunctiveState.singleton(state, bound=self._disj_bound)
        for stmt in stmts:
            disj = disj.live()
            if disj.is_empty():
                break
            # Collapse to one state before any non-straight-line statement so
            # control flow (returns/loops/…) is handled in single-state mode.
            if disj.width() > 1 and not isinstance(stmt, self._DISJ_CONTINUABLE):
                disj = DisjunctiveState.singleton(disj.collapse(), bound=self._disj_bound)
            if isinstance(stmt, ast.If):
                disj = disj.flat_map(lambda s, _st=stmt: self._if_branches(_st, s))
            else:
                disj = disj.map(lambda s, _st=stmt: self.exec_stmt(_st, s))
        return disj.collapse()

    def _if_branches(self, stmt: ast.If, state: State) -> List[State]:
        """The disjuncts an ``if`` produces from one entry state: the executed
        then/else branch-exit states (Step 57).  Statically-decided guards yield
        a single branch; otherwise both, each guard-refined.  This is exactly
        :meth:`_st_If` without the eager join of the two branch states."""
        cond = self.eval_expr(stmt.test, state)
        self._check_bool_context(stmt.test, cond)
        truth = _known_truth(cond)
        if truth is True:
            return [self.exec_block(stmt.body, state)]
        if truth is False:
            return [self.exec_block(stmt.orelse, state)]
        then_state = state.copy()
        else_state = state.copy()
        self._refine(stmt.test, True, then_state)
        self._refine(stmt.test, False, else_state)
        then_state = self.exec_block(stmt.body, then_state)
        else_state = self.exec_block(stmt.orelse, else_state)
        if self._constructing:
            # During module construction an unresolved guard must collapse
            # eagerly (Step 15) so conditionally-registered submodules can be
            # marked opaque on the merged ``self`` store; keeping the branches
            # disjoint would let a later collapse silently union them as
            # unconditionally-present (unsound).  Forward analysis (where the
            # disjunctive precision matters) never runs with ``_constructing``.
            joined = then_state.join(else_state)
            self._opacify_conditional_submodules(then_state, else_state, joined)
            return [joined]
        return [then_state, else_state]

    def exec_stmt(self, stmt: ast.stmt, state: State) -> State:
        # Publish the current path's dim constraints so report sites reached
        # while executing this statement can be feasibility-gated (Step 52).
        self._cur_dim_facts = state.dim_facts
        m = getattr(self, f"_st_{type(stmt).__name__}", None)
        if m is None:
            self._record_coverage(stmt, modeled=False, state=state)
            return state  # unmodeled statement: no effect, stay sound
        result = m(stmt, state)
        self._record_coverage(stmt, modeled=True, state=result)
        return result

    def _emit(self, bug: "SymBug", conditions: tuple = ()) -> None:
        """Single reporting choke point with a feasibility gate (Step 52).

        A report is suppressed only when Z3 *proves* that the current path
        constraints conjoined with ``conditions`` (the failing condition, when a
        detector can express it as ``DimConstraint``s) are unsatisfiable — i.e.
        the fault is unreachable.  Missing solver / ``unknown`` keeps the report,
        so the zero-false-positive guarantee is never traded for a false
        negative we cannot prove."""
        facts = self._cur_dim_facts
        smt_checked = False
        if facts or conditions:
            smt_checked = True
            combined = [*facts, *conditions]
            ref = cegar.refine(combined)
            if ref.spurious:
                # Spurious abstract fault: the failing condition is unreachable
                # under the path.  Record the refuting interpolant and suppress.
                self._refinements.append(ref)
                return
            # Step 86 ``sound`` mode: keep a path-conditioned report only when the
            # solver can *positively* confirm the path (plus the failing
            # condition) is satisfiable.  ``balanced`` keeps it on ``unknown`` /
            # missing solver (defence against false negatives we cannot prove);
            # ``sound`` instead drops it (defence against the residual false
            # positive), making sound a strict subset of balanced.
            if self.config.require_feasibility and not smt_bridge.feasible(combined):
                return
        # Confidence calibration (Step 63): raise the detector's prior by the
        # corroborating evidence gathered for this report.  Presentation-only —
        # never changes whether the bug is reported, only how strongly it ranks.
        signals = ConfidenceSignals.from_evidence(
            bug.evidence,
            has_path_constraints=bool(facts or conditions),
            smt_checked=smt_checked,
        )
        calibrated = confidence_mod.calibrate(bug.confidence, signals)
        if calibrated != bug.confidence:
            bug = replace(bug, confidence=calibrated)
        # Step 86 confidence floor: a pure triage gate.  ``balanced`` uses a 0.0
        # floor so nothing is filtered (byte-identical to the historic
        # behaviour); ``sound`` discards weak-prior findings.
        if not self.config.allows_confidence(bug.confidence):
            return
        self.bugs.append(bug)

    def _st_Assign(self, stmt: ast.Assign, state: State) -> State:
        value = self.eval_expr(stmt.value, state)
        for target in stmt.targets:
            self._assign_to(target, value, state, stmt, rhs=stmt.value)
        return state

    def _st_AnnAssign(self, stmt: ast.AnnAssign, state: State) -> State:
        if stmt.value is not None:
            value = self.eval_expr(stmt.value, state)
            self._assign_to(stmt.target, value, state, stmt, rhs=stmt.value)
        return state

    def _st_AugAssign(self, stmt: ast.AugAssign, state: State) -> State:
        # result type usually matches the LHS; keep it simple and sound
        val = self.eval_expr(stmt.value, state)
        cur = self.eval_expr(stmt.target, state) if isinstance(stmt.target, ast.Name) else TOP
        res = cur if isinstance(cur, TensorVal) else (val if isinstance(val, TensorVal) else TOP)
        self._assign_to(stmt.target, res, state, stmt)
        return state

    def _st_Expr(self, stmt: ast.Expr, state: State) -> State:
        self.eval_expr(stmt.value, state)
        self._check_discarded_transform(stmt, state)
        return state

    def _check_missing_super_init(self, stmt: ast.ClassDef) -> None:
        """Heuristic (Step 86 ``heuristic`` mode): an ``nn.Module`` subclass whose
        ``__init__`` never calls ``super().__init__()`` (or ``nn.Module.__init__``)
        leaves the module's internal state uninitialised, so parameter/submodule
        registration silently breaks.  It often does not raise at definition time,
        so this is heuristic-only and suppressed in ``sound``/``balanced``."""
        if not self.config.enable_heuristics:
            return
        if not any(_is_module_base(b) for b in stmt.bases):
            return
        init = next(
            (n for n in stmt.body
             if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
            None,
        )
        if init is None:
            return  # inherits nn.Module.__init__ — fine
        for sub in ast.walk(init):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "__init__"
            ):
                return  # some __init__ (super().__init__ / Base.__init__) is called
        self._emit(
            SymBug(
                kind=SymBugKind.MISSING_SUPER_INIT,
                message=(
                    f"nn.Module subclass {stmt.name!r} defines __init__ but never "
                    f"calls super().__init__(); parameter/submodule registration "
                    f"will not work"
                ),
                line=getattr(init, "lineno", getattr(stmt, "lineno", 0)),
                col=getattr(init, "col_offset", 0),
                function=stmt.name,
                severity="warning",
                confidence=0.8,
                fix_suggestion="add super().__init__() as the first statement of __init__",
            )
        )

    def _check_tensor_data_access(self, node: ast.Attribute) -> None:
        """Heuristic (Step 86 ``heuristic`` mode): accessing ``tensor.data``
        bypasses autograd tracking — a well-known footgun.  In-place edits through
        ``.data`` silently skip the graph and produce wrong gradients.  It does not
        crash, so it is heuristic-only; ``.detach()`` is the safe replacement."""
        self._emit(
            SymBug(
                kind=SymBugKind.TENSOR_DATA_ACCESS,
                message=(
                    ".data bypasses autograd tracking and is an error-prone "
                    "footgun; prefer .detach()"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                severity="warning",
                confidence=0.6,
                fix_suggestion="use .detach() instead of .data",
            )
        )

    def _check_direct_forward(self, node, func: ast.Attribute) -> None:
        """Heuristic (Step 86 ``heuristic`` mode): calling ``module.forward(x)``
        directly instead of ``module(x)`` bypasses ``nn.Module.__call__`` and the
        registered forward/​pre-forward hooks.  It does not crash but is a
        well-known anti-pattern almost always written by mistake.  Suppressed in
        ``sound``/``balanced``.  ``super().forward(...)`` is not a ``ModuleVal``
        receiver, so legitimate base-class delegation is never flagged."""
        if not self.config.enable_heuristics:
            return
        self._emit(
            SymBug(
                kind=SymBugKind.DIRECT_FORWARD_CALL,
                message=(
                    "calling .forward() directly bypasses nn.Module.__call__ and "
                    "its registered hooks; call the module instance instead"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                severity="warning",
                confidence=0.7,
                fix_suggestion="use module(x) instead of module.forward(x)",
            )
        )

    def _check_discarded_transform(self, stmt: ast.Expr, state: State) -> None:
        """Heuristic (Step 86 ``heuristic`` mode): a bare statement that is a pure
        out-of-place tensor transform (``x.to(...)``/``x.cuda()``/``x.reshape(...)``)
        discards its only result — a no-op the author almost certainly meant to
        assign.  Never raises, so this is recall-only and is suppressed in
        ``sound``/``balanced`` (keeping their zero-false-positive guarantee)."""
        if not self.config.enable_heuristics:
            return
        call = stmt.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            return
        method = call.func.attr
        if method not in _PURE_TENSOR_TRANSFORMS:
            return
        recv = self.eval_expr(call.func.value, state)
        if not isinstance(recv, TensorVal):
            return
        self._emit(
            SymBug(
                kind=SymBugKind.DISCARDED_TENSOR_RESULT,
                message=(
                    f".{method}() returns a new tensor and has no in-place effect, "
                    f"but its result is discarded; this statement is a no-op "
                    f"(did you mean to assign it back?)"
                ),
                line=getattr(stmt, "lineno", 0),
                col=getattr(stmt, "col_offset", 0),
                function=self._cur_func_name(),
                severity="warning",
                confidence=0.7,
                fix_suggestion=f"assign the result, e.g. x = x.{method}(...)",
            )
        )

    # ------------------------------------------------------------------
    # Intent pack — training-loop hygiene ("why isn't it training?")
    # ------------------------------------------------------------------
    @staticmethod
    def _loop_method_calls(body: List[ast.stmt], names: set) -> Dict[str, List[ast.Call]]:
        """Collect, per method name in ``names``, every ``recv.<name>(...)`` call
        appearing in ``body`` — but NOT descending into nested function/lambda
        scopes (a closure defined in the loop is a different execution context and
        its calls do not run per loop iteration).  Pure structural walk; never
        evaluates receivers, so it is robust on un-analysable training loops."""
        found: Dict[str, List[ast.Call]] = {n: [] for n in names}

        def walk(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue  # different scope — not part of this loop body
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr in names
                ):
                    found[child.func.attr].append(child)
                walk(child)

        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # a def in the loop body is a separate scope
            walk(stmt)
        return found

    def _check_training_loop(self, stmt: ast.stmt) -> None:
        """Heuristic (``heuristic`` mode only): structural hygiene of a PyTorch
        training loop.  A loop whose body calls ``.backward()`` and/or
        ``optimizer.step()`` is a training step; three classic *silent* mistakes
        — code reasons a model "isn't training" — are flagged:

        * ``MISSING_ZERO_GRAD`` — ``.backward()`` and ``.step()`` present but no
          ``zero_grad()``: gradients accumulate across iterations, so every update
          uses the sum of all prior batches' gradients (training silently wrong).
        * ``STEP_WITHOUT_BACKWARD`` — ``.step()`` and ``zero_grad()`` present (an
          unambiguous optimizer) but no ``.backward()``: parameters never receive
          gradients, so the model never learns.
        * ``BACKWARD_WITHOUT_STEP`` — ``.backward()`` present but no ``.step()``:
          gradients are computed and then never applied to the parameters.

        None of these raise at runtime, so they are heuristic-only and suppressed
        in ``sound``/``balanced`` (preserving the zero-false-positive guarantee).
        The scan is gated outside module construction (training loops never run in
        ``__init__``) and runs at most once per loop node."""
        if self._constructing or not self.config.enable_heuristics:
            return
        key = id(stmt)
        if key in self._training_loops_checked:
            return
        self._training_loops_checked.add(key)
        body = getattr(stmt, "body", None)
        if not body:
            return
        calls = self._loop_method_calls(body, {"backward", "step", "zero_grad"})
        has_backward = bool(calls["backward"])
        has_step = bool(calls["step"])
        has_zero_grad = bool(calls["zero_grad"])
        line = getattr(stmt, "lineno", 0)
        col = getattr(stmt, "col_offset", 0)
        func = self._cur_func_name()

        if has_backward and has_step and not has_zero_grad:
            self._emit(
                SymBug(
                    kind=SymBugKind.MISSING_ZERO_GRAD,
                    message=(
                        "training loop calls .backward() and optimizer.step() but "
                        "never zero_grad(); gradients accumulate across iterations, "
                        "so each step uses the sum of all previous batches' gradients"
                    ),
                    line=line,
                    col=col,
                    function=func,
                    severity="warning",
                    confidence=0.75,
                    fix_suggestion="call optimizer.zero_grad() before loss.backward()",
                )
            )
        elif has_step and has_zero_grad and not has_backward:
            self._emit(
                SymBug(
                    kind=SymBugKind.STEP_WITHOUT_BACKWARD,
                    message=(
                        "training loop calls optimizer.zero_grad()/step() but never "
                        ".backward(); no gradients are computed, so the parameters "
                        "never update and the model does not train"
                    ),
                    line=line,
                    col=col,
                    function=func,
                    severity="warning",
                    confidence=0.7,
                    fix_suggestion="call loss.backward() between zero_grad() and step()",
                )
            )
        elif has_backward and not has_step:
            self._emit(
                SymBug(
                    kind=SymBugKind.BACKWARD_WITHOUT_STEP,
                    message=(
                        "training loop calls .backward() but no optimizer.step(); "
                        "gradients are computed and then never applied, so the "
                        "parameters never update"
                    ),
                    line=line,
                    col=col,
                    function=func,
                    severity="warning",
                    confidence=0.6,
                    fix_suggestion="call optimizer.step() after loss.backward()",
                )
            )

    def _st_Assert(self, stmt: ast.Assert, state: State) -> State:
        # An ``assert`` that survives narrows the post-state exactly like the
        # then-branch of ``if <test>:`` — every later statement may assume it.
        self.eval_expr(stmt.test, state)
        self._refine(stmt.test, True, state)
        return state

    def _st_Return(self, stmt: ast.Return, state: State) -> State:
        frame = self._frames[-1] if self._frames else None
        if frame is not None:
            frame.returned_explicitly = True
            if stmt.value is not None:
                frame.returns.append(self.eval_expr(stmt.value, state))
                if isinstance(stmt.value, ast.Tuple):
                    frame.return_arities.append(len(stmt.value.elts))
                else:
                    frame.return_arities.append(1)
            else:
                frame.returns.append(NONE)
                frame.return_arities.append(1)
        state.reachable = False
        return state

    def _st_If(self, stmt: ast.If, state: State) -> State:
        cond = self.eval_expr(stmt.test, state)
        self._check_bool_context(stmt.test, cond)
        truth = _known_truth(cond)
        if truth is True:
            return self.exec_block(stmt.body, state)
        if truth is False:
            return self.exec_block(stmt.orelse, state)
        then_state = state.copy()
        else_state = state.copy()
        # Path-sensitive refinement: narrow the abstract values of variables
        # mentioned in the guard so each branch sees the facts the guard
        # establishes (e.g. ``n`` excludes 0 after ``if n == 0: return``).
        self._refine(stmt.test, True, then_state)
        self._refine(stmt.test, False, else_state)
        then_state = self.exec_block(stmt.body, then_state)
        else_state = self.exec_block(stmt.orelse, else_state)
        joined = then_state.join(else_state)
        if self._constructing:
            self._opacify_conditional_submodules(then_state, else_state, joined)
        return joined

    # -- Step 15: conditional submodule registration ----------------------
    def _opacify_conditional_submodules(
        self, then_state: State, else_state: State, joined: State
    ) -> None:
        """Mark every ``self.<attr>`` submodule whose *existence or identity*
        differs between the two branches of an unresolved construction-time
        ``if`` as an opaque conditional module, so the contract deriver abstains
        on that subtree only (never inventing nor silently dropping its params).

        Only meaningful while building an ``nn.Module.__init__`` (gated by the
        caller on ``self._constructing``), so forward analysis is untouched.
        When either branch is unreachable the assignment is unconditional on the
        surviving path, so nothing is opacified."""
        if not (then_state.reachable and else_state.reachable):
            return
        then_self = then_state.store.get("self", {})
        else_self = else_state.store.get("self", {})
        joined_self = joined.store.get("self")
        if joined_self is None:
            return
        _MISSING = object()
        for attr in set(then_self) | set(else_self):
            tv = then_self.get(attr, _MISSING)
            ev = else_self.get(attr, _MISSING)
            if tv is ev:
                continue  # same object on both paths -> unconditional
            if tv == ev:
                continue  # equal value on both paths -> unconditional
            # The attribute is registered/identified conditionally.  Only a
            # value that the contract would walk (a submodule) needs opacifying;
            # plain ints/flags are ignored by the deriver already.
            present = [v for v in (tv, ev) if v is not _MISSING]
            if not any(isinstance(v, ModuleVal) for v in present):
                continue
            cls = next(
                (v.class_name for v in present if isinstance(v, ModuleVal)),
                "?",
            )
            joined_self[attr] = ModuleVal(
                class_name=cls, meta=(("__conditional__", 1),)
            )

    # -- path-sensitive guard refinement ----------------------------------
    def _refine(self, test: ast.expr, want: bool, state: State) -> None:
        """Mutate ``state`` so that the guard ``test`` is assumed to evaluate to
        ``want``.  Sound by construction: when a fact cannot be represented it is
        simply dropped (no narrowing), never over-claimed."""
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            self._refine(test.operand, not want, state)
            return
        if isinstance(test, ast.BoolOp):
            # ``a and b`` true  ⇒ both true; ``a or b`` false ⇒ both false.
            if isinstance(test.op, ast.And) and want:
                for v in test.values:
                    self._refine(v, True, state)
            elif isinstance(test.op, ast.Or) and not want:
                for v in test.values:
                    self._refine(v, False, state)
            return
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            self._refine_compare(test, want, state)
            return
        if isinstance(test, ast.Name):
            cur = state.get(test.id)
            if isinstance(cur, IntVal):
                if want:
                    nv = _exclude_zero(cur)
                    if nv is CONTRA:
                        state.reachable = False
                    elif isinstance(nv, IntVal):
                        state.set(test.id, nv)
                else:
                    if _provably_nonzero(cur):
                        state.reachable = False  # 0 required but value is non-zero
                    else:
                        state.set(test.id, int_const(0))
            elif isinstance(cur, NoneVal) and want:
                # ``None`` is falsy; the truthy branch is infeasible for it.
                state.reachable = False
            return

    def _refine_compare(self, cmp: ast.Compare, want: bool, state: State) -> None:
        op = cmp.ops[0]
        left, right = cmp.left, cmp.comparators[0]
        # Record a symbolic-dimension path fact when both operands are dimension
        # expressions (e.g. ``a.size(0) == b.size(1)``) so feasibility-gated
        # reporting (Step 52) can later rule out unreachable faults.
        self._record_dim_fact(op, left, right, want, state)
        name: Optional[str] = None
        other: Optional[ast.expr] = None
        swapped = False
        if isinstance(left, ast.Name):
            name, other = left.id, right
        elif isinstance(right, ast.Name):
            name, other, swapped = right.id, left, True
        if name is None or other is None:
            return

        # ``x is None`` / ``x is not None``
        if isinstance(op, (ast.Is, ast.IsNot)) and _is_none_literal(other):
            cond_is_none = isinstance(op, ast.Is)
            means_none = cond_is_none if want else not cond_is_none
            cur = state.get(name)
            if means_none:
                if _is_definitely_non_none(cur):
                    state.reachable = False  # known non-None, can't be None here
                else:
                    state.set(name, NONE)
            else:
                if isinstance(cur, NoneVal):
                    state.reachable = False  # known None, can't be non-None here
            return

        # ``x == c`` / ``x != c`` / ``x < c`` … with a constant ``c``
        c = _const_int_of(other)
        if c is None:
            return
        cur = state.get(name)
        if not isinstance(cur, IntVal):
            return  # only narrow values we already know to be ints (sound)
        eff = _swap_op(op) if swapped else _op_name(op)
        if eff is None:
            return
        if not want:
            eff = _negate_op(eff)
        nv = _narrow_int(cur, eff, c)
        if nv is CONTRA:
            state.reachable = False
        elif isinstance(nv, IntVal):
            state.set(name, nv)

    # -- Step 52: symbolic-dimension path facts ---------------------------
    _CMP_OP_TO_REL = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    }
    _REL_NEGATE = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}

    def _symdim_of_expr(self, node: ast.expr, state: State) -> Optional[SymDim]:
        """Return the affine ``SymDim`` an expression denotes, when it is a
        dimension query (``t.size(i)`` / ``t.shape[i]`` / ``t.ndim`` …) or an
        integer that participates in dim arithmetic.  ``None`` otherwise."""
        try:
            val = self.eval_expr(node, state)
        except Exception:  # pragma: no cover - eval is side-effecting but safe
            return None
        if isinstance(val, IntVal) and val.sym is not None:
            return val.sym
        return None

    def _record_dim_fact(self, op, left, right, want: bool, state: State) -> None:
        rel = self._CMP_OP_TO_REL.get(type(op))
        if rel is None:
            return
        if not want:
            rel = self._REL_NEGATE[rel]
        lsym = self._symdim_of_expr(left, state)
        rsym = self._symdim_of_expr(right, state)
        if lsym is None or rsym is None:
            return
        # A constraint between two constants carries no information.
        if lsym.is_const and rsym.is_const:
            return
        fact = smt_bridge.DimConstraint(lsym, rel, rsym)
        if fact not in state.dim_facts:
            state.dim_facts = state.dim_facts + (fact,)
            # CEGAR (Step 55): the freshly-asserted guard may contradict an
            # enclosing one (e.g. ``if a != b: if a == b:``).  When the path
            # facts are now *proved* jointly unsatisfiable, the branch is dead
            # for every concrete shape — prune it and record the interpolant.
            # Sound: only a Z3-proved contradiction prunes; unknown/no-z3 keeps
            # the path reachable.
            ref = cegar.refine(state.dim_facts)
            if ref.spurious:
                state.reachable = False
                self._refinements.append(ref)


    def _st_For(self, stmt: ast.For, state: State) -> State:
        self._check_training_loop(stmt)
        # Step 12 — bounded precise unrolling during module construction.  When
        # an ``nn.Module.__init__`` runs a ``for i in range(<const N>)`` loop to
        # build a registered container (``self.layers.append(Block(i))``), the
        # generic widening fixpoint below would JOIN iterations and never resolve
        # the N distinct children.  If the trip count is a statically-known finite
        # constant (<= cap) and the body has no break/continue/return, we instead
        # execute the body N times sequentially, binding the target to each
        # concrete int and threading the (mutating) state forward, so the
        # container accumulates exactly N submodules.  Precise unrolling is a
        # sound *refinement* of the fixpoint (it executes the real iterations).
        if self._constructing:
            values = self._static_for_values(stmt.iter, state)
            if (
                values is not None
                and len(values) <= _MAX_CONSTRUCT_UNROLL
                and not _has_loop_control(stmt.body)
            ):
                return self._unroll_for(stmt, values, state)
            # Not precisely unrolled while constructing (symbolic / unbounded /
            # too-large trip count, or break/continue/return in the body): any
            # registered container the body appends to cannot be enumerated, so
            # mark it opaque — the deriver then abstains rather than emitting a
            # guessed (and unsound) child count.
            self._opacify_appended_containers(stmt.body, state)

        # Sound over-approximation of "0 or more iterations" via a widening
        # fixpoint (Steps 17/38): the loop-head invariant over-approximates the
        # state before *every* iteration, so multi-iteration data-flow bugs are
        # caught and accumulating values (counters, growing ranges) terminate.
        iter_val = self.eval_expr(stmt.iter, state)
        elem = _element_of(iter_val)

        def enter(s: State) -> State:
            s2 = s.copy()
            self._assign_to(stmt.target, elem, s2, stmt)
            return s2

        exit_state = self._run_loop(stmt, stmt.body, state, enter)
        return self.exec_block(stmt.orelse, exit_state) if stmt.orelse else exit_state

    def _static_for_values(self, iter_node: ast.expr, state: State) -> Optional[List[AbstractValue]]:
        """The explicit, ordered list of concrete loop-variable values for a
        statically-enumerable iterable, else ``None`` (abstain → fixpoint).

        Handles ``range(<const>[, <const>[, <const>]])`` and a literal
        list/tuple of values.  Returns ``None`` for any symbolic bound, a
        zero step, or an unrecognised iterable."""
        # range(...) with constant arguments
        if (
            isinstance(iter_node, ast.Call)
            and isinstance(iter_node.func, ast.Name)
            and iter_node.func.id == "range"
            and not any(isinstance(a, ast.Starred) for a in iter_node.args)
            and not iter_node.keywords
            and 1 <= len(iter_node.args) <= 3
        ):
            consts: List[int] = []
            for a in iter_node.args:
                v = self.eval_expr(a, state)
                if not (isinstance(v, IntVal) and v.const is not None):
                    return None
                consts.append(v.const)
            if len(consts) == 1:
                rng = range(consts[0])
            elif len(consts) == 2:
                rng = range(consts[0], consts[1])
            else:
                if consts[2] == 0:
                    return None
                rng = range(consts[0], consts[1], consts[2])
            return [int_const(i) for i in rng]
        # a literal list/tuple of elements — iterate the concrete elements
        if isinstance(iter_node, (ast.List, ast.Tuple)) and not any(
            isinstance(e, ast.Starred) for e in iter_node.elts
        ):
            return [self.eval_expr(e, state) for e in iter_node.elts]
        return None

    def _unroll_for(self, stmt: ast.For, values: List[AbstractValue], state: State) -> State:
        """Execute ``stmt.body`` once per value in ``values`` (sequentially,
        threading the mutating state forward), binding ``stmt.target`` to each."""
        s = state
        for cval in values:
            self._assign_to(stmt.target, cval, s, stmt)
            s = self.exec_block(stmt.body, s)
            if not s.reachable:
                break
        return self.exec_block(stmt.orelse, s) if stmt.orelse else s

    def _opacify_appended_containers(self, body, state: State) -> None:
        """Mark every registered container (``ModuleList``/``ModuleDict``) that
        ``body`` mutates via ``.append``/``.extend``/``.insert``/``.update`` as
        opaque, so a loop we cannot enumerate yields an abstention rather than a
        guessed (unsound) child count.  Targets a simple ``Name`` or
        ``self.<attr>`` receiver only; anything else is already over-approximated."""
        seen: set = set()
        for stmt in body:
            for n in ast.walk(stmt):
                if not (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("append", "extend", "insert", "update")
                ):
                    continue
                recv_node = n.func.value
                key = ast.dump(recv_node)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    recv = self.eval_expr(recv_node, state)
                except Exception:
                    continue
                if (
                    isinstance(recv, ModuleVal)
                    and recv.class_name in ("ModuleList", "ModuleDict")
                    and recv.get_meta("__opaque_container__") != 1
                ):
                    opaque = ModuleVal(
                        class_name=recv.class_name, meta=self._OPAQUE_CONTAINER_META
                    )
                    self._rebind_lvalue(recv_node, opaque, state)

    def _st_While(self, stmt: ast.While, state: State) -> State:
        self._check_training_loop(stmt)
        cond = self.eval_expr(stmt.test, state)
        self._check_bool_context(stmt.test, cond)
        if _known_truth(cond) is False:
            return self.exec_block(stmt.orelse, state) if stmt.orelse else state

        def enter(s: State) -> State:
            s2 = s.copy()
            self._refine(stmt.test, True, s2)
            return s2

        exit_state = self._run_loop(stmt, stmt.body, state, enter)
        # On normal exit the guard is false; refine for precision (sound: every
        # fall-through path has a false guard).  ``break`` paths are already
        # folded into the invariant, so this only tightens, never over-claims.
        self._refine(stmt.test, False, exit_state)
        return self.exec_block(stmt.orelse, exit_state) if stmt.orelse else exit_state

    def _run_loop(self, stmt: ast.stmt, body, entry: State, enter) -> State:
        """Drive a loop body to a sound loop-head invariant.

        ``enter(s)`` returns the body-entry state derived from a loop-head state
        ``s`` (binding the ``for`` target or assuming a ``while`` guard).  The
        returned state is the loop-head invariant — a sound over-approximation
        of the state on *every* iteration and therefore of the post-loop state.

        Strategy: unroll ``_LOOP_UNROLL`` iterations precisely **with reporting**
        (catching certain bugs at their exact early-iteration states), then run a
        widening Kleene iteration **silently** for the steady state, then a
        **narrowing** pass that recovers precision the widening over-shot (±∞
        bounds the loop guard actually constrains), then one final reporting pass
        from the converged invariant.  The engine de-duplicates identical reports
        across these passes.

        A fixpoint cache (Step 20) short-circuits the whole computation when this
        very loop has already been analysed from a lattice-identical entry: the
        converged invariant is reused and the loop's reporting bugs are re-emitted
        (so a cache hit inside a later reporting pass never drops a bug), which
        keeps results deterministic and bounds the cost of nested loops.
        """
        key = (id(stmt), dumps(entry))
        cached = self._loop_cache.get(key)
        if cached is not None:
            c_entry, c_inv, c_bugs = cached
            if c_entry.equals(entry):  # exact structural match — safe to reuse
                self.bugs.extend(c_bugs)
                return c_inv.copy()

        start = len(self.bugs)
        head = entry
        inv = entry
        converged_early = False
        # -- precise unrolling (reporting on) --------------------------------
        for _ in range(_LOOP_UNROLL):
            be = enter(head)
            if not be.reachable:
                inv, converged_early = head, True
                break
            after = self.exec_block(body, be)
            nxt = entry.join(after)
            if nxt.equals(head):
                inv, converged_early = head, True  # already a fixpoint
                break
            head = nxt
        if not converged_early:
            # -- widening fixpoint (silent) ----------------------------------
            mark = len(self.bugs)
            inv = head
            for _ in range(_LOOP_FIX_MAX):
                be = enter(inv)
                if not be.reachable:
                    break
                after = self.exec_block(body, be)
                widened = inv.widen(entry.join(after))
                if widened.equals(inv):
                    break
                inv = widened
            # -- narrowing pass: recover precision lost to widening (silent) -
            for _ in range(_LOOP_NARROW_MAX):
                be = enter(inv)
                if not be.reachable:
                    break
                after = self.exec_block(body, be)
                refined = inv.narrow(entry.join(after))
                if refined.equals(inv):
                    break
                inv = refined
            del self.bugs[mark:]  # discard reports from the invariant search
            # -- final reporting pass from the converged invariant -----------
            be = enter(inv)
            if be.reachable:
                self.exec_block(body, be)

        self._loop_cache[key] = (entry.copy(), inv.copy(), list(self.bugs[start:]))
        return inv

    def _st_With(self, stmt: ast.With, state: State) -> State:
        for item in stmt.items:
            v = self.eval_expr(item.context_expr, state)
            if item.optional_vars is not None:
                self._assign_to(item.optional_vars, v, state, stmt)
        return self.exec_block(stmt.body, state)

    def _st_Try(self, stmt: ast.Try, state: State) -> State:
        body_state = self.exec_block(stmt.body, state.copy())
        merged = state.join(body_state)
        for handler in stmt.handlers:
            merged = merged.join(self.exec_block(handler.body, state.copy()))
        if stmt.orelse:
            merged = self.exec_block(stmt.orelse, merged)
        if stmt.finalbody:
            merged = self.exec_block(stmt.finalbody, merged)
        return merged

    # -- assignment / unpacking -----------------------------------------
    def _assign_to(
        self, target: ast.expr, value: AbstractValue, state: State, ctx: ast.stmt, rhs: Optional[ast.expr] = None
    ) -> None:
        if isinstance(target, ast.Name):
            state.set(target.id, value)
            return
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            state.set_attr(target.value.id, target.attr, value)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            self._assign_unpack(target, value, state, ctx, rhs=rhs)
            return
        if isinstance(target, ast.Subscript):
            # x[i] = v : bounds-check the index, then propagate the updated
            # container value so later reads see the new element.
            base = self.eval_expr(target.value, state)
            if isinstance(base, TensorVal):
                self._check_tensor_index_bounds(target, base, state)
                return  # tensor metadata (rank/shape/dtype) is unchanged
            if isinstance(base, (ListVal, TupleVal)):
                self._check_seq_index_bounds(target, base, state)
            updated = self._updated_container(base, target.slice, value, state)
            if updated is not None:
                self._rebind_lvalue(target.value, updated, state)
            return
        # Unmodeled target form: ignore.

    def _updated_container(self, base, slc, value, state: State):
        """Return a new container reflecting ``base[idx] = value``, or ``None`` if
        ``base`` is not a modeled mutable container.  Soundly weakens to a summary
        when the index/key is not a known constant."""
        if isinstance(base, ListVal):
            idx = self._index_const(slc, state)
            if (
                idx is not None
                and base.exact_elems is not None
                and -len(base.exact_elems) <= idx < len(base.exact_elems)
            ):
                elems = list(base.exact_elems)
                elems[idx] = value
                return ListVal(elem=join_many(elems), length=base.length, exact_elems=tuple(elems))
            # unknown index (or no exact elements): any slot may now hold value
            return ListVal(elem=join_many([base.elem, value]), length=base.length, exact_elems=None)
        if isinstance(base, DictVal):
            key = _const_str(slc)
            if key is not None:
                known = [(k, v) for k, v in base.known if k != key]
                known.append((key, value))
                return DictVal(
                    value=join_many([base.value, value]),
                    known=tuple(known),
                    exact_keys=base.exact_keys,
                )
            # unknown key: weaken to a summary (an existing key may be overwritten)
            return DictVal(value=join_many([base.value, value]), known=(), exact_keys=False)
        return None

    def _rebind_lvalue(self, node: ast.expr, value: AbstractValue, state: State) -> None:
        """Write ``value`` back to a simple lvalue (``name`` or ``obj.attr``) so a
        mutated container is visible to later statements.  Deeper/unknown lvalue
        forms are left unchanged (the container was already weakened)."""
        if isinstance(node, ast.Name):
            state.set(node.id, value)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            state.set_attr(node.value.id, node.attr, value)

    def _assign_unpack(
        self, target, value: AbstractValue, state: State, ctx: ast.stmt, rhs: Optional[ast.expr] = None
    ) -> None:
        has_star = any(isinstance(e, ast.Starred) for e in target.elts)
        n = len(target.elts)

        arity = _unpack_arity(value)
        reported = False
        if arity is not None and not has_star and arity != n:
            self._report_unpack(target, value, n, arity, ctx)
            reported = True

        # Structural return-arity contract: the right-hand side is a call to a
        # resolved function that returns a single (non-tuple) value on every
        # reachable path, yet we unpack into n>=2 targets.  This catches the
        # titans-pytorch #60 class even when the returned value abstracts to TOP.
        if (
            not reported
            and not has_star
            and n >= 2
            and isinstance(rhs, ast.Call)
            and self._call_single.get(id(rhs)) is True
        ):
            self._report_return_arity(target, n, ctx, value)

        # Bind element values when we can, else TOP.
        elems = _value_elems(value, n)
        for tgt, ev in zip(target.elts, elems):
            inner = tgt.value if isinstance(tgt, ast.Starred) else tgt
            self._assign_to(inner, ev, state, ctx)

    def _report_return_arity(self, target, n: int, ctx: ast.stmt, value) -> None:
        self._emit(
            SymBug(
                kind=SymBugKind.RETURN_ARITY_CONTRACT,
                message=(
                    f"unpacking into {n} targets, but the called function returns a "
                    f"single value on every path — expected {n} values, got 1"
                ),
                line=getattr(ctx, "lineno", 0),
                col=getattr(ctx, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.8,
                fix_suggestion=(
                    "make the function return the expected number of values (e.g. pass "
                    "the flag that enables the extra return), or unpack a single value"
                ),
                evidence="; ".join(value.provenance) if getattr(value, "provenance", None) else None,
            )
        )

    def _report_unpack(self, target, value, n, arity, ctx) -> None:
        if isinstance(value, NoneVal):
            kind, what = SymBugKind.NONE_PROPAGATION, "None"
        elif isinstance(value, TensorVal) and value.rank == 0:
            kind, what = SymBugKind.UNPACK_ARITY_MISMATCH, "a 0-d tensor"
        elif isinstance(value, (TensorVal,)):
            kind, what = SymBugKind.RETURN_ARITY_CONTRACT, "a single tensor (not a tuple)"
        elif isinstance(value, TupleVal):
            kind, what = SymBugKind.UNPACK_ARITY_MISMATCH, f"a {arity}-tuple"
        else:
            kind, what = SymBugKind.UNPACK_ARITY_MISMATCH, "a non-iterable value"
        self._emit(
            SymBug(
                kind=kind,
                message=(
                    f"cannot unpack {what} into {n} target"
                    f"{'s' if n != 1 else ''} — expected {n} values, the right-hand side "
                    f"yields {arity if arity is not None else 'a different number of'} value"
                    f"{'s' if arity != 1 else ''}"
                ),
                line=getattr(ctx, "lineno", 0),
                col=getattr(ctx, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9 if kind != SymBugKind.RETURN_ARITY_CONTRACT else 0.8,
                fix_suggestion=(
                    "return (or pass) the expected number of values, or unpack the "
                    "matching arity"
                ),
                evidence="; ".join(value.provenance) if value.provenance else None,
            )
        )

    # -- expression evaluation ------------------------------------------
    def eval_expr(self, node: ast.expr, state: State) -> AbstractValue:
        m = getattr(self, f"_ex_{type(node).__name__}", None)
        if m is None:
            # The AST modeler has no handler for this construct (e.g. a ``Set``
            # literal or an f-string).  Step 81: before abstaining to ``Top``,
            # try the sound bytecode constant-folder, which can recover a precise
            # value for straight-line pure expressions over known constants.
            folded = self._bytecode_fallback(node, state)
            return folded if folded is not None else TOP
        return m(node, state)

    def _bytecode_fallback(self, node: ast.expr, state: State):
        """Recover a concrete value for an unmodeled expression, or ``None``.

        Builds a ``names`` environment from the abstract state by projecting
        each free identifier the expression reads to its known concrete constant
        (if any), then folds the expression with the side-effect-free bytecode
        machine.  Returns an :class:`AbstractValue` on success, else ``None``.
        Never touches the abstain ledger, so abstain coverage — and therefore
        proof fingerprints — are unaffected when the fold does not apply.
        """
        from . import bytecode as _bc

        names = {}
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                concrete = _bc.abstract_to_concrete(state.get(child.id))
                if concrete is not _bc.NOT_CONCRETE:
                    names[child.id] = concrete
        return _bc.fold_to_abstract(node, names)

    def _ex_Constant(self, node: ast.Constant, state: State) -> AbstractValue:
        v = node.value
        if v is None:
            return NONE.with_prov(self._prov_label(node, "None literal"))
        if isinstance(v, bool):
            return BoolVal(const=v)
        if isinstance(v, int):
            return int_const(v)
        if isinstance(v, float):
            return FloatVal(const=v)
        if isinstance(v, str):
            return StrVal(const=v)
        return TOP

    def _ex_Name(self, node: ast.Name, state: State) -> AbstractValue:
        return state.get(node.id)

    def _ex_Attribute(self, node: ast.Attribute, state: State) -> AbstractValue:
        base = self.eval_expr(node.value, state)
        # self.<attr> / obj.<attr> via store
        if isinstance(node.value, ast.Name):
            stored = state.store.get(node.value.id, {})
            if node.attr in stored:
                return stored[node.attr]
        if isinstance(base, ModuleVal):
            got = base.get_attr(node.attr)
            if got is not None:
                return got
        if isinstance(base, TensorVal):
            if node.attr == "shape":
                if base.rank is not None:
                    elems = tuple(IntVal(sym=base.dim(i)) for i in range(base.rank))
                    return TupleVal(elems=elems, exact_len=True)
                return TupleVal(elems=(), exact_len=False)
            if node.attr in ("ndim",):
                return IntVal(sym=SymDim.const_dim(base.rank) if base.rank is not None else None)
            if node.attr in ("T", "mT", "data"):
                if node.attr == "data" and self.config.enable_heuristics:
                    self._check_tensor_data_access(node)
                return self._derive(
                    TensorVal(rank=base.rank, dtype=base.dtype, device=base.device),
                    node, f".{node.attr}", base,
                )
        if isinstance(base, NoneVal):
            self._report_none_deref(node, node.attr, base)
        return TOP

    def _ex_Tuple(self, node: ast.Tuple, state: State) -> AbstractValue:
        if any(isinstance(e, ast.Starred) for e in node.elts):
            return TupleVal(elems=(), exact_len=False)
        return TupleVal(elems=tuple(self.eval_expr(e, state) for e in node.elts), exact_len=True)

    def _ex_List(self, node: ast.List, state: State) -> AbstractValue:
        elems = tuple(self.eval_expr(e, state) for e in node.elts)
        return ListVal(elem=join_many(list(elems)) if elems else TOP, length=len(elems), exact_elems=elems)

    def _ex_Dict(self, node: ast.Dict, state: State) -> AbstractValue:
        known = []
        exact = True
        for k, v in zip(node.keys, node.values):
            if k is None:  # ``**spread`` — unknown extra keys
                exact = False
                continue
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                known.append((k.value, self.eval_expr(v, state)))
            else:
                exact = False
        value = join_many([v for _, v in known]) if known else TOP
        return DictVal(value=value, known=tuple(known), exact_keys=exact)

    # -- comprehensions & generators (precise element abstraction) ----------
    def _comp_child(self, generators: List[ast.comprehension], state: State) -> Optional[State]:
        """Build a child state in which every comprehension target is bound to the
        element abstraction of its iterable and each ``if`` clause is assumed true.
        The iterables (and filter guards) are evaluated for their bug side-effects.
        Returns ``None`` if a guard makes the body provably unreachable."""
        child = state.copy()
        for gen in generators:
            iter_val = self.eval_expr(gen.iter, child)
            self._assign_to(gen.target, _element_of(iter_val), child, gen.target)
            for guard in gen.ifs:
                self.eval_expr(guard, child)  # surface bugs inside the filter
                self._refine(guard, True, child)
                if not child.reachable:
                    return None
        return child

    def _ex_ListComp(self, node: ast.ListComp, state: State) -> AbstractValue:
        # Step 13 — precise enumeration during module construction: a
        # comprehension over a statically-enumerable iterable (``range(const)`` /
        # literal / a list-or-tuple-valued name) yields the EXACT element list, so
        # ``nn.ModuleList([Block(d) for _ in range(N)])`` resolves to N registered
        # children — the same contract as the explicit ``append`` loop (step 12).
        if self._constructing:
            states = self._comp_enumerate(node.generators, state)
            if states is not None:
                elems = [self.eval_expr(node.elt, s) for s in states]
                return ListVal(
                    elem=join_many(elems) if elems else TOP,
                    length=len(elems),
                    exact_elems=tuple(elems),
                )
        child = self._comp_child(node.generators, state)
        if child is None:
            return ListVal(elem=TOP, length=0, exact_elems=())
        elem = self.eval_expr(node.elt, child)
        return ListVal(elem=elem, length=None, exact_elems=None)

    def _ex_SetComp(self, node: ast.SetComp, state: State) -> AbstractValue:
        if self._constructing:
            states = self._comp_enumerate(node.generators, state)
            if states is not None:
                elems = [self.eval_expr(node.elt, s) for s in states]
                return SetVal(
                    elem=join_many(elems) if elems else TOP, length=len(elems)
                )
        child = self._comp_child(node.generators, state)
        if child is None:
            return SetVal(elem=TOP, length=0)
        return SetVal(elem=self.eval_expr(node.elt, child), length=None)

    def _ex_GeneratorExp(self, node: ast.GeneratorExp, state: State) -> AbstractValue:
        # A generator is lazy, but for element-iteration purposes a list is a
        # sound over-approximation of the values it yields; when constructing we
        # enumerate it precisely so ``nn.ModuleList(Block(d) for _ in range(N))``
        # (generator argument form) resolves like its list-comprehension twin.
        if self._constructing:
            states = self._comp_enumerate(node.generators, state)
            if states is not None:
                elems = [self.eval_expr(node.elt, s) for s in states]
                return ListVal(
                    elem=join_many(elems) if elems else TOP,
                    length=len(elems),
                    exact_elems=tuple(elems),
                )
        child = self._comp_child(node.generators, state)
        if child is None:
            return ListVal(elem=TOP, length=0, exact_elems=())
        return ListVal(elem=self.eval_expr(node.elt, child), length=None, exact_elems=None)

    def _ex_DictComp(self, node: ast.DictComp, state: State) -> AbstractValue:
        # Step 13 — ``{k: M() for k in keys}`` resolves to an exact key→value map
        # when the iterable is statically enumerable AND every produced key is a
        # constant string, so ``nn.ModuleDict({...comprehension...})`` names its
        # children faithfully.  Any non-constant key falls back to the summary.
        if self._constructing:
            states = self._comp_enumerate(node.generators, state)
            if states is not None:
                known: Dict[str, AbstractValue] = {}
                enumerable = True
                for s in states:
                    kv = self.eval_expr(node.key, s)
                    val = self.eval_expr(node.value, s)
                    if isinstance(kv, StrVal) and kv.const is not None:
                        known[kv.const] = val  # later key wins, mirroring dict
                    else:
                        enumerable = False
                        break
                if enumerable:
                    items = tuple(known.items())
                    return DictVal(
                        value=join_many([v for _, v in items]) if items else TOP,
                        known=items,
                        exact_keys=True,
                    )
        child = self._comp_child(node.generators, state)
        if child is None:
            return DictVal(value=TOP, known=(), exact_keys=False)
        self.eval_expr(node.key, child)  # surface bugs in the key expression
        val = self.eval_expr(node.value, child)
        return DictVal(value=val, known=(), exact_keys=False)

    def _enumerate_iter(self, iter_node: ast.expr, state: State) -> Optional[List[AbstractValue]]:
        """The explicit element list of a statically-enumerable iterable for a
        comprehension generator: a ``range(const)`` / literal sequence (via
        :meth:`_static_for_values`) or a name/expression that evaluates to an
        exact-element ``ListVal``/``TupleVal``.  ``None`` otherwise (abstain)."""
        vals = self._static_for_values(iter_node, state)
        if vals is not None:
            return vals
        return self._enumerable_seq(self.eval_expr(iter_node, state))

    def _comp_enumerate(
        self, generators: List[ast.comprehension], state: State
    ) -> Optional[List[State]]:
        """All concrete child states of a fully statically-enumerable
        comprehension — the Cartesian product of every generator's enumerable
        iterable, keeping only bindings whose ``if`` guards are *statically* true.

        Returns ``None`` (→ caller uses the sound length-unknown summary) when any
        generator's iterable is not enumerable, a guard is not statically
        decidable, the comprehension is ``async``, or the product would exceed the
        unroll cap.  Never guesses: an undecidable comprehension stays a summary."""
        states: List[State] = [state.copy()]
        for gen in generators:
            if getattr(gen, "is_async", 0):
                return None
            nxt: List[State] = []
            for s in states:
                elems = self._enumerate_iter(gen.iter, s)
                if elems is None:
                    return None
                for v in elems:
                    child = s.copy()
                    self._assign_to(gen.target, v, child, gen.target)
                    keep = True
                    for guard in gen.ifs:
                        g = self.eval_expr(guard, child)
                        truth = _known_truth(g)
                        if truth is None:
                            return None  # filter not statically decidable
                        if not truth:
                            keep = False
                            break
                        self._refine(guard, True, child)
                    if keep and child.reachable:
                        nxt.append(child)
            states = nxt
            if len(states) > _MAX_CONSTRUCT_UNROLL:
                return None
        return states

    def _ex_UnaryOp(self, node: ast.UnaryOp, state: State) -> AbstractValue:
        v = self.eval_expr(node.operand, state)
        if isinstance(node.op, ast.Not):
            self._check_bool_context(node, v)
            t = _known_truth(v)
            return BoolVal(const=(not t) if t is not None else None)
        if isinstance(node.op, ast.USub) and isinstance(v, IntVal) and v.sym is not None:
            return IntVal(sym=v.sym * -1)
        return v if isinstance(v, (IntVal, FloatVal)) else TOP

    def _ex_BinOp(self, node: ast.BinOp, state: State) -> AbstractValue:
        a = self.eval_expr(node.left, state)
        b = self.eval_expr(node.right, state)
        if isinstance(b, IntVal):
            self._check_div_by_zero(node, b)
        if isinstance(a, IntVal) and isinstance(b, IntVal):
            sym = None
            if a.sym is not None and b.sym is not None:
                try:
                    if isinstance(node.op, ast.Add):
                        sym = a.sym + b.sym
                    elif isinstance(node.op, ast.Sub):
                        sym = a.sym - b.sym
                    elif isinstance(node.op, ast.Mult):
                        sym = a.sym * b.sym
                    elif isinstance(node.op, ast.FloorDiv):
                        sym = a.sym.floordiv(b.sym)
                    elif isinstance(node.op, ast.Mod):
                        sym = a.sym.mod(b.sym)
                except Exception:
                    sym = None
            iv = _interval_binop(node.op, a.interval, b.interval)
            return IntVal(sym=sym, interval=iv)
        if isinstance(a, TensorVal) or isinstance(b, TensorVal):
            if isinstance(node.op, ast.MatMult):
                out = self._check_matmul(node, a, b)
                return out if out is not None else TOP
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                self._check_broadcast(node, a, b)
            ra = a.rank if isinstance(a, TensorVal) else None
            rb = b.rank if isinstance(b, TensorVal) else None
            rank = ra if ra is not None else rb
            if ra is not None and rb is not None:
                rank = max(ra, rb)
            return TensorVal(rank=rank)
        return TOP

    def _check_device(self, node, a: "TensorVal", b: "TensorVal", where: str) -> None:
        """Flag a binary op on two tensors whose *device types* are known and
        different (e.g. ``cpu`` vs ``cuda``).  This is a forced ``RuntimeError``
        at runtime ("Expected all tensors to be on the same device").  Sound: we
        only fire when both device types are pinned and genuinely differ — same
        type with different ordinals (``cuda`` vs ``cuda:0``) is normalised away,
        and any unknown device abstains."""
        da, db = a.device, b.device
        if da is None or db is None:
            return
        if da == db:
            return
        self._emit(
            SymBug(
                kind=SymBugKind.DEVICE_MISMATCH,
                message=(
                    f"{where} mixes tensors on different devices: "
                    f"{da} vs {db}; RuntimeError at runtime"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion=(
                    f"move both operands to the same device "
                    f"(e.g. .to('{da}') / .to('{db}'))"
                ),
            )
        )

    def _check_broadcast(self, node, a: AbstractValue, b: AbstractValue) -> None:
        """Flag two tensors whose trailing dims cannot broadcast.

        Concrete dims (both ``> 1`` and unequal) are a forced failure.  For
        *symbolic* dims the broadcast rule is discharged through the Z3 bridge
        (Step 53 — theory reuse): a dim pair ``(da, db)`` is broadcastable iff
        ``da == db ∨ da == 1 ∨ db == 1`` is satisfiable under the current path
        constraints; when **all three** disjuncts are infeasible the mismatch is
        forced for every valid concretization."""
        if not (isinstance(a, TensorVal) and isinstance(b, TensorVal)):
            return
        self._check_device(node, a, b, "elementwise op")
        sa, sb = a.shape, b.shape
        if sa is None or sb is None:
            return self._abstain(
                AbstainCategory.UNKNOWN_SHAPE, "broadcast", node,
                "operand shape unknown",
            )
        facts = list(self._cur_dim_facts)
        for da, db in zip(reversed(sa), reversed(sb)):
            va = da.value if da is not None else None
            vb = db.value if db is not None else None
            if va is not None and vb is not None:
                if va == 1 or vb == 1:
                    continue  # broadcastable
                if va != vb:
                    self._report_broadcast(node, sa, sb, va, vb)
                    return
                continue
            # at least one symbolic dim.  Step 86 ``heuristic`` mode surfaces a
            # clearly-labelled, low-confidence *suspicion* at the sites where
            # ``balanced``/``sound`` would soundly abstain (below) — never in
            # place of a Z3-proven report (the ``compat`` check still runs first
            # for symbolic-with-constraints dims).  ``known`` is the concrete
            # value when exactly one side is concrete.
            known = va if vb is None else vb
            heuristic_suspect = (
                self.config.enable_heuristics
                and isinstance(known, int)
                and known > 1
            )
            if da is None or db is None:
                # wholly-unknown dim: no constraints can ever force it.
                if heuristic_suspect:
                    self._report_broadcast_suspected(node, sa, sb, known)
                    return
                self._abstain(
                    AbstainCategory.UNKNOWN_DIM, "broadcast", node,
                    "wholly-unknown trailing dim",
                )
                continue
            if not facts:
                if heuristic_suspect:
                    self._report_broadcast_suspected(node, sa, sb, known)
                    return
                self._abstain(
                    AbstainCategory.NO_PATH_CONSTRAINTS, "broadcast", node,
                    "symbolic dim with no path constraints to force it",
                )
                continue  # no constraints to force the symbolic dim
            compat = (
                smt_bridge.feasible([*facts, smt_bridge.eq(da, db)])
                or smt_bridge.feasible([*facts, smt_bridge.eq(da, 1)])
                or smt_bridge.feasible([*facts, smt_bridge.eq(db, 1)])
            )
            if not compat:
                evidence = self._broadcast_counterexample(sa, sb, da, db)
                evidence = self._with_minimal_trace(
                    evidence, facts, lambda s: self._broadcast_forced(s, da, db)
                )
                self._report_broadcast(node, sa, sb, da, db, evidence=evidence)
                return

    @staticmethod
    def _broadcast_forced(facts, da, db) -> bool:
        """Whether the dim pair ``(da, db)`` still cannot broadcast under
        ``facts`` — i.e. equality and both ``==1`` escapes are all infeasible."""
        return (
            not smt_bridge.feasible([*facts, smt_bridge.eq(da, db)])
            and not smt_bridge.feasible([*facts, smt_bridge.eq(da, 1)])
            and not smt_bridge.feasible([*facts, smt_bridge.eq(db, 1)])
        )

    def _with_minimal_trace(self, evidence, facts, holds) -> Optional[str]:
        """Append the 1-minimal slice of ``facts`` that still forces the failure
        (Step 58 — trace minimization) to a report's evidence string.

        ``holds(subset)`` must report whether the fault is still forced under a
        subset of the path facts.  The minimal slice names the exact conditions
        the owner must change; an empty slice means the fault is unconditional."""
        if not facts or not smt_bridge.Z3_AVAILABLE:
            return evidence
        minimal = trace_min.minimize(list(facts), holds)
        if minimal:
            shown = " ∧ ".join(_constraint_str(c) for c in minimal)
            note = f"minimal failing conditions: {shown}"
        else:
            note = "minimal failing conditions: none (fault is unconditional)"
        return note if not evidence else f"{evidence}; {note}"

    def _report_broadcast(self, node, sa, sb, va, vb, evidence=None) -> None:
        self._emit(
            SymBug(
                kind=SymBugKind.BROADCAST_MISMATCH,
                message=(
                    f"tensors of shapes {_shape_str(sa)} and {_shape_str(sb)} "
                    f"cannot broadcast ({va} vs {vb}); RuntimeError at runtime"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion="align the operand shapes so each dim is equal or 1",
                evidence=evidence,
            )
        )

    def _report_broadcast_suspected(self, node, sa, sb, known: int) -> None:
        """Heuristic-mode (Step 86) low-confidence broadcast suspicion: a known
        concrete dim ``> 1`` aligned with a wholly-unknown dim.  Emitted only
        when ``config.enable_heuristics``; clearly labelled as possibly a false
        positive and ranked well below the engine's proven findings."""
        self._emit(
            SymBug(
                kind=SymBugKind.BROADCAST_MISMATCH,
                message=(
                    f"tensors of shapes {_shape_str(sa)} and {_shape_str(sb)} "
                    f"may fail to broadcast: a concrete dim {known} is aligned "
                    f"with an unknown dim (suspected; heuristic mode)"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.5,
                fix_suggestion=(
                    "confirm the unknown dim equals "
                    f"{known} or 1 to broadcast safely"
                ),
                evidence="heuristic suspicion (may be a false positive)",
            )
        )

    def _check_matmul(self, node, a: AbstractValue, b: AbstractValue):
        """Model ``a @ b`` / ``torch.matmul(a, b)``.  Flag a contracted-dimension
        mismatch (``a.shape[-1] != b.shape[-2]``) when both contracting dims are
        statically known, and return the result tensor with the correct rank and
        (where determinable) output shape.  Abstains on any symbolic contracting
        dim or unknown rank; never reports on those."""
        if not (isinstance(a, TensorVal) and isinstance(b, TensorVal)):
            return None
        self._check_device(node, a, b, "matmul")
        ra, rb = a.rank, b.rank
        if ra is None or rb is None or ra == 0 or rb == 0:
            if not (isinstance(a, TensorVal) and isinstance(b, TensorVal) and ra == 0 and rb == 0):
                self._abstain(
                    AbstainCategory.UNKNOWN_RANK, "matmul", node,
                    "operand rank unknown",
                )
            return None
        # contracting dims: last of a, and (2nd-last of b if rb>=2 else only dim)
        a_last = a.dim(ra - 1) if a.shape is not None else None
        if rb == 1:
            b_contract = b.dim(0) if b.shape is not None else None
        else:
            b_contract = b.dim(rb - 2) if b.shape is not None else None
        va = a_last.value if a_last is not None else None
        vb = b_contract.value if b_contract is not None else None
        if va is not None and vb is not None and va != vb:
            sa = _shape_str(a.shape) if a.shape is not None else f"rank-{ra}"
            sb = _shape_str(b.shape) if b.shape is not None else f"rank-{rb}"
            cert = self._certify_matmul(a, b, ra, rb, rb == 1)
            evidence = self._witness2(a, b)
            if cert is not None:
                evidence = f"{evidence} | {cert}" if evidence else cert
            self._emit(
                SymBug(
                    kind=SymBugKind.MATMUL_DIM_MISMATCH,
                    message=(
                        f"matmul contracted-dim mismatch: {sa} @ {sb} "
                        f"({va} vs {vb}); RuntimeError at runtime"
                    ),
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    function=self._cur_func_name(),
                    confidence=0.95 if cert is not None else 0.9,
                    fix_suggestion=f"align the inner dimensions ({va} vs {vb})",
                    evidence=evidence,
                )
            )
        elif va is None or vb is None:
            self._abstain(
                AbstainCategory.UNKNOWN_DIM, "matmul", node,
                "symbolic contracting dim",
            )
        # output rank per torch.matmul semantics
        if ra == 1 and rb == 1:
            out_rank = 0
        elif ra == 1:
            out_rank = rb - 1
        elif rb == 1:
            out_rank = ra - 1
        else:
            out_rank = max(ra, rb)
        out_shape = None
        if ra >= 2 and rb >= 2 and ra == rb and a.shape is not None and b.shape is not None:
            batch = list(a.shape[:-2])
            m = a.dim(ra - 2)
            n = b.dim(rb - 1)
            out_shape = tuple(batch + [m, n])
        out = TensorVal(rank=0, shape=()) if out_rank == 0 else TensorVal(rank=out_rank, shape=out_shape)
        return self._derive(out, node, "matmul", a, b)

    def _seq_tensors(self, val):
        """Return the list of element TensorVals if ``val`` is a statically-known
        list/tuple whose every element is a TensorVal; else ``None`` (abstain)."""
        if isinstance(val, ListVal) and val.exact_elems is not None:
            elems = val.exact_elems
        elif isinstance(val, TupleVal) and val.exact_len:
            elems = val.elems
        else:
            return None
        if not elems or any(not isinstance(e, TensorVal) for e in elems):
            return None
        return list(elems)

    def _check_cat_stack(self, node, op: str, pos, kw):
        """Model ``torch.cat``/``concat``/``stack`` (+ ``hstack``/``vstack``).

        ``cat`` requires every input to share all dims except the concat axis;
        ``stack`` requires *identical* shapes.  Fire only when a concrete pair of
        dims that must be equal are statically known and unequal.  Abstains on
        any symbolic dim, mixed/unknown rank, or a non-literal sequence."""
        ts = self._seq_tensors(pos[0])
        if ts is None or len(ts) < 2:
            return self._abstain(
                AbstainCategory.NON_LITERAL_PATTERN, "cat_stack", node,
                f"torch.{op} inputs not a statically-known tensor sequence",
            )
        ranks = {t.rank for t in ts}
        if None in ranks or len(ranks) != 1:
            return self._abstain(
                AbstainCategory.UNKNOWN_RANK, "cat_stack", node,
                f"torch.{op} inputs have unknown / differing rank",
            )  # unknown or differing rank: caller handles rank elsewhere
        rank = next(iter(ranks))
        if rank == 0:
            return None
        is_stack = op in ("stack",)
        # resolve concat axis for the cat family (stack compares every axis)
        cat_axis = None
        if not is_stack:
            dim_v = kw.get("dim")
            if dim_v is None and len(pos) >= 2:
                dim_v = pos[1]
            if op == "hstack":
                cat_axis = 0 if rank == 1 else 1
            elif op == "vstack":
                cat_axis = 0
            elif isinstance(dim_v, IntVal) and dim_v.const is not None:
                cat_axis = dim_v.const % rank if rank else 0
            else:
                cat_axis = 0 if dim_v is None else None  # symbolic dim → abstain on axis
        for axis in range(rank):
            if not is_stack and axis == cat_axis:
                continue
            vals = []
            for t in ts:
                d = t.dim(axis) if t.shape is not None else None
                vals.append(d.value if d is not None else None)
            known = [v for v in vals if v is not None]
            if len(known) >= 2 and len(set(known)) > 1:
                self._emit(
                    SymBug(
                        kind=SymBugKind.CAT_SHAPE_MISMATCH,
                        message=(
                            f"torch.{op} inputs disagree on dim {axis} "
                            f"({sorted(set(known))}); RuntimeError at runtime"
                        ),
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        function=self._cur_func_name(),
                        confidence=0.9,
                        fix_suggestion=f"make all inputs share dim {axis}",
                    )
                )
                return TensorVal(rank=rank + 1 if is_stack else rank)
        return TensorVal(rank=rank + 1 if is_stack else rank)

    def _check_einsum(self, node, eq_val, operands, has_starred: bool):
        """Model ``torch.einsum(equation, *operands)``.

        Two forced-failure checks (sound — fire only on statically-known
        conflicts): a per-operand subscript/rank arity mismatch, and a
        repeated-index dimension mismatch (the same letter bound to two different
        concrete sizes across operands).  Always returns the result tensor with
        the einsum output rank/shape (letters resolved to known sizes), or a
        rank-unknown tensor when the equation is non-literal or uses ``...``."""
        if not isinstance(eq_val, StrVal) or eq_val.const is None:
            self._abstain(
                AbstainCategory.NON_LITERAL_PATTERN, "einsum", node,
                "einsum equation is non-literal",
            )
            return TensorVal(rank=None)
        parsed = _parse_einsum_eq(eq_val.const)
        if parsed is None:  # ellipsis / malformed → abstain on precision, stay sound
            self._abstain(
                AbstainCategory.ELLIPSIS_PATTERN, "einsum", node,
                "einsum equation uses ellipsis / is malformed",
            )
            return TensorVal(rank=None)
        subs, out_sub = parsed
        tensors = [o if isinstance(o, TensorVal) else None for o in operands]
        aligned = (not has_starred) and len(subs) == len(operands)

        # -- per-operand subscript/rank arity --------------------------------
        if aligned:
            for s, t in zip(subs, tensors):
                if t is not None and t.rank is not None and len(s) != t.rank:
                    self._emit(
                        SymBug(
                            kind=SymBugKind.EINSUM_DIM_MISMATCH,
                            message=(
                                f"einsum subscript '{s}' has {len(s)} indices but "
                                f"operand is rank-{t.rank}; RuntimeError at runtime"
                            ),
                            line=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            function=self._cur_func_name(),
                            confidence=0.9,
                            fix_suggestion=f"give subscript '{s}' exactly {t.rank} indices",
                        )
                    )
                    break

        # -- repeated-index dimension consistency + collect known sizes ------
        dim_map: Dict[str, int] = {}
        if aligned:
            conflict = None
            for s, t in zip(subs, tensors):
                if t is None or t.shape is None or t.rank != len(s):
                    continue
                for p, letter in enumerate(s):
                    d = t.dim(p)
                    v = d.value if d is not None else None
                    if v is None:
                        continue
                    if letter in dim_map and dim_map[letter] != v:
                        conflict = (letter, dim_map[letter], v)
                        break
                    dim_map[letter] = v
                if conflict is not None:
                    break
            if conflict is not None:
                letter, v0, v1 = conflict
                self._emit(
                    SymBug(
                        kind=SymBugKind.EINSUM_DIM_MISMATCH,
                        message=(
                            f"einsum index '{letter}' bound to mismatched sizes "
                            f"{v0} vs {v1}; RuntimeError at runtime"
                        ),
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        function=self._cur_func_name(),
                        confidence=0.95,
                        fix_suggestion=f"make index '{letter}' the same size in all operands",
                    )
                )

        # -- output tensor ---------------------------------------------------
        out_letters = out_sub if out_sub is not None else _einsum_implicit_out(subs)
        out_shape = tuple(
            SymDim.const_dim(dim_map[ch]) if ch in dim_map else None for ch in out_letters
        )
        result = TensorVal(rank=len(out_letters), shape=out_shape if out_letters else ())
        known = [o for o in operands if isinstance(o, TensorVal)]
        return self._derive(result, node, "einsum", *known)

    def _check_einops(self, node, op: str, recv, pattern_val, kw):
        """Model ``einops.rearrange/reduce/repeat(tensor, 'lhs -> rhs', **axes)``.

        Sound forced-failure checks (fire only on statically-certain conflicts):

        * **rank mismatch** — the input tensor rank disagrees with the number of
          top-level groups on the LHS pattern;
        * **duplicate axis** — the same axis name twice on one side (always an
          ``EinopsError``);
        * **undefined output axis** — an axis name on the RHS that is absent from
          the LHS and (for ``repeat``) not supplied as a keyword size; for
          ``rearrange``/``reduce`` new axes are never allowed;
        * **decomposition mismatch** — a composed group ``(a b)`` whose fully
          known factor sizes don't divide / don't equal the known input dim.

        Returns the result tensor with inferred rank/shape, or a rank-unknown
        tensor when the pattern is non-literal or uses an ellipsis (abstain)."""
        if not isinstance(recv, TensorVal):
            return None
        if not isinstance(pattern_val, StrVal) or pattern_val.const is None:
            self._abstain(
                AbstainCategory.NON_LITERAL_PATTERN, "einops_rearrange", node,
                "rearrange pattern is non-literal",
            )
            return TensorVal(rank=None)
        parsed = _parse_einops_pattern(pattern_val.const)
        if parsed is None:  # no arrow / ellipsis / unparseable → abstain
            self._abstain(
                AbstainCategory.ELLIPSIS_PATTERN, "einops_rearrange", node,
                "rearrange pattern uses ellipsis / is unparseable",
            )
            return TensorVal(rank=None)
        lhs_groups, rhs_groups = parsed
        kw_sizes: Dict[str, int] = {
            k: v.const for k, v in kw.items() if isinstance(v, IntVal) and v.const is not None
        }
        lhs_names = _einops_names(lhs_groups)
        rhs_names = _einops_names(rhs_groups)

        def _report(msg: str, fix: str, conf: float = 0.9):
            self._emit(
                SymBug(
                    kind=SymBugKind.EINOPS_PATTERN_MISMATCH,
                    message=msg,
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    function=self._cur_func_name(),
                    confidence=conf,
                    fix_suggestion=fix,
                )
            )

        # -- input rank vs LHS group count -----------------------------------
        if recv.rank is not None and len(lhs_groups) != recv.rank:
            _report(
                f"einops.{op} pattern LHS has {len(lhs_groups)} axes but the input "
                f"tensor is rank-{recv.rank}; EinopsError at runtime",
                f"give the LHS exactly {recv.rank} top-level axes",
                conf=0.92,
            )

        # -- duplicate axis names on a side ----------------------------------
        for side, names in (("LHS", lhs_names), ("RHS", rhs_names)):
            seen = set()
            for nm in names:
                if nm in seen:
                    _report(
                        f"einops.{op} repeats axis '{nm}' on the {side}; EinopsError at runtime",
                        f"use each axis name at most once per side",
                        conf=0.95,
                    )
                    break
                seen.add(nm)

        # -- undefined output axes -------------------------------------------
        lhs_set = set(lhs_names)
        for nm in rhs_names:
            if nm in lhs_set:
                continue
            if op == "repeat" and nm in kw_sizes:
                continue  # a genuinely new repeated axis with an explicit size
            extra = "" if op != "repeat" else " (and was given no size)"
            _report(
                f"einops.{op} output axis '{nm}' is not present in the input pattern"
                f"{extra}; EinopsError at runtime",
                f"introduce '{nm}' on the LHS"
                + ("" if op != "repeat" else f" or pass {nm}=<size>"),
                conf=0.9,
            )

        # -- resolve axis sizes + decomposition divisibility -----------------
        sizes: Dict[str, int] = dict(kw_sizes)
        aligned = recv.rank is not None and len(lhs_groups) == recv.rank and recv.shape is not None
        if aligned:
            for i, g in enumerate(lhs_groups):
                d = recv.dim(i)
                dim = d.value if d is not None else None
                names = [m for m in g if isinstance(m, str)]
                lits = [m for m in g if isinstance(m, int)]
                lit_prod = 1
                for m in lits:
                    lit_prod *= m
                if len(g) == 1 and names:
                    if dim is not None:
                        sizes[names[0]] = dim
                    continue
                # composed group: check divisibility / consistency when known
                if dim is None:
                    continue
                known_prod = lit_prod
                unknown = []
                for nm in names:
                    if nm in sizes:
                        known_prod *= sizes[nm]
                    else:
                        unknown.append(nm)
                if known_prod == 0:
                    continue
                if not unknown:
                    if known_prod != dim:
                        _report(
                            f"einops.{op} group {tuple(g)} multiplies to {known_prod} "
                            f"but axis {i} of the input is {dim}; EinopsError at runtime",
                            f"make the factors of axis {i} multiply to {dim}",
                            conf=0.93,
                        )
                elif len(unknown) == 1:
                    if dim % known_prod != 0:
                        _report(
                            f"einops.{op} axis {i} of size {dim} is not divisible by the "
                            f"known factors {known_prod} of group {tuple(g)}; EinopsError at runtime",
                            f"choose factor sizes that divide {dim}",
                            conf=0.93,
                        )
                    else:
                        sizes[unknown[0]] = dim // known_prod

        # -- output shape inference ------------------------------------------
        out_shape = []
        for g in rhs_groups:
            total = 1
            ok = True
            for m in g:
                if isinstance(m, int):
                    total *= m
                elif m in sizes:
                    total *= sizes[m]
                else:
                    ok = False
                    break
            if not g:  # empty group () → singleton axis
                total = 1
                ok = True
            out_shape.append(SymDim.const_dim(total) if ok else None)
        result = TensorVal(
            rank=len(rhs_groups), shape=tuple(out_shape) if rhs_groups else ()
        )
        return self._derive(result, node, f"einops.{op}", recv)

    # axis-range checking for dim-taking tensor methods --------------------
    # method -> tuple of positional indices that name an axis (besides 'dim'/'axis' kw)
    _AXIS_POS = {
        "unsqueeze": (0,),
        "squeeze": (0,),
        "select": (0,),
        "transpose": (0, 1),
        "swapaxes": (0, 1),
        "swapdims": (0, 1),
        "softmax": (0,),
        "log_softmax": (0,),
        "cumsum": (0,),
        "cumprod": (0,),
        "flip": (),
        "index_select": (0,),
        "narrow": (0,),
        "unbind": (0,),
        "chunk": (1,),
        "movedim": (0, 1),
        "moveaxis": (0, 1),
    }
    # methods whose every listed arg is an axis (variadic permute-style)
    _AXIS_VARIADIC = {"permute"}

    def _check_axis(self, node, recv: TensorVal, method: str, pos, kw) -> None:
        """Flag a constant ``dim``/``axis`` argument that is out of range for the
        receiver's known rank.  ``unsqueeze`` allows ``[-r-1, r]``; every other
        modeled method allows ``[-r, r-1]``.  Abstains on unknown rank or a
        non-constant axis (no false positives)."""
        r = recv.rank
        if r is None:
            return
        if method == "unsqueeze":
            lo, hi = -r - 1, r
        else:
            lo, hi = -r, r - 1

        def flag(v):
            self._emit(
                SymBug(
                    kind=SymBugKind.AXIS_OUT_OF_RANGE,
                    message=(
                        f".{method}() got dim {v} but the tensor has rank {r} "
                        f"(valid range [{lo}, {hi}]); IndexError at runtime"
                    ),
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    function=self._cur_func_name(),
                    confidence=0.9,
                    fix_suggestion=f"use a dim within [{lo}, {hi}]",
                )
            )

        def axis_const(val):
            return val.const if isinstance(val, IntVal) else None

        # variadic permute(*dims): dims may be passed as separate args or a tuple/list
        if method in self._AXIS_VARIADIC:
            dims = pos
            if len(pos) == 1 and isinstance(pos[0], (TupleVal, ListVal)):
                seq = pos[0]
                dims = list(seq.elems) if isinstance(seq, TupleVal) else (seq.exact_elems or ())
            for d in dims:
                c = axis_const(d)
                if c is not None and not (lo <= c <= hi):
                    flag(c)
                    return
            return

        # 'dim'/'axis' keyword
        kw_axis = kw.get("dim", kw.get("axis"))
        if kw_axis is not None:
            c = axis_const(kw_axis)
            if c is not None and not (lo <= c <= hi):
                flag(c)
                return

        for idx in self._AXIS_POS.get(method, ()):
            if idx < len(pos):
                c = axis_const(pos[idx])
                if c is not None and not (lo <= c <= hi):
                    flag(c)
                    return
        # reductions also take a leading positional dim
        if method in (
            "sum", "mean", "max", "min", "prod", "std", "var", "argmax",
            "argmin", "median", "norm", "logsumexp", "amax", "amin",
            "nanmean", "nansum", "any", "all",
        ) and pos:
            c = axis_const(pos[0])
            if c is not None and not (lo <= c <= hi):
                flag(c)

    def _check_repeat(self, node, recv: TensorVal, pos, kw) -> None:
        """Flag ``tensor.repeat(*sizes)`` given *fewer* size args than the
        receiver's rank — torch raises ``RuntimeError`` ("Number of dimensions of
        repeat dims can not be smaller than number of dimensions of tensor").

        ``repeat`` accepts the sizes either as separate positional ints
        (``x.repeat(2, 3)``) or as a single tuple/list (``x.repeat((2, 3))``).
        We only fire when the *number* of provided dims is statically known and
        strictly less than a known rank; anything else abstains (no false
        positives)."""
        r = recv.rank
        if r is None:
            return
        # A single sequence argument carries the dims; otherwise each positional
        # is one dim.  ``repeat`` takes no keyword dims.
        if len(pos) == 1 and isinstance(pos[0], (TupleVal, ListVal)):
            seq = pos[0]
            if isinstance(seq, TupleVal):
                ndims = len(seq.elems)
            else:
                elems = seq.exact_elems
                if elems is None:
                    return  # unknown-length list: abstain
                ndims = len(elems)
        elif pos:
            ndims = len(pos)
        else:
            return  # no sizes given: abstain (degenerate call)
        if ndims < r:
            self._emit(
                SymBug(
                    kind=SymBugKind.REPEAT_DIMS_TOO_FEW,
                    message=(
                        f".repeat() got {ndims} repeat dim(s) but the tensor has "
                        f"rank {r}; repeat dims must be >= the tensor rank, "
                        f"RuntimeError at runtime"
                    ),
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    function=self._cur_func_name(),
                    confidence=0.9,
                    fix_suggestion=(
                        f"pass at least {r} repeat dims (one per tensor dimension)"
                    ),
                )
            )

    def _check_expand(self, node, recv: TensorVal, pos, kw) -> None:
        """Flag a ``tensor.expand(*sizes)`` whose target shape provably cannot be
        produced.  Three sound, forced ``RuntimeError`` cases (torch-verified):

        * **too few sizes** — fewer sizes than the receiver's rank;
        * **non-singleton mismatch** — an aligned existing dim is a *known*
          constant ``!= 1`` while its target is a *known* size that is neither
          ``-1`` (keep) nor equal to it;
        * **leading ``-1``** — a *new* leading dimension given the placeholder
          ``-1`` (only allowed for an existing dim).

        Sizes may be separate ints (``x.expand(2, 3)``) or one tuple/list
        (``x.expand((2, 3))``).  Abstains on unknown rank, an unknown-length size
        sequence, or any size/dim that is not a known constant (no false
        positives)."""
        r = recv.rank
        if len(pos) == 1 and isinstance(pos[0], (TupleVal, ListVal)):
            seq = pos[0]
            if isinstance(seq, TupleVal):
                sizes = list(seq.elems)
            else:
                elems = seq.exact_elems
                if elems is None:
                    return  # unknown-length list: abstain
                sizes = list(elems)
        elif pos:
            sizes = list(pos)
        else:
            return
        ndims = len(sizes)

        def emit(msg, fix):
            self._emit(
                SymBug(
                    kind=SymBugKind.EXPAND_SHAPE_MISMATCH,
                    message=msg,
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    function=self._cur_func_name(),
                    confidence=0.9,
                    fix_suggestion=fix,
                )
            )

        if r is not None and ndims < r:
            emit(
                f".expand() got {ndims} size(s) but the tensor has rank {r}; "
                f"expand requires at least as many sizes as the rank, "
                f"RuntimeError at runtime",
                f"pass at least {r} sizes (one per tensor dimension)",
            )
            return
        if r is None:
            return

        lead = ndims - r  # leading positions create *new* dimensions
        shape = recv.shape

        def size_const(v):
            return v.const if isinstance(v, IntVal) else None

        for i, sz in enumerate(sizes):
            t = size_const(sz)
            if t is None:
                continue
            if i < lead:
                if t == -1:
                    emit(
                        f".expand() uses -1 for a new leading dimension "
                        f"(position {i}); -1 is only allowed for an existing "
                        f"dimension, RuntimeError at runtime",
                        "give a concrete non-negative size for new leading dims",
                    )
                    return
                continue
            if shape is None:
                continue
            di = i - lead
            dim = shape[di] if di < len(shape) else None
            d = getattr(dim, "value", None) if dim is not None else None
            if d is None:
                continue
            if d != 1 and t != -1 and t != d:
                emit(
                    f".expand() target size {t} at dim {di} must match the "
                    f"existing non-singleton size {d} (or be -1); "
                    f"RuntimeError at runtime",
                    f"use {d} or -1 for this dimension",
                )
                return

    def _check_div_by_zero(self, node: ast.BinOp, divisor: IntVal) -> None:
        """Report when ``/``, ``//`` or ``%`` is applied to a divisor whose
        every concretization is zero.  Only fires when the divisor is *provably*
        the constant 0 (no false positives on merely-possible zeros)."""
        if not isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            return
        if divisor.contains_only_zero():
            self._emit(
                SymBug(
                    kind=SymBugKind.DIVISION_BY_ZERO,
                    message="division or modulo by zero — the divisor is always 0",
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    function=self._cur_func_name(),
                    confidence=0.95,
                    fix_suggestion="ensure the divisor is non-zero before dividing",
                )
            )

    def _ex_BoolOp(self, node: ast.BoolOp, state: State) -> AbstractValue:
        vals = [self.eval_expr(v, state) for v in node.values]
        return join_many(vals)

    def _ex_Compare(self, node: ast.Compare, state: State) -> AbstractValue:
        left = self.eval_expr(node.left, state)
        if len(node.ops) == 1 and len(node.comparators) == 1:
            right = self.eval_expr(node.comparators[0], state)
            op = node.ops[0]
            if isinstance(op, (ast.Is, ast.Eq)) and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None:
                if isinstance(left, NoneVal):
                    return BoolVal(const=True)
            if isinstance(op, (ast.IsNot, ast.NotEq)) and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None:
                if isinstance(left, NoneVal):
                    return BoolVal(const=False)
            if isinstance(op, ast.Eq) and isinstance(left, IntVal) and isinstance(right, IntVal):
                if left.sym is not None and right.sym is not None:
                    eq = left.sym.maybe_eq(right.sym)
                    if eq is not None:
                        return BoolVal(const=eq)
        return BoolVal(const=None)

    def _ex_IfExp(self, node: ast.IfExp, state: State) -> AbstractValue:
        t = _known_truth(self.eval_expr(node.test, state))
        if t is True:
            return self.eval_expr(node.body, state)
        if t is False:
            return self.eval_expr(node.orelse, state)
        return join_many([self.eval_expr(node.body, state), self.eval_expr(node.orelse, state)])

    def _ex_Subscript(self, node: ast.Subscript, state: State) -> AbstractValue:
        base = self.eval_expr(node.value, state)
        self._check_subscript_rank(node, base, state)
        if isinstance(base, TensorVal):
            self._check_tensor_index_bounds(node, base, state)
            return self._subscript_tensor(node, base)
        if isinstance(base, TupleVal) and base.exact_len:
            idx = self._index_const(node.slice, state)
            if idx is not None:
                if -len(base.elems) <= idx < len(base.elems):
                    return base.elems[idx]
                self._report_index_oob(node, idx, len(base.elems), "tuple")
                return TOP
        if isinstance(base, ListVal):
            idx = self._index_const(node.slice, state)
            if idx is not None and base.length is not None:
                if not (-base.length <= idx < base.length):
                    self._report_index_oob(node, idx, base.length, "list")
                    return TOP
                if base.exact_elems is not None:
                    return base.exact_elems[idx]
            return base.elem
        if isinstance(base, DictVal):
            key = _const_str(node.slice)
            if key is not None:
                got = base.get_key(key)
                if got is not None:
                    return got
                if base.exact_keys:
                    self._report_missing_key(node, key)
                    return TOP
            return base.value
        if isinstance(base, NoneVal):
            self._report_none_deref(node, "[...]", base)
        return TOP

    def _index_const(self, slc, state: State) -> Optional[int]:
        """A constant integer index: syntactic first (handles negative literals),
        then via abstract evaluation (so ``xs[len(xs)]`` / ``xs[len(xs) - 1]``
        resolve)."""
        idx = _const_index(slc)
        if idx is not None:
            return idx
        node = slc.value if isinstance(slc, ast.Index) else slc
        if isinstance(node, ast.Slice):
            return None
        try:
            v = self.eval_expr(node, state)
        except Exception:
            return None
        return v.const if isinstance(v, IntVal) else None

    # -- provenance / replayable witnesses (Step 7) ---------------------
    def _prov_label(self, node, msg: str) -> str:
        """A single line-tagged provenance step, e.g. ``"L3: torch.zeros(2, 3)"``."""
        ln = getattr(node, "lineno", None)
        return f"L{ln}: {msg}" if ln else msg

    def _derive(self, out: AbstractValue, node, label: str, *inputs: AbstractValue) -> AbstractValue:
        """Attach a replayable derivation to ``out``: the accumulated provenance of
        every ``input`` followed by this op's line-tagged ``label``.  Consecutive
        duplicates are collapsed so a chain reads source→…→sink without noise."""
        chain: List[str] = []
        for v in inputs:
            for step in getattr(v, "provenance", ()) or ():
                if not chain or chain[-1] != step:
                    chain.append(step)
        step = self._prov_label(node, label)
        if not chain or chain[-1] != step:
            chain.append(step)
        return out.with_prov(*chain)

    @staticmethod
    def _witness(value: Optional[AbstractValue]) -> Optional[str]:
        """Render a value's provenance chain as a human-readable witness string."""
        prov = getattr(value, "provenance", None) if value is not None else None
        return " → ".join(prov) if prov else None

    @staticmethod
    def _witness2(*values: AbstractValue) -> Optional[str]:
        """Render the combined provenance of several operands (de-duplicated, in
        order) as a single witness string."""
        chain: List[str] = []
        for v in values:
            for step in getattr(v, "provenance", ()) or ():
                if step not in chain:
                    chain.append(step)
        return " → ".join(chain) if chain else None

    @staticmethod
    def _certify_matmul(a, b, ra: int, rb: int, b_is_vec: bool) -> Optional[str]:
        """Use the concretization oracle to certify a matmul mismatch: find a
        concrete pair of shapes in γ(a)×γ(b) on which the contracted dims really
        disagree, proving the report is not spurious."""
        from . import concretize as _C

        ca, cb = _C.gamma(a), _C.gamma(b)
        if not isinstance(ca, _C.ConcreteTensor) or not isinstance(cb, _C.ConcreteTensor):
            return None
        if ca.rank < 1 or cb.rank < 1:
            return None
        a_inner = ca.shape[-1]
        b_inner = cb.shape[0] if b_is_vec else cb.shape[-2]
        if a_inner == b_inner:
            return None
        return (
            f"certified counterexample: a.shape={tuple(ca.shape)} @ "
            f"b.shape={tuple(cb.shape)} (inner {a_inner} ≠ {b_inner})"
        )

    def _report_none_deref(self, node, what: str, value: Optional[AbstractValue] = None) -> None:
        self._emit(
            SymBug(
                kind=SymBugKind.NONE_PROPAGATION,
                message=(
                    f"value is None here; accessing {what!r} raises "
                    f"AttributeError/TypeError at runtime"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.85,
                fix_suggestion="guard with `if x is not None:` or assign a non-None value first",
                evidence=self._witness(value),
            )
        )

    def _report_index_oob(self, node, idx: int, length: int, kind: str) -> None:
        self._emit(
            SymBug(
                kind=SymBugKind.RANK_INDEX_ERROR,
                message=(
                    f"{kind} index {idx} is out of range for a {kind} of length {length}"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.93,
                fix_suggestion=f"use an index in [-{length}, {length - 1}]",
            )
        )

    def _check_reshape(self, node, recv: "TensorVal", pos: List[AbstractValue]) -> None:
        """Element-count preservation for ``.view``/``.reshape``.

        Fires only when *both* the receiver's element count and every requested
        size are statically known (so the failure is forced).  Handles a single
        ``-1`` inference slot and a single tuple/list size argument.  When the
        element count is *not* concretely known (symbolic dims), defer to the
        Z3-backed reshape theory (Step 53) which can still prove incompatibility
        for some symbolic shapes (e.g. reshaping ``[a, b]`` to ``[a, b, 2]``).
        """
        numel = _known_numel(recv)
        if numel is None:
            self._check_reshape_symbolic(node, recv, pos)
            return
        targets = _reshape_target_sizes(pos)
        if targets is None:
            return
        neg_ones = [t for t in targets if t == -1]
        if len(neg_ones) > 1:
            return  # invalid call shape; let runtime handle, don't guess
        concrete = [t for t in targets if t != -1]
        if any(t < 0 for t in concrete):
            return
        prod = 1
        for t in concrete:
            prod *= t
        if neg_ones:
            if prod == 0:
                return
            if numel % prod != 0:
                self._report_reshape(node, numel, targets)
            return
        if prod != numel:
            self._report_reshape(node, numel, targets)

    # ------------------------------------------------------------------
    # Step 54 — counterexample lifting.
    #
    # When a symbolic detector proves a forced failure under the current
    # path constraints, ``smt_bridge.model`` is asked for a concrete
    # satisfying assignment of the dimension variables.  Substituting that
    # assignment back into the operand shapes yields a *concrete entry
    # shape* that actually triggers the reported error — a replayable
    # witness an owner can paste into a repro.
    # ------------------------------------------------------------------

    def _lift_model(self, syms, extra=()):
        """A concrete ``{dim-var-name: int}`` assignment over the dimension
        variables appearing in ``syms`` (each pinned ``>= 1``) under the
        current path constraints, or ``None`` when z3 is unavailable / the
        path is infeasible.  ``extra`` adds further constraints."""
        names = set()
        for sd in syms:
            if sd is not None:
                for n, _ in getattr(sd, "terms", ()):  # SymDim.terms
                    names.add(n)
        bounds = [smt_bridge.ge(SymDim.var(n), 1) for n in sorted(names)]
        facts = list(self._cur_dim_facts)
        try:
            return smt_bridge.model([*facts, *bounds, *extra])
        except Exception:  # pragma: no cover - defensive: never crash analysis
            return None

    @staticmethod
    def _concretize_symdim(sym: Optional[SymDim], model: dict):
        """Substitute a ``{name: int}`` model into an affine ``SymDim`` to get a
        concrete ``int`` (or ``None`` if a referenced variable is absent)."""
        if sym is None or model is None:
            return None
        total = int(sym.const)
        for name, coeff in sym.terms:
            if name not in model:
                return None
            total += int(coeff) * int(model[name])
        return total

    def _concretize_shape(self, shape, model):
        """Render a tuple of ``SymDim`` under a model as a concrete shape tuple of
        ``int`` (or ``None`` if any dim cannot be concretized)."""
        if shape is None or model is None:
            return None
        out = []
        for d in shape:
            v = self._concretize_symdim(d, model)
            if v is None:
                return None
            out.append(v)
        return tuple(out)

    @staticmethod
    def _shape_tuple_str(dims) -> str:
        return "(" + ", ".join(str(d) for d in dims) + ")"

    def _reshape_counterexample(self, recv_shape, target) -> Optional[str]:
        """Lift a concrete input shape + element counts that witness a forced
        reshape size mismatch.  Returns an evidence string, or ``None``."""
        target_syms = [SymDim.var(t) for t in target if isinstance(t, str)]
        model = self._lift_model(list(recv_shape) + target_syms)
        if not model:
            return None
        in_dims = self._concretize_shape(recv_shape, model)
        if in_dims is None:
            return None
        in_numel = 1
        for v in in_dims:
            in_numel *= v
        tgt_concrete = []
        has_infer = False
        for t in target:
            if isinstance(t, int):
                if t == -1:
                    has_infer = True
                tgt_concrete.append(t)
            elif isinstance(t, str):
                if t not in model:
                    return None
                tgt_concrete.append(int(model[t]))
            else:  # pragma: no cover - defensive
                return None
        in_str = self._shape_tuple_str(in_dims)
        tgt_str = self._shape_tuple_str(tgt_concrete)
        if has_infer:
            known = 1
            for t in tgt_concrete:
                if t != -1:
                    known *= t
            return (
                f"concrete counterexample: input shape {in_str} has {in_numel} "
                f"elements, which is not divisible by {known} (product of the "
                f"fixed target dims), so reshape {tgt_str} cannot infer -1"
            )
        tgt_numel = 1
        for t in tgt_concrete:
            tgt_numel *= t
        return (
            f"concrete counterexample: input shape {in_str} has {in_numel} "
            f"elements but reshape target {tgt_str} needs {tgt_numel}"
        )

    def _broadcast_counterexample(self, sa, sb, da, db) -> Optional[str]:
        """Lift concrete operand shapes that witness a forced broadcast mismatch.
        Returns an evidence string, or ``None``."""
        model = self._lift_model(list(sa) + list(sb))
        if not model:
            return None
        ca = self._concretize_shape(sa, model)
        cb = self._concretize_shape(sb, model)
        cda = self._concretize_symdim(da, model)
        cdb = self._concretize_symdim(db, model)
        if cda is None or cdb is None:
            return None
        pair = f" (dims {cda} vs {cdb} are unequal and neither is 1)"
        if ca is not None and cb is not None:
            return (
                f"concrete counterexample: shapes {self._shape_tuple_str(ca)} and "
                f"{self._shape_tuple_str(cb)} cannot broadcast{pair}"
            )
        return f"concrete counterexample:{pair}"

    def _check_reshape_symbolic(self, node, recv: "TensorVal", pos: List[AbstractValue]) -> None:
        """Symbolic-dimension reshape feasibility via the existing Z3 reshape
        theory (Step 53 — theory reuse).  Both the receiver shape and the target
        sizes are lowered into the theory's ``TensorShape``/dim-entry form; a
        non-``None`` result means the element count cannot be preserved for *any*
        positive concretization, so the reshape is a forced runtime error."""
        if recv.shape is None or recv.rank is None:
            return self._abstain(
                AbstainCategory.UNKNOWN_SHAPE, "reshape", node,
                "receiver shape/rank unknown",
            )
        in_entries = []
        for d in recv.shape:
            e = _symdim_to_shape_entry(d)
            if e is None:
                return self._abstain(
                    AbstainCategory.UNREPRESENTABLE_AFFINE, "reshape", node,
                    "input dim not representable as one theory dim",
                )  # an input dim we can't represent: abstain (sound)
            in_entries.append(e)
        target = _reshape_target_entries(pos)
        if target is None:
            return self._abstain(
                AbstainCategory.UNKNOWN_TARGET, "reshape", node,
                "reshape target not statically known",
            )
        # Only engage when something is actually symbolic; the all-concrete case
        # is handled precisely (and faster) by the analytic path above.
        if all(isinstance(e, int) for e in in_entries) and all(
            isinstance(t, int) for t in target
        ):
            return
        try:
            from src.smt.reshape_theory import check_reshape_compatible
            from src.tensor_shapes import ShapeDim, TensorShape
        except Exception:  # pragma: no cover - theory/z3 unavailable: abstain
            return self._abstain(
                AbstainCategory.THEORY_UNAVAILABLE, "reshape", node,
                "reshape theory backend unavailable",
            )
        input_shape = TensorShape(tuple(ShapeDim(e) for e in in_entries))
        try:
            msg = check_reshape_compatible(input_shape, tuple(target))
        except Exception:  # pragma: no cover - defensive: never crash analysis
            return
        if msg is not None:
            self._report_reshape_symbolic(node, input_shape, target, recv.shape)

    def _report_reshape_symbolic(self, node, input_shape, target, recv_shape=None) -> None:
        shown = ", ".join(str(t) for t in target)
        evidence = None
        if recv_shape is not None:
            evidence = self._reshape_counterexample(recv_shape, target)
        self._emit(
            SymBug(
                kind=SymBugKind.RESHAPE_SIZE_MISMATCH,
                message=(
                    f"reshape target ({shown}) cannot preserve the element count of "
                    f"a tensor with shape {input_shape.pretty()} for any positive "
                    f"dimension sizes (RuntimeError at runtime)"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion="make the product of the new sizes equal the element count, or use -1",
                evidence=evidence,
            )
        )

    def _check_inplace_leaf(self, node, recv, method: str) -> None:
        """Flag an in-place op (``add_``/``mul_``/…) applied directly to a tensor
        that is a *leaf* requiring grad — a forced ``RuntimeError`` ("a leaf
        Variable that requires grad is being used in an in-place operation").
        Sound: fires only when both ``requires_grad`` and ``is_leaf`` are
        positively known to be ``True``; abstains otherwise.  ``requires_grad_``
        and ``detach_`` are permitted in-place ops and are never flagged."""
        if not isinstance(recv, TensorVal):
            return
        if method not in _INPLACE_AUTOGRAD_OPS:
            return
        if recv.requires_grad is not True or recv.is_leaf is not True:
            return
        self._emit(
            SymBug(
                kind=SymBugKind.INPLACE_ON_LEAF,
                message=(
                    f".{method}() is an in-place op on a leaf tensor that requires "
                    f"grad; RuntimeError at runtime"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion=(
                    "use the out-of-place form (e.g. x = x + ...), or operate "
                    "under torch.no_grad(), or call .detach() first"
                ),
            )
        )

    def _check_bool_context(self, node, val) -> None:
        """Flag a tensor used in a boolean context (``if t:``/``while t:``/
        ``not t``/``bool(t)``) when its element count is statically known and not
        exactly 1 — a forced ``RuntimeError`` ("Boolean value of Tensor with more
        than one value is ambiguous").  Sound: abstains on any unknown dim."""
        if not isinstance(val, TensorVal):
            return
        numel = _known_numel(val)
        if numel is None or numel == 1:
            return
        self._emit(
            SymBug(
                kind=SymBugKind.BOOL_ON_NONSCALAR,
                message=(
                    f"using a tensor with {numel} elements in a boolean context is "
                    f"ambiguous; RuntimeError at runtime"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion=(
                    "reduce to a single bool first, e.g. (t > 0).any() / .all() / "
                    "t.item()"
                ),
            )
        )

    def _check_item(self, node, recv) -> None:
        """Flag ``tensor.item()`` on a tensor whose element count is statically
        known and not exactly 1 — a forced ``RuntimeError`` ("only one element
        tensors can be converted to Python scalars").  Sound: abstains whenever
        the element count is not fully known (any symbolic/unknown dim)."""
        if not isinstance(recv, TensorVal):
            return
        numel = _known_numel(recv)
        if numel is None or numel == 1:
            return
        shape = recv.shape
        shape_str = (
            "(" + ", ".join(str(d.value) for d in shape) + ")"
            if shape is not None else "?"
        )
        self._emit(
            SymBug(
                kind=SymBugKind.ITEM_ON_NONSCALAR,
                message=(
                    f".item() requires exactly one element but the tensor has "
                    f"{numel} (shape {shape_str}); RuntimeError at runtime"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion=(
                    "index/reduce to a single element first, or use .tolist()/"
                    ".detach().cpu().numpy() for multi-element tensors"
                ),
            )
        )

    def _check_backward(self, node, recv, method: str, pos, kw) -> None:
        """Flag ``tensor.backward()`` on a non-scalar tensor with no ``gradient``
        argument — a forced ``RuntimeError`` ("grad can be implicitly created only
        for scalar outputs").  Sound: fires only when ``requires_grad`` is
        positively known to be ``True`` (otherwise a *different* "does not require
        grad" error masks this one), the element count is statically known and not
        1, and no ``gradient`` argument was supplied.  Abstains otherwise."""
        if method != "backward" or not isinstance(recv, TensorVal):
            return
        if recv.requires_grad is not True:
            return
        if pos or "gradient" in kw:
            return  # an explicit gradient was supplied — no error
        numel = _known_numel(recv)
        if numel is None or numel == 1:
            return
        self._emit(
            SymBug(
                kind=SymBugKind.BACKWARD_ON_NONSCALAR,
                message=(
                    f".backward() on a non-scalar tensor with {numel} elements and "
                    f"no gradient= argument; grad can be implicitly created only for "
                    f"scalar outputs (RuntimeError at runtime)"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion=(
                    "reduce to a scalar first (e.g. loss.mean()/.sum()), or pass an "
                    "explicit gradient=torch.ones_like(t)"
                ),
            )
        )

    def _check_backward_no_grad(self, node, recv, method: str, pos, kw) -> None:
        """Heuristic intent check: ``tensor.backward()`` on a tensor that is
        *positively known* not to require grad — almost always because the value
        was ``.detach()``-ed (or ``requires_grad_(False)``-ed) before backprop.
        At runtime PyTorch raises ``RuntimeError: element 0 of tensors does not
        require grad and does not have a grad_fn``; statically it is a silent
        training-killer (no gradients flow).  Heuristic-only and suppressed in
        ``sound``/``balanced``: it relies on positive ``requires_grad is False``
        provenance (e.g. from ``detach()``), never on the *absence* of grad info,
        so it does not fire on tensors whose grad status is merely unknown."""
        if method != "backward" or not isinstance(recv, TensorVal):
            return
        if not self.config.enable_heuristics:
            return
        if recv.requires_grad is not False:
            return  # only positive non-grad provenance (e.g. after .detach())
        self._emit(
            SymBug(
                kind=SymBugKind.BACKWARD_NO_GRAD,
                message=(
                    ".backward() on a tensor that does not require grad (e.g. after "
                    ".detach()/requires_grad_(False)); no gradients flow and PyTorch "
                    "raises 'does not require grad and does not have a grad_fn'"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                severity="warning",
                confidence=0.7,
                fix_suggestion=(
                    "call .backward() on the grad-tracking loss itself; do not "
                    ".detach() the value you backpropagate through"
                ),
            )
        )

    def _check_copy_construct(self, node) -> None:
        """Heuristic (even_more #5b): ``torch.tensor(existing_tensor)`` copy-
        constructs from a tensor, which emits a UserWarning at runtime and is
        almost always unintended (it detaches/copies silently).  Heuristic-only:
        suppressed in ``sound``/``balanced``; ``severity="warning"``."""
        if not self.config.enable_heuristics:
            return
        self._emit(
            SymBug(
                kind=SymBugKind.TENSOR_COPY_CONSTRUCT,
                message=(
                    "torch.tensor(...) is copy-constructing from an existing tensor; "
                    "this copies/detaches silently (UserWarning at runtime)"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                severity="warning",
                confidence=0.85,
                fix_suggestion=(
                    "use sourceTensor.clone().detach() (optionally "
                    ".requires_grad_(True)) instead of torch.tensor(sourceTensor)"
                ),
            )
        )

    def _check_numpy(self, node, recv) -> None:
        """Flag ``tensor.numpy()`` on a tensor whose ``requires_grad`` is positively
        known to be ``True`` — a forced ``RuntimeError`` ("Can't call numpy() on
        Tensor that requires grad").  Sound: abstains unless ``requires_grad`` is
        known ``True``; ``.detach().numpy()`` is never flagged (detach clears the
        flag)."""
        if not isinstance(recv, TensorVal) or recv.requires_grad is not True:
            return
        self._emit(
            SymBug(
                kind=SymBugKind.NUMPY_ON_GRAD,
                message=(
                    ".numpy() on a tensor that requires grad; RuntimeError at "
                    "runtime (\"Can't call numpy() on Tensor that requires grad\")"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion="call .detach() first, e.g. tensor.detach().numpy()",
            )
        )

    def _check_requires_grad_dtype(self, node, dtype, requires_grad, where) -> None:
        """Flag a tensor constructor that sets ``requires_grad=True`` on a
        non-floating, non-complex dtype (integer/bool) — a forced ``RuntimeError``
        ("Only Tensors of floating point and complex dtype can require
        gradients").  Sound: fires only when the dtype is a *known* integer/bool
        type and ``requires_grad`` is known ``True``; abstains otherwise."""
        if requires_grad is not True or not isinstance(dtype, str):
            return
        if dtype not in _NON_DIFF_DTYPES:
            return
        self._emit(
            SymBug(
                kind=SymBugKind.REQUIRES_GRAD_NON_FLOAT,
                message=(
                    f"{where} sets requires_grad=True on a {dtype} (integer/bool) "
                    f"tensor; only floating-point and complex tensors can require "
                    f"grad (RuntimeError at runtime)"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion=(
                    "use a floating dtype (e.g. torch.float32), or drop "
                    "requires_grad=True for an integer/bool tensor"
                ),
            )
        )

    def _report_reshape(self, node, numel: int, targets) -> None:
        shown = ", ".join(str(t) for t in targets)
        self._emit(
            SymBug(
                kind=SymBugKind.RESHAPE_SIZE_MISMATCH,
                message=(
                    f"reshape target ({shown}) is incompatible with a tensor of "
                    f"{numel} elements (RuntimeError at runtime)"
                ),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion="make the product of the new sizes equal the element count, or use -1",
            )
        )


    def _report_missing_key(self, node, key: str) -> None:
        self._emit(
            SymBug(
                kind=SymBugKind.NONE_PROPAGATION,
                message=f"key {key!r} is not present in the dict (KeyError at runtime)",
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.8,
                fix_suggestion="add the key, or use dict.get with a default",
            )
        )

    def _ex_Call(self, node: ast.Call, state: State) -> AbstractValue:
        return self._eval_call(node, state)

    # -- subscript rank safety (OpenStrawberry #113 class) ---------------
    def _check_subscript_rank(self, node: ast.Subscript, base: AbstractValue, state: State) -> None:
        if not isinstance(base, TensorVal) or base.rank is None:
            return
        n_index_dims = _num_index_dims(node.slice)
        if n_index_dims is None:
            return
        if n_index_dims > base.rank:
            self._emit(
                SymBug(
                    kind=SymBugKind.RANK_INDEX_ERROR,
                    message=(
                        f"indexing a {base.rank}-D tensor with {n_index_dims} index "
                        f"dimensions — too many indices for tensor of dimension "
                        f"{base.rank}"
                    ),
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    function=self._cur_func_name(),
                    confidence=0.92,
                    fix_suggestion=(
                        "reduce the number of index dimensions, or ensure the tensor "
                        f"has at least {n_index_dims} dimensions before indexing"
                    ),
                    evidence="; ".join(base.provenance) if base.provenance else None,
                )
            )

    def _check_seq_index_bounds(self, node: ast.Subscript, base, state: State) -> None:
        """Bounds-check a constant index on the LHS of ``xs[i] = v`` for a
        known-length list/tuple (mirrors the read-side check)."""
        length = base.length if isinstance(base, (ListVal, TupleVal)) else None
        if isinstance(base, TupleVal):
            length = len(base.elems) if base.exact_len else None
        if length is None:
            return
        idx = self._index_const(node.slice, state)
        if idx is not None and not (-length <= idx < length):
            kind = "list" if isinstance(base, ListVal) else "tuple"
            self._report_index_oob(node, idx, length, kind)

    def _check_tensor_index_bounds(self, node: ast.Subscript, base: TensorVal, state: State) -> None:
        """Flag a constant integer index that is out of bounds for the matching
        dimension's known size.  Only the leading run of plain integer indices is
        checked (mapping index position k → dim k); a slice keeps its dim, a
        ``None`` insert is skipped, and an ellipsis stops the scan.  Abstains on
        symbolic dim sizes and non-constant indices."""
        if base.rank is None or base.shape is None:
            return
        slc = node.slice
        if isinstance(slc, ast.Index):  # py<3.9 compatibility
            slc = slc.value
        elts = list(slc.elts) if isinstance(slc, ast.Tuple) else [slc]
        dim = 0
        for el in elts:
            if isinstance(el, ast.Slice):
                dim += 1
                continue
            if isinstance(el, ast.Constant) and el.value is None:
                continue  # np.newaxis / None inserts a dim, consumes none
            if isinstance(el, ast.Starred) or (
                isinstance(el, ast.Constant) and el.value is Ellipsis
            ):
                return  # ellipsis: remaining dim mapping is ambiguous
            if dim >= base.rank:
                return
            try:
                v = self.eval_expr(el, state)
            except Exception:
                v = None
            c = v.const if isinstance(v, IntVal) else None
            if c is not None:
                d = base.dim(dim)
                s = d.value if d is not None else None
                if s is not None and not (-s <= c < s):
                    self._emit(
                        SymBug(
                            kind=SymBugKind.TENSOR_INDEX_OOB,
                            message=(
                                f"index {c} is out of bounds for dimension {dim} "
                                f"with size {s}; IndexError at runtime"
                            ),
                            line=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            function=self._cur_func_name(),
                            confidence=0.93,
                            fix_suggestion=f"use an index in [-{s}, {s - 1}] for dim {dim}",
                            evidence="; ".join(base.provenance) if base.provenance else None,
                        )
                    )
                    return
            dim += 1

    def _subscript_tensor(self, node: ast.Subscript, base: TensorVal) -> AbstractValue:
        # rank after basic indexing = rank - (#integer indices) + (#None inserts)
        consumed, inserted = _index_rank_delta(node.slice)
        if base.rank is None or consumed is None:
            return self._derive(
                TensorVal(rank=None, dtype=base.dtype, device=base.device), node, "index", base
            )
        new_rank = base.rank - consumed + inserted
        new_rank = max(new_rank, 0)
        return self._derive(
            TensorVal(rank=new_rank, dtype=base.dtype, device=base.device), node, "index", base
        )

    # -- call resolution / interprocedural ------------------------------
    def _eval_call(self, node: ast.Call, state: State) -> AbstractValue:
        func = node.func
        pos = [self.eval_expr(a, state) for a in node.args if not isinstance(a, ast.Starred)]
        kw = {k.arg: self.eval_expr(k.value, state) for k in node.keywords if k.arg}

        # torch tensor constructors with explicit sizes -> precise shape/rank
        ctor = self._tensor_ctor(func, node, state)
        if ctor is not None:
            return ctor

        # nn-layer constructors (nn.Linear(...) etc.) -> a ModuleVal carrying meta
        nn_ctor = self._nn_ctor(func, pos, kw)
        if nn_ctor is not None:
            return nn_ctor

        # collections.OrderedDict(...) -> an (insertion-ordered) DictVal so that
        # nn.Sequential / nn.ModuleDict can name their children faithfully.
        odict = self._ordered_dict_ctor(func, node, pos, kw, state)
        if odict is not None:
            return odict

        # nn.Sequential(...) -> a container ModuleVal whose children are checked
        # for adjacent feature-dim compatibility when applied.
        seq = self._seq_ctor(func, node, pos, state)
        if seq is not None:
            return seq

        # nn.ModuleList([...]) / nn.ModuleDict({...}) -> a container ModuleVal whose
        # registered children are index-/key-addressed attrs.
        container = self._module_container_ctor(func, node, pos, state)
        if container is not None:
            return container

        # torch free-function matmul family: torch.matmul/mm/bmm(a, b)
        chain = _attr_chain(func)
        if chain is not None and chain.split(".")[-1] in ("matmul", "mm", "bmm"):
            parts = chain.split(".")
            if "torch" in parts[:-1] and len(pos) >= 2:
                out = self._check_matmul(node, pos[0], pos[1])
                if out is not None:
                    return out

        # torch free-function concat/stack family
        if chain is not None and chain.split(".")[-1] in ("cat", "concat", "stack", "hstack", "vstack"):
            parts = chain.split(".")
            if "torch" in parts[:-1] and pos:
                op = chain.split(".")[-1]
                out = self._check_cat_stack(node, op, pos, kw)
                if out is not None:
                    return out

        # torch.einsum(equation, *operands)
        if chain is not None and chain.split(".")[-1] == "einsum":
            parts = chain.split(".")
            if "torch" in parts[:-1] and len(pos) >= 1:
                has_starred = any(isinstance(a, ast.Starred) for a in node.args)
                out = self._check_einsum(node, pos[0], pos[1:], has_starred)
                if out is not None:
                    return out

        # einops.rearrange/reduce/repeat(tensor, 'lhs -> rhs', **axes).  We only
        # engage when the 2nd positional is a literal pattern containing '->', so
        # an unrelated function named `reduce`/`repeat` (e.g. functools.reduce)
        # never trips the einops modeling.
        if chain is not None and chain.split(".")[-1] in ("rearrange", "reduce", "repeat"):
            op = chain.split(".")[-1]
            pat = pos[1] if len(pos) >= 2 else None
            if (
                isinstance(pat, StrVal)
                and pat.const is not None
                and "->" in pat.const
                and isinstance(pos[0], TensorVal)
            ):
                out = self._check_einops(node, op, pos[0], pat, kw)
                if out is not None:
                    return out

        # method call: recv.method(...)
        if isinstance(func, ast.Attribute):
            # ``super().<attr>(...)`` delegation to a base class (Step 16): run
            # the inherited method so base submodules register / its bugs surface.
            sup = self._eval_super_call(func, pos, kw, node, state)
            if sup is not None:
                return sup
            recv = self.eval_expr(func.value, state)
            if isinstance(recv, TensorVal):
                if func.attr in ("matmul", "mm", "bmm") and pos:
                    out = self._check_matmul(node, recv, pos[0])
                    if out is not None:
                        return out
                self._check_axis(node, recv, func.attr, pos, kw)
                if func.attr == "repeat":
                    self._check_repeat(node, recv, pos, kw)
                if func.attr == "expand":
                    self._check_expand(node, recv, pos, kw)
                if func.attr in ("view", "reshape"):
                    self._check_reshape(node, recv, pos)
                if func.attr == "item" and not pos:
                    self._check_item(node, recv)
                if func.attr == "backward":
                    self._check_backward(node, recv, func.attr, pos, kw)
                    self._check_backward_no_grad(node, recv, func.attr, pos, kw)
                if func.attr == "numpy" and not pos:
                    self._check_numpy(node, recv)
                self._check_inplace_leaf(node, recv, func.attr)
                out = tensor_method(recv, func.attr, pos, kw)
                if isinstance(out, TensorVal):
                    return self._derive(out, node, f".{func.attr}(...)", recv)
                return out
            if isinstance(recv, ModuleVal):
                # Step 12 — accumulate into a registered container during module
                # construction so a loop-built ``self.layers.append(Block(i))``
                # resolves to N distinct submodules.  Gated on ``_constructing``
                # so it only ever shapes the derived weights contract (forward
                # analysis is unaffected).
                if (
                    self._constructing
                    and func.attr in ("append", "extend", "insert")
                    and recv.class_name in ("ModuleList",)
                ):
                    self._container_mutate(func.value, recv, func.attr, pos, node, state)
                    return recv
                # higher-order ``module.apply(fn)`` / registered hooks (Step 50)
                if func.attr == "apply":
                    cb = self._resolve_callback(node.args[0]) if node.args else None
                    if cb is not None:
                        self._call_user_func(cb, [recv], {}, node, self_val=None)
                    return recv
                if func.attr.startswith("register_") and func.attr.endswith("_hook"):
                    return TOP
                if func.attr == "forward":
                    self._check_direct_forward(node, func)
                method = None
                cls = self.classes.get(recv.class_name)
                if cls is not None:
                    method = self._class_method(cls, func.attr)
                if method is not None:
                    return self._call_user_func(method, pos, kw, node, self_val=recv)
                # ``self.fc(x)`` where ``fc`` is a stored nn-layer attribute:
                # resolve the attribute to its value and apply it.
                callee = self.eval_expr(func, state)
                if isinstance(callee, ModuleVal):
                    applied = self._apply_nn_layer(callee, pos, node)
                    if applied is not None:
                        return applied
                applied = self._apply_nn_layer(recv, pos, node)
                if applied is not None:
                    return applied
            # nn-layer instance call producing a tensor: abstain to TOP unless modeled
            stub = self._stub_call(func, pos, kw, node)
            if stub is not None:
                return stub
            return TOP

        # plain name: free function or class instantiation or callable var
        if isinstance(func, ast.Name):
            # higher-order builtins whose callback we resolve from the AST so
            # bugs *inside* the callback body surface (Step 50).
            if func.id in ("map", "filter") and len(node.args) >= 2:
                ho = self._eval_map_filter(func.id, node, pos, state)
                if ho is not None:
                    return ho
            builtin = self._eval_builtin(func.id, pos)
            if builtin is not None:
                return builtin
            cls = self.classes.get(func.id)
            if cls is not None:
                return self._instantiate(cls, pos, kw, node)
            fn = self._lookup_func_by_name(func.id)
            if fn is not None:
                return self._call_user_func(fn, pos, kw, node, self_val=None)
            val = state.get(func.id)
            if isinstance(val, ModuleVal):
                # calling a module instance == calling its forward
                cls2 = self.classes.get(val.class_name)
                fwd = self._class_method(cls2, "forward") if cls2 else None
                if fwd is not None:
                    return self._call_user_func(fwd, pos, kw, node, self_val=val)
                applied = self._apply_nn_layer(val, pos, node)
                if applied is not None:
                    return applied
        stub = self._stub_call(func, pos, kw, node)
        if stub is not None:
            return stub
        return TOP

    def _call_user_func(self, fn, pos, kw, node, self_val) -> AbstractValue:
        bound: Dict[str, AbstractValue] = {}
        names = [a.arg for a in fn.args.args]
        if self_val is not None and names and names[0] == "self":
            names = names[1:]
        for name, val in zip(names, pos):
            bound[name] = val
        bound.update(kw)
        self._last_single = None
        result = self.run_function(fn, bound, self_val=self_val)
        if node is not None and self._last_single is not None:
            self._call_single[id(node)] = self._last_single
        return result

    def _instantiate(self, cls: ast.ClassDef, pos, kw, node) -> AbstractValue:
        meta = self._nn_layer_meta(cls.name, pos, kw)
        resolved_init = self._resolve_method(cls, "__init__")
        attrs: Tuple[Tuple[str, AbstractValue], ...] = ()
        mv = ModuleVal(class_name=cls.name, attrs=attrs, meta=tuple(meta.items()))
        if resolved_init is not None and len(self._frames) < _MAX_DEPTH:
            init, def_cls = resolved_init
            # run __init__ to populate self attrs
            st = State()
            st.set("self", mv)
            st.store["self"] = {}
            self._bind_params(init, dict(zip([a.arg for a in init.args.args[1:]], pos)), st, has_self=True)
            for name, val in kw.items():
                st.set(name, val)
            frame = Frame(func=init)
            self._frames.append(frame)
            # The class that *defines* the running __init__ anchors ``super()``
            # resolution (Step 16): an inherited __init__ resolves super relative
            # to its own defining class, not the instantiated subclass.
            self._class_stack.append(def_cls)
            self._constructing += 1
            try:
                final = self.exec_block(init.body, st)
            finally:
                self._constructing -= 1
                self._class_stack.pop()
                self._frames.pop()
            # Read ``self`` attrs from the *returned* state, not ``st``: control
            # flow (an unresolved ``if``, Step 15) merges branch copies into a
            # fresh joined state, so conditionally-registered submodules live
            # there.  Fall back to ``st`` only when the path terminated (e.g. an
            # unconditional early ``return`` leaves an unreachable merged state).
            src = final if final.reachable and "self" in final.store else st
            attrs = tuple(src.store.get("self", {}).items())
        return ModuleVal(class_name=cls.name, attrs=attrs, meta=tuple(meta.items()))

    # -- Step 16: super() / inheritance ----------------------------------
    def _super_target(self, attr: str) -> Optional[Tuple[ast.FunctionDef, ast.ClassDef]]:
        """Resolve ``super().<attr>`` from the current lexical class context: look
        up ``attr`` starting at the *bases* of the class defining the running
        method (so ``super().__init__`` finds the base ``__init__``, never the
        current one).  ``None`` when there is no known user base defining it."""
        if not self._class_stack:
            return None
        cur = self._class_stack[-1]
        for base in self._base_classdefs(cur):
            found = self._resolve_method(base, attr)
            if found is not None:
                return found
        return None

    def _run_inherited_init(
        self, init: ast.FunctionDef, def_cls: ast.ClassDef,
        self_val: AbstractValue, pos, kw, state: State,
    ) -> None:
        """Execute a base-class ``__init__`` so the submodules/params it registers
        accumulate into the *current* ``self`` store (the same dict the subclass
        __init__ is building).  Runs with ``def_cls`` pushed so a chained
        ``super().__init__()`` inside the base resolves one level further up."""
        if len(self._frames) >= _MAX_DEPTH:
            return
        child = State()
        child.set("self", self_val)
        # Seed with attrs registered so far so the base sees them, then merge the
        # base's registrations back after it runs.
        child.store["self"] = dict(state.store.get("self", {}))
        self._bind_params(
            init, dict(zip([a.arg for a in init.args.args[1:]], pos)),
            child, has_self=True,
        )
        for name, val in kw.items():
            child.set(name, val)
        self._frames.append(Frame(func=init))
        self._class_stack.append(def_cls)
        self._constructing += 1
        try:
            final = self.exec_block(init.body, child)
        finally:
            self._constructing -= 1
            self._class_stack.pop()
            self._frames.pop()
        src = final if final.reachable and "self" in final.store else child
        caller_self = state.store.setdefault("self", {})
        caller_self.update(src.store.get("self", {}))

    def _eval_super_call(
        self, func: ast.Attribute, pos, kw, node, state: State
    ) -> Optional[AbstractValue]:
        """Handle ``super().<attr>(...)`` (zero-arg ``super()`` form).  Returns the
        call's value, or ``None`` if this is not a ``super()`` delegation we model.

        * ``super().__init__(...)`` during construction runs the base ``__init__``
          so inherited submodules/params are registered into the contract.
        * Other ``super().<m>(...)`` delegate to the inherited method's body so
          bugs inside it still surface and its return value is propagated."""
        if not _is_super_call(func.value):
            return None
        target = self._super_target(func.attr)
        self_val = state.get("self")
        if func.attr == "__init__":
            if target is not None and isinstance(self_val, ModuleVal):
                init, def_cls = target
                self._run_inherited_init(init, def_cls, self_val, pos, kw, state)
            return NONE  # __init__ returns None; base may be nn.Module (no-op)
        if target is not None:
            method, _def_cls = target
            return self._call_user_func(method, pos, kw, node, self_val=self_val)
        return None

    def _eval_map_filter(self, name, node, pos, state) -> Optional[AbstractValue]:
        """Model ``map(fn, iterable)`` / ``filter(fn, iterable)``: resolve the
        callback (named function or lambda), apply it to the element abstraction
        of the iterable so any bug in the callback body surfaces, and return a
        ``ListVal`` summarising the result element.  Returns ``None`` to fall back
        to default handling if the callback can't be resolved."""
        if len(pos) < 2:
            return None
        elem = _element_of(pos[1])
        cb_node = node.args[0]
        result_elem: AbstractValue = TOP
        if isinstance(cb_node, ast.Lambda):
            params = [a.arg for a in cb_node.args.args]
            sub = State()
            sub.env = dict(state.env)
            if params:
                sub.set(params[0], elem)
            result_elem = self.eval_expr(cb_node.body, sub)
        else:
            fn = self._resolve_callback(cb_node)
            if fn is None:
                return None
            result_elem = self._call_user_func(fn, [elem], {}, node, self_val=None)
        if name == "filter":
            # filter keeps elements of the same type; summarise as the element.
            return ListVal(elem=elem, length=None)
        return ListVal(elem=result_elem, length=None)

    def _eval_builtin(self, name: str, pos: List[AbstractValue]) -> Optional[AbstractValue]:
        """Model a handful of pure builtins precisely.  Returns ``None`` when the
        name is not a modeled builtin (so normal resolution proceeds)."""
        if name == "len" and len(pos) == 1:
            return self._len_of(pos[0])
        if name == "abs" and len(pos) == 1 and isinstance(pos[0], IntVal):
            c = pos[0].const
            return int_const(abs(c)) if c is not None else IntVal()
        if name == "int":
            if pos and isinstance(pos[0], IntVal):
                return pos[0]
            return IntVal()
        if name == "float":
            return FloatVal()
        if name == "bool":
            return BoolVal()
        if name == "str":
            return StrVal()
        return None

    def _len_of(self, v: AbstractValue) -> AbstractValue:
        """``len(v)`` as a constant ``IntVal`` when statically known, else a
        non-negative unknown int."""
        if isinstance(v, TupleVal) and v.exact_len:
            return int_const(len(v.elems))
        if isinstance(v, ListVal) and v.length is not None:
            return int_const(v.length)
        if isinstance(v, SetVal) and v.length is not None:
            return int_const(v.length)
        if isinstance(v, DictVal) and v.exact_keys:
            return int_const(len(v.known))
        if isinstance(v, StrVal) and v.const is not None:
            return int_const(len(v.const))
        if isinstance(v, TensorVal) and v.rank is not None and v.rank >= 1:
            d0 = v.dim(0)
            if d0 is not None and d0.value is not None:
                return int_const(d0.value)
        # len is always a non-negative integer
        return int_range(0, None)

    def _tensor_ctor(self, func, node, state) -> Optional[AbstractValue]:
        name = _attr_chain(func)
        if name == "torch.tensor" and node.args:
            first = self.eval_expr(node.args[0], state)
            if isinstance(first, TensorVal):
                self._check_copy_construct(node)
            return None
        torch_ctors = {
            "torch.randn", "torch.rand", "torch.zeros", "torch.ones",
            "torch.empty", "torch.full", "torch.randint", "torch.arange",
        }
        if name in torch_ctors:
            sizes = []
            raw = node.args
            if name == "torch.full" and raw:
                raw = raw[:1]  # full(size, fill) - size is first arg (a tuple)
            for a in raw:
                v = self.eval_expr(a, state)
                if isinstance(v, IntVal):
                    self._check_negative_dim(node, v, f"{name}()")
                    sizes.append(v.sym)
                elif isinstance(v, TupleVal) and v.exact_len:
                    for e in v.elems:
                        if isinstance(e, IntVal):
                            self._check_negative_dim(node, e, f"{name}()")
                        sizes.append(e.sym if isinstance(e, IntVal) and e.sym is not None else None)
                else:
                    sizes.append(None)
            ctor_device, ctor_dtype, ctor_rg = self._ctor_device_dtype(node, state)
            self._check_requires_grad_dtype(node, ctor_dtype, ctor_rg, f"{name}()")
            # A tensor produced directly by a constructor is an autograd *leaf*.
            if name == "torch.arange":
                return TensorVal(
                    rank=1, dtype=ctor_dtype, device=ctor_device,
                    requires_grad=ctor_rg, is_leaf=True,
                ).with_prov(self._prov_label(node, f"{name}(...)"))
            if sizes:
                shown = ", ".join(str(s) if s is not None else "?" for s in sizes)
                return TensorVal(
                    rank=len(sizes), shape=tuple(sizes),
                    dtype=ctor_dtype, device=ctor_device,
                    requires_grad=ctor_rg, is_leaf=True,
                ).with_prov(self._prov_label(node, f"{name}({shown})"))
            return TensorVal(
                rank=None, dtype=ctor_dtype, device=ctor_device,
                requires_grad=ctor_rg, is_leaf=True,
            )
        return None

    def _ctor_device_dtype(self, node, state):
        """Extract the ``device=`` / ``dtype=`` / ``requires_grad=`` kwargs of a
        tensor constructor as normalised values (or ``None`` when absent/unknown)."""
        from .transfer import _device_type

        device = None
        dtype = None
        requires_grad = None
        for kw in getattr(node, "keywords", []) or []:
            if kw.arg == "device":
                v = self.eval_expr(kw.value, state)
                if isinstance(v, StrVal) and v.const:
                    t = _device_type(v.const)
                    if t in ("cpu", "cuda", "mps", "xpu"):
                        device = t
            elif kw.arg == "dtype":
                # dtype is usually ``torch.float32`` (an Attribute) — record the
                # leaf name when we can read it; otherwise leave unknown.
                leaf = _attr_chain(kw.value) if isinstance(kw.value, ast.Attribute) else None
                if leaf and leaf.startswith("torch."):
                    dtype = leaf.split(".")[-1]
            elif kw.arg == "requires_grad":
                v = self.eval_expr(kw.value, state)
                if isinstance(v, BoolVal) and v.const is not None:
                    requires_grad = v.const
                elif isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                    requires_grad = kw.value.value
        return device, dtype, requires_grad

    def _check_negative_dim(self, node, v: "IntVal", where: str) -> None:
        """Flag a tensor dimension that is *provably* negative (``< 0``).  Zero is
        a legal (empty) dimension and ``-1`` inference is only legal for
        view/reshape/expand, which never reach this check."""
        neg = False
        c = v.const
        if c is not None:
            neg = c < 0
        else:
            hi = v.hi()
            neg = hi is not None and hi < 0
        if neg:
            self._emit(
                SymBug(
                    kind=SymBugKind.NEGATIVE_DIMENSION,
                    message=(
                        f"{where} is given a negative dimension; this raises "
                        f"RuntimeError at runtime"
                    ),
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    function=self._cur_func_name(),
                    confidence=0.9,
                    fix_suggestion="ensure the dimension is non-negative",
                )
            )


    def _nn_layer_meta(self, class_name: str, pos, kw) -> Dict[str, int]:
        meta: Dict[str, int] = {}

        def as_int(v):
            if isinstance(v, IntVal):
                if v.sym is not None and v.sym.value is not None:
                    return v.sym.value
                return v.const
            return None

        if class_name in ("Linear",):
            ina = kw.get("in_features", pos[0] if len(pos) >= 1 else None)
            outa = kw.get("out_features", pos[1] if len(pos) >= 2 else None)
            if ina is not None and as_int(ina) is not None:
                meta["in_features"] = as_int(ina)
            if outa is not None and as_int(outa) is not None:
                meta["out_features"] = as_int(outa)
            meta["bias"] = self._bool_flag(
                kw.get("bias", pos[2] if len(pos) >= 3 else None), default=True
            )
        if class_name in ("Conv1d", "Conv2d", "Conv3d"):
            ina = kw.get("in_channels", pos[0] if len(pos) >= 1 else None)
            outa = kw.get("out_channels", pos[1] if len(pos) >= 2 else None)
            if ina is not None and as_int(ina) is not None:
                meta["in_channels"] = as_int(ina)
            if outa is not None and as_int(outa) is not None:
                meta["out_channels"] = as_int(outa)
            spatial = self._CONV_SPATIAL[class_name]
            ks = kw.get("kernel_size", pos[2] if len(pos) >= 3 else None)
            ksdims = self._spatial_dims(ks, spatial)
            if ksdims is not None:
                for i, d in enumerate(ksdims):
                    meta[f"k{i}"] = d
                meta["k_len"] = len(ksdims)
            grp = kw.get("groups", pos[6] if len(pos) >= 7 else None)
            if grp is None or as_int(grp) is not None:
                meta["groups"] = as_int(grp) if grp is not None else 1
            meta["bias"] = self._bool_flag(
                kw.get("bias", pos[7] if len(pos) >= 8 else None), default=True
            )
        if class_name in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d"):
            nf = kw.get("num_features", pos[0] if len(pos) >= 1 else None)
            if nf is not None and as_int(nf) is not None:
                meta["num_features"] = as_int(nf)
            meta["affine"] = self._bool_flag(kw.get("affine"), default=True)
            meta["track_running_stats"] = self._bool_flag(
                kw.get("track_running_stats"), default=True
            )
        if class_name == "Embedding":
            ne = kw.get("num_embeddings", pos[0] if len(pos) >= 1 else None)
            ed = kw.get("embedding_dim", pos[1] if len(pos) >= 2 else None)
            if ne is not None and as_int(ne) is not None:
                meta["num_embeddings"] = as_int(ne)
            if ed is not None and as_int(ed) is not None:
                meta["embedding_dim"] = as_int(ed)
        if class_name == "LayerNorm":
            ns = kw.get("normalized_shape", pos[0] if len(pos) >= 1 else None)
            shape = self._normalized_shape(ns)
            if shape is not None:
                for i, s in enumerate(shape):
                    meta[f"ns{i}"] = s
                meta["ns_len"] = len(shape)
            meta["elementwise_affine"] = self._bool_flag(
                kw.get("elementwise_affine"), default=True
            )
        return meta

    @staticmethod
    def _bool_flag(v, *, default: bool) -> int:
        """Resolve a constructor boolean flag (e.g. ``bias=``) to 1/0, or -1 when
        it is present but not a statically-known bool (so callers can abstain)."""
        if v is None:
            return 1 if default else 0
        if isinstance(v, BoolVal) and v.const is not None:
            return 1 if v.const else 0
        return -1

    @staticmethod
    def _spatial_dims(v, spatial: int) -> Optional[List[int]]:
        """Resolve a Conv ``kernel_size`` (int -> repeated; tuple/list -> as-is)
        into ``spatial`` concrete ints, or ``None`` if not statically known."""
        def as_int(x):
            if isinstance(x, IntVal):
                if x.sym is not None and x.sym.value is not None:
                    return x.sym.value
                return x.const
            return None
        if isinstance(v, IntVal):
            c = as_int(v)
            return [c] * spatial if c is not None else None
        if isinstance(v, TupleVal) and v.exact_len:
            out = [as_int(e) for e in v.elems]
            return out if all(x is not None for x in out) else None
        if isinstance(v, ListVal) and v.exact_elems is not None:
            out = [as_int(e) for e in v.exact_elems]
            return out if all(x is not None for x in out) else None
        return None

    @staticmethod
    def _normalized_shape(v) -> Optional[List[int]]:
        """Read LayerNorm's ``normalized_shape`` (an int or a tuple/list of ints)
        into a concrete list, or ``None`` if not statically known."""
        def as_int(x):
            if isinstance(x, IntVal):
                if x.sym is not None and x.sym.value is not None:
                    return x.sym.value
                return x.const
            return None
        if isinstance(v, IntVal):
            c = as_int(v)
            return [c] if c is not None else None
        if isinstance(v, TupleVal) and v.exact_len:
            out = [as_int(e) for e in v.elems]
            return out if all(x is not None for x in out) else None
        if isinstance(v, ListVal) and v.exact_elems is not None:
            out = [as_int(e) for e in v.exact_elems]
            return out if all(x is not None for x in out) else None
        return None

    # nn-layer construction & application -------------------------------
    _NN_LAYERS = {
        "Linear", "Conv1d", "Conv2d", "Conv3d",
        "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
        "Embedding", "LayerNorm",
    }
    _CONV_SPATIAL = {"Conv1d": 1, "Conv2d": 2, "Conv3d": 3}
    _BATCHNORM = {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}

    def _nn_ctor(self, func, pos, kw) -> Optional[AbstractValue]:
        """Recognise ``nn.Linear(...)`` / ``torch.nn.Linear(...)`` constructor
        calls and produce a ``ModuleVal`` carrying in/out-feature meta."""
        if not isinstance(func, ast.Attribute):
            return None
        chain = _attr_chain(func)
        if chain is None:
            return None
        parts = chain.split(".")
        layer = parts[-1]
        if layer not in self._NN_LAYERS or "nn" not in parts[:-1]:
            return None
        meta = self._nn_layer_meta(layer, pos, kw)
        return ModuleVal(class_name=layer, meta=tuple(meta.items()))

    def _apply_nn_layer(self, recv: ModuleVal, pos, node) -> Optional[AbstractValue]:
        """Apply a modeled nn layer to its input, checking the feature contract
        and returning the output tensor.  Returns ``None`` for unmodeled layers
        so the caller can abstain."""
        if recv.class_name == "Linear":
            return self._apply_linear(recv, pos, node)
        if recv.class_name in self._CONV_SPATIAL:
            return self._apply_conv(recv, pos, node)
        if recv.class_name in self._BATCHNORM:
            return self._apply_batchnorm(recv, pos, node)
        if recv.class_name == "Embedding":
            return self._apply_embedding(recv, pos, node)
        if recv.class_name == "LayerNorm":
            return self._apply_layernorm(recv, pos, node)
        if recv.class_name == "Sequential":
            return self._apply_sequential(recv, pos, node)
        return None

    def _apply_linear(self, recv: ModuleVal, pos, node) -> Optional[AbstractValue]:
        meta = dict(recv.meta)
        in_f = meta.get("in_features")
        out_f = meta.get("out_features")
        if not pos or not isinstance(pos[0], TensorVal):
            return TensorVal(rank=None)
        t = pos[0]
        last = t.dim(t.rank - 1) if t.rank else None
        last_val = last.value if last is not None else None
        if in_f is not None and last_val is not None and last_val != in_f:
            self._emit(
                SymBug(
                    kind=SymBugKind.LAYER_DIM_MISMATCH,
                    message=(
                        f"nn.Linear expects last input dim {in_f} but received "
                        f"{last_val}; RuntimeError at runtime"
                    ),
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    function=self._cur_func_name(),
                    confidence=0.9,
                    fix_suggestion=f"feed a tensor whose last dim is {in_f}, or set in_features={last_val}",
                    evidence=self._witness(t),
                )
            )
        label = f"Linear({in_f}->{out_f})" if (in_f is not None or out_f is not None) else "Linear"
        if t.rank is not None and t.shape is not None and out_f is not None:
            new_shape = list(t.shape)
            new_shape[-1] = SymDim.const_dim(out_f)
            return self._derive(
                TensorVal(rank=t.rank, shape=tuple(new_shape), dtype=t.dtype, device=t.device),
                node, label, t,
            )
        return self._derive(TensorVal(rank=t.rank, dtype=t.dtype, device=t.device), node, label, t)

    def _apply_conv(self, recv: ModuleVal, pos, node) -> Optional[AbstractValue]:
        """Check the in-channels contract of a Conv{1,2,3}d and propagate output
        channels.  The channel axis is ``rank - spatial - 1`` (handles both the
        batched ``(N,C,*)`` and unbatched ``(C,*)`` layouts).  Spatial extents are
        not tracked precisely (kernel/stride/padding) so they become symbolic."""
        cls = recv.class_name
        spatial = self._CONV_SPATIAL[cls]
        meta = dict(recv.meta)
        in_c = meta.get("in_channels")
        out_c = meta.get("out_channels")
        if not pos or not isinstance(pos[0], TensorVal):
            return TensorVal(rank=None)
        t = pos[0]
        r = t.rank
        if r is None or r < spatial + 1:
            return TensorVal(rank=r, dtype=t.dtype, device=t.device)
        ch_idx = r - spatial - 1
        ch = t.dim(ch_idx) if t.shape is not None else None
        ch_val = ch.value if ch is not None else None
        if in_c is not None and ch_val is not None and ch_val != in_c:
            self._emit(
                SymBug(
                    kind=SymBugKind.LAYER_DIM_MISMATCH,
                    message=(
                        f"nn.{cls} expects {in_c} input channels but received "
                        f"{ch_val} (channel dim {ch_idx}); RuntimeError at runtime"
                    ),
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    function=self._cur_func_name(),
                    confidence=0.9,
                    fix_suggestion=f"feed a tensor with {in_c} channels, or set in_channels={ch_val}",
                )
            )
        if t.shape is not None and out_c is not None:
            new_shape = list(t.shape)
            new_shape[ch_idx] = SymDim.const_dim(out_c)
            for i in range(ch_idx + 1, r):  # spatial extents become unknown
                new_shape[i] = None
            return TensorVal(rank=r, shape=tuple(new_shape), dtype=t.dtype, device=t.device)
        return TensorVal(rank=r, dtype=t.dtype, device=t.device)

    def _layer_bug(self, node, message: str, fix: str) -> None:
        self._emit(
            SymBug(
                kind=SymBugKind.LAYER_DIM_MISMATCH,
                message=message,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function=self._cur_func_name(),
                confidence=0.9,
                fix_suggestion=fix,
            )
        )

    def _apply_batchnorm(self, recv: ModuleVal, pos, node) -> Optional[AbstractValue]:
        """``BatchNorm{1,2,3}d`` requires a batched input whose channel dim (dim 1)
        equals ``num_features``.  Output shape is preserved."""
        cls = recv.class_name
        nf = dict(recv.meta).get("num_features")
        if not pos or not isinstance(pos[0], TensorVal):
            return TensorVal(rank=None)
        t = pos[0]
        if t.rank is not None and t.rank >= 2 and t.shape is not None:
            ch = t.dim(1)
            ch_val = ch.value if ch is not None else None
            if nf is not None and ch_val is not None and ch_val != nf:
                self._layer_bug(
                    node,
                    f"nn.{cls} expects {nf} channels (num_features) but received "
                    f"{ch_val} at dim 1; RuntimeError at runtime",
                    f"feed a tensor with {ch_val} channels, or set num_features={ch_val}",
                )
        return TensorVal(rank=t.rank, shape=t.shape, dtype=t.dtype, device=t.device)

    def _apply_embedding(self, recv: ModuleVal, pos, node) -> Optional[AbstractValue]:
        """``Embedding`` maps an index tensor of shape ``S`` to ``S + (embedding_dim,)``
        — rank grows by one and the new last dim is ``embedding_dim``.  (Index
        *values* are tensor contents, not shape, so they are not checked here.)"""
        ed = dict(recv.meta).get("embedding_dim")
        if not pos or not isinstance(pos[0], TensorVal):
            return TensorVal(rank=None)
        t = pos[0]
        if t.rank is None:
            return TensorVal(rank=None, dtype=t.dtype, device=t.device)
        if t.shape is not None and ed is not None:
            new_shape = tuple(t.shape) + (SymDim.const_dim(ed),)
            return TensorVal(rank=t.rank + 1, shape=new_shape, dtype=t.dtype, device=t.device)
        return TensorVal(rank=t.rank + 1, dtype=t.dtype, device=t.device)

    def _apply_layernorm(self, recv: ModuleVal, pos, node) -> Optional[AbstractValue]:
        """``LayerNorm(normalized_shape)`` requires the input's trailing dims to
        equal ``normalized_shape``.  Output shape is preserved."""
        meta = dict(recv.meta)
        ns_len = meta.get("ns_len")
        if not pos or not isinstance(pos[0], TensorVal):
            return TensorVal(rank=None)
        t = pos[0]
        if ns_len is not None and t.rank is not None and t.shape is not None and t.rank >= ns_len:
            ns = [meta.get(f"ns{i}") for i in range(ns_len)]
            for j in range(ns_len):
                dim_idx = t.rank - ns_len + j
                d = t.dim(dim_idx)
                dv = d.value if d is not None else None
                if ns[j] is not None and dv is not None and dv != ns[j]:
                    self._layer_bug(
                        node,
                        f"nn.LayerNorm normalized_shape {ns} does not match input "
                        f"trailing dim {dim_idx} ({dv} vs {ns[j]}); RuntimeError at runtime",
                        f"set normalized_shape to match the input trailing dims, or feed dim {dim_idx}={ns[j]}",
                    )
                    break
        return TensorVal(rank=t.rank, shape=t.shape, dtype=t.dtype, device=t.device)

    # -- Step 50: nn.Sequential (higher-order container) ----------------
    # Feature/shape-preserving children that the feature chain passes through.
    _TRANSPARENT_LAYERS = frozenset(
        {
            "ReLU", "ReLU6", "LeakyReLU", "PReLU", "RReLU", "ELU", "CELU", "SELU",
            "GELU", "SiLU", "Mish", "Hardswish", "Hardtanh", "Hardsigmoid",
            "Sigmoid", "Tanh", "Softmax", "Softmax2d", "LogSoftmax", "Softplus",
            "Softsign", "Tanhshrink", "GLU", "Dropout", "Dropout1d", "Dropout2d",
            "Dropout3d", "AlphaDropout", "Identity",
        }
    )

    def _layer_io(self, child):
        """Return ``(in_feat, out_feat, kind, transparent)`` for a Sequential
        child, where ``kind`` ('feat' = last/feature dim, 'chan' = channel dim)
        constrains which adjacent layers may be compared, ``transparent`` marks a
        shape/feature-preserving layer the chain passes through, and ``None``
        in/out means unknown.  Unknown/custom modules return all-None (breaking
        the chain) so we never compare across something we don't understand."""
        if not isinstance(child, ModuleVal):
            return (None, None, None, False)
        cn = child.class_name
        m = dict(child.meta)
        if cn == "Linear":
            return (m.get("in_features"), m.get("out_features"), "feat", False)
        if cn in self._CONV_SPATIAL:
            return (m.get("in_channels"), m.get("out_channels"), "chan", False)
        if cn == "Embedding":
            return (None, m.get("embedding_dim"), "feat", False)
        if cn in self._TRANSPARENT_LAYERS or cn in self._BATCHNORM or cn == "LayerNorm":
            return (None, None, None, True)
        return (None, None, None, False)

    def _seq_children(self, recv: ModuleVal):
        """Ordered child values of a Sequential (attrs keyed by index string)."""
        items = []
        for k, v in recv.attrs:
            if isinstance(k, str) and k.isdigit():
                items.append((int(k), v))
        items.sort()
        return [v for _, v in items]

    def _apply_sequential(self, recv: ModuleVal, pos, node) -> Optional[AbstractValue]:
        """Apply a ``nn.Sequential`` to its input.

        When the input rank is known, thread the real tensor through each modeled
        child (`_apply_nn_layer`) so per-layer contracts and shape propagation
        fire exactly as for a hand-written forward.  When the input rank is
        *unknown* — the case a plain ``forward(self, x)`` entry leaves us in — run
        an input-independent **structural feature-chain check**: adjacent
        same-kind layers whose out/in feature dims are both known and unequal are
        a guaranteed runtime error regardless of input."""
        children = self._seq_children(recv)
        inp = pos[0] if pos else None
        if isinstance(inp, TensorVal) and inp.rank is not None:
            cur: AbstractValue = inp
            for child in children:
                if isinstance(child, ModuleVal) and isinstance(cur, TensorVal):
                    out = self._apply_nn_layer(child, [cur], node)
                    cur = out if isinstance(out, TensorVal) else TensorVal(rank=None)
                else:
                    cur = TensorVal(rank=None)
            return self._derive(cur, node, "Sequential", inp) if isinstance(cur, TensorVal) else cur
        # input rank unknown -> structural same-kind feature-chain check
        self._seq_structural_check(children, node)
        return TensorVal(rank=None)

    def _seq_structural_check(self, children, node) -> None:
        pending = None  # (feature_value, kind, producer_label)
        for child in children:
            in_f, out_f, kind, transparent = self._layer_io(child)
            if transparent:
                continue
            cn = child.class_name if isinstance(child, ModuleVal) else "?"
            if (
                in_f is not None
                and pending is not None
                and pending[1] == kind
                and pending[0] != in_f
            ):
                self._layer_bug(
                    node,
                    f"nn.Sequential: {pending[2]} produces feature dim {pending[0]} "
                    f"but the following nn.{cn} expects {in_f}; RuntimeError at runtime",
                    f"match the feature dims of adjacent layers ({pending[0]} -> {in_f})",
                )
                pending = (out_f, kind, f"nn.{cn}") if out_f is not None else None
                continue
            if out_f is not None:
                pending = (out_f, kind, f"nn.{cn}")
            else:
                pending = None

    def _seq_ctor(self, func, node, pos, state) -> Optional[AbstractValue]:
        """Recognise ``nn.Sequential(layer0, layer1, ...)`` (and the common
        ``nn.Sequential(*layers)`` form) and build a Sequential ``ModuleVal`` whose
        children are stored as index-keyed attrs (mirroring PyTorch's own naming),
        so it can later be applied or indexed."""
        if not isinstance(func, ast.Attribute):
            return None
        chain = _attr_chain(func)
        if chain is None:
            return None
        parts = chain.split(".")
        if parts[-1] != "Sequential" or "nn" not in parts[:-1]:
            return None
        # Named form: nn.Sequential(OrderedDict([("conv", ...), ("bn", ...)])) — the
        # children get their *declared* names (not 0/1/...).  Parsed from the AST so
        # insertion order is preserved (a sorted DictVal would reorder the chain).
        if (
            len(node.args) == 1
            and not node.keywords
            and not isinstance(node.args[0], ast.Starred)
        ):
            named = self._seq_named_children(node.args[0], state)
            if named is not None:
                return ModuleVal(class_name="Sequential", attrs=tuple(named))
            if self._is_named_seq_arg(node.args[0]):
                # A recognized dict/OrderedDict argument we could NOT statically
                # enumerate: abstain on the whole container rather than mis-model
                # it as a single positional child.
                return ModuleVal(
                    class_name="Sequential", meta=self._OPAQUE_CONTAINER_META
                )
        children = list(pos)
        if not children and len(node.args) == 1 and isinstance(node.args[0], ast.Starred):
            seqv = self.eval_expr(node.args[0].value, state)
            if isinstance(seqv, ListVal) and seqv.exact_elems is not None:
                children = list(seqv.exact_elems)
            elif isinstance(seqv, TupleVal) and seqv.exact_len:
                children = list(seqv.elems)
            else:
                # nn.Sequential(*xs) where xs is NOT statically enumerable: abstain
                # on the container (its child count is unknown) rather than model an
                # empty Sequential, mirroring the ModuleList splat policy.
                return ModuleVal(
                    class_name="Sequential", meta=self._OPAQUE_CONTAINER_META
                )
            arg_nodes = [None] * len(children)
        else:
            arg_nodes = list(node.args)
        # Upgrade children that didn't model to a ModuleVal (e.g. unmodeled
        # activations like ``nn.ReLU()``) into a bare class-named ModuleVal, so the
        # structural feature-chain check can pass *through* them rather than
        # breaking on an opaque TOP.
        upgraded = []
        for i, c in enumerate(children):
            if not isinstance(c, ModuleVal) and i < len(arg_nodes):
                cls = self._nn_class_name(arg_nodes[i])
                if cls is not None:
                    c = ModuleVal(class_name=cls)
            upgraded.append(c)
        children = upgraded
        attrs = tuple((str(i), c) for i, c in enumerate(children))
        return ModuleVal(class_name="Sequential", attrs=attrs)

    @staticmethod
    def _callee_name(func) -> Optional[str]:
        """The bare callee name of a call target (``OrderedDict`` for both
        ``OrderedDict(...)`` and ``collections.OrderedDict(...)``)."""
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _is_named_seq_arg(self, arg_node: ast.expr) -> bool:
        """True if ``arg_node`` is a dict-shaped Sequential argument (an
        ``OrderedDict(...)`` call or a dict literal) — i.e. the *named* form,
        regardless of whether we can statically enumerate it."""
        if isinstance(arg_node, ast.Dict):
            return True
        return (
            isinstance(arg_node, ast.Call)
            and self._callee_name(arg_node.func) in ("OrderedDict", "dict")
        )

    def _ordered_named_pairs(self, arg_node: ast.expr, state: State):
        """Insertion-ordered ``(key_str, value_node)`` pairs of a dict-shaped
        argument (an ``OrderedDict(...)`` call or a dict literal), or ``None`` if
        it is not statically a constant-string-keyed mapping.  ``value_node`` is
        the AST of each value (used to upgrade unmodeled activations), or ``None``
        when only the abstract value is available."""
        # dict literal: {"a": x, "b": y}
        if isinstance(arg_node, ast.Dict):
            pairs = []
            for k, v in zip(arg_node.keys, arg_node.values):
                if k is None or not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    return None
                pairs.append((k.value, v))
            return pairs
        if not (isinstance(arg_node, ast.Call)
                and self._callee_name(arg_node.func) in ("OrderedDict", "dict")):
            return None
        # OrderedDict(a=x, b=y) — keyword form (insertion order = source order)
        if arg_node.keywords and not arg_node.args:
            pairs = []
            for kw in arg_node.keywords:
                if kw.arg is None:  # **spread — unknown keys
                    return None
                pairs.append((kw.arg, kw.value))
            return pairs
        # OrderedDict() — empty
        if not arg_node.args and not arg_node.keywords:
            return []
        # OrderedDict(<single arg>): a list/tuple of (key, value) pairs, or a dict
        if len(arg_node.args) == 1 and not arg_node.keywords:
            inner = arg_node.args[0]
            if isinstance(inner, ast.Dict):
                return self._ordered_named_pairs(inner, state)
            if isinstance(inner, (ast.List, ast.Tuple)):
                pairs = []
                for elt in inner.elts:
                    if not (isinstance(elt, ast.Tuple) and len(elt.elts) == 2):
                        return None
                    k_node, v_node = elt.elts
                    if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                        return None
                    pairs.append((k_node.value, v_node))
                return pairs
            # a name/expr bound to an enumerable list/tuple of (key, value) pairs
            seq = self._pairs_from_value(self.eval_expr(inner, state))
            if seq is not None:
                return [(k, None, v) for k, v in seq]  # marked: value already abstract
        return None

    def _seq_named_children(self, arg_node: ast.expr, state: State):
        """Resolve the named children of ``nn.Sequential(OrderedDict(...))`` to a
        list of ``(name, ModuleVal/value)`` attrs (insertion order preserved), or
        ``None`` when the argument is not a statically-enumerable named mapping."""
        pairs = self._ordered_named_pairs(arg_node, state)
        if pairs is None:
            return None
        out = []
        for pair in pairs:
            if len(pair) == 3:  # (key, None, abstract_value) — pre-evaluated
                key, _, val = pair
            else:
                key, v_node = pair
                val = self.eval_expr(v_node, state)
                if not isinstance(val, ModuleVal):
                    cls = self._nn_class_name(v_node)
                    if cls is not None:
                        val = ModuleVal(class_name=cls)
            out.append((str(key), val))
        return out

    @staticmethod
    def _pairs_from_value(v) -> Optional[list]:
        """A statically-enumerable sequence of 2-element ``(str, value)`` tuples →
        a list of ``(str, value)``; otherwise ``None``."""
        if isinstance(v, ListVal) and v.exact_elems is not None:
            elems = list(v.exact_elems)
        elif isinstance(v, TupleVal) and v.exact_len:
            elems = list(v.elems)
        else:
            return None
        out = []
        for e in elems:
            if isinstance(e, TupleVal) and e.exact_len and len(e.elems) == 2:
                k, val = e.elems
                if isinstance(k, StrVal) and k.const is not None:
                    out.append((k.const, val))
                    continue
            return None
        return out

    def _ordered_dict_ctor(self, func, node, pos, kw, state) -> Optional[AbstractValue]:
        """Model ``OrderedDict(...)`` / ``collections.OrderedDict(...)`` as a
        ``DictVal`` (constant-string keys → ``exact_keys``), so nn.Sequential /
        nn.ModuleDict built from one name their children faithfully.  Returns
        ``None`` when the callee is not ``OrderedDict``."""
        if self._callee_name(func) != "OrderedDict":
            return None
        pairs = self._ordered_named_pairs(node, state)
        if pairs is None:
            return DictVal(value=TOP, known=(), exact_keys=False)
        known = []
        for pair in pairs:
            if len(pair) == 3:
                key, _, val = pair
            else:
                key, v_node = pair
                val = self.eval_expr(v_node, state)
            known.append((key, val))
        return DictVal(
            value=join_many([v for _, v in known]) if known else TOP,
            known=tuple(known),
            exact_keys=True,
        )

    def _container_mutate(self, target_node, recv: "ModuleVal", attr: str, pos, node, state: State) -> None:
        """Apply ``recv.<attr>(...)`` (``append`` / ``extend`` / ``insert``) to a
        registered ``ModuleList`` ``ModuleVal``, rebuilding it with contiguous
        ``0..n-1`` index keys (mirroring PyTorch's ``state_dict`` naming) and
        writing the result back to ``target_node``.

        Soundness: an already-opaque container stays opaque; an ``extend`` whose
        argument is not statically enumerable, or an ``insert`` at a non-constant
        position, makes the whole container opaque (abstain) rather than guessing.
        Only a simple ``Name`` / ``self.<attr>`` receiver is rebound; a deeper
        l-value is left as-is (already over-approximated by the caller)."""
        # Children always re-indexed 0..n-1 (ModuleList ignores any prior keys).
        existing = [v for _, v in recv.attrs]

        def reindex(children):
            return tuple((str(i), c) for i, c in enumerate(children))

        def go_opaque() -> None:
            self._rebind_lvalue(
                target_node,
                ModuleVal(class_name=recv.class_name, meta=self._OPAQUE_CONTAINER_META),
                state,
            )

        if recv.get_meta("__opaque_container__") == 1:
            return  # already non-enumerable — leave opaque

        if attr == "append":
            if not pos:
                return
            existing.append(pos[0])
        elif attr == "extend":
            seq = self._enumerable_seq(pos[0]) if pos else None
            if seq is None:
                go_opaque()
                return
            existing.extend(seq)
        elif attr == "insert":
            if len(pos) < 2:
                return
            idx = pos[0].const if isinstance(pos[0], IntVal) else None
            if idx is None:
                go_opaque()
                return
            n = len(existing)
            if idx < 0:
                idx += n
            idx = max(0, min(idx, n))
            existing.insert(idx, pos[1])
        else:  # pragma: no cover - dispatch guards this
            return

        self._rebind_lvalue(
            target_node, ModuleVal(class_name=recv.class_name, attrs=reindex(existing)), state
        )

    # -- Step 11: nn.ModuleList / nn.ModuleDict (registered containers) ------
    #: Marker stored in a container ``ModuleVal``'s ``meta`` when its registered
    #: children could not be statically enumerated, so the contract deriver
    #: abstains on the subtree rather than guessing (or silently dropping it).
    _OPAQUE_CONTAINER_META = (("__opaque_container__", 1),)

    @staticmethod
    def _enumerable_seq(v) -> Optional[list]:
        """The exact element list of a statically-enumerable sequence, else None."""
        if isinstance(v, ListVal) and v.exact_elems is not None:
            return list(v.exact_elems)
        if isinstance(v, TupleVal) and v.exact_len:
            return list(v.elems)
        return None

    @staticmethod
    def _enumerable_map(v) -> Optional[list]:
        """The exact ``(key, value)`` items of a statically-enumerable string-keyed
        dict, else None."""
        if isinstance(v, DictVal) and v.exact_keys:
            return list(v.known)
        return None

    def _module_container_ctor(self, func, node, pos, state) -> Optional[AbstractValue]:
        """Recognise ``nn.ModuleList([...])`` / ``nn.ModuleDict({...})`` and build a
        container ``ModuleVal`` whose registered children are index-/key-addressed
        attrs (mirroring PyTorch's ``state_dict`` naming: ``list.0.*`` /
        ``dict.key.*``).

        Only ``nn.ModuleList`` / ``nn.ModuleDict`` (and ``nn.Sequential``) register
        their children — a *plain* ``list``/``dict``/``tuple`` attribute does not —
        so this is the sole path that turns a container of modules into walked,
        emitted submodules.  When the contents are not statically enumerable (e.g.
        a comprehension over a symbolic count) we return an **opaque** container
        marker so the deriver abstains on that subtree rather than guessing."""
        if not isinstance(func, ast.Attribute):
            return None
        chain = _attr_chain(func)
        if chain is None:
            return None
        parts = chain.split(".")
        name = parts[-1]
        if name not in ("ModuleList", "ModuleDict") or "nn" not in parts[:-1]:
            return None

        if name == "ModuleList":
            if not node.args:
                return ModuleVal(class_name="ModuleList")
            arg = pos[0] if pos else self.eval_expr(node.args[0], state)
            children = self._enumerable_seq(arg)
            if children is None:
                return ModuleVal(class_name="ModuleList",
                                 meta=self._OPAQUE_CONTAINER_META)
            attrs = tuple((str(i), c) for i, c in enumerate(children))
            return ModuleVal(class_name="ModuleList", attrs=attrs)

        # ModuleDict
        if not node.args:
            return ModuleVal(class_name="ModuleDict")
        arg = pos[0] if pos else self.eval_expr(node.args[0], state)
        items = self._enumerable_map(arg)
        if items is None:
            return ModuleVal(class_name="ModuleDict",
                             meta=self._OPAQUE_CONTAINER_META)
        attrs = tuple((str(k), c) for k, c in items)
        return ModuleVal(class_name="ModuleDict", attrs=attrs)

    @staticmethod
    def _nn_class_name(arg_node) -> Optional[str]:
        """If ``arg_node`` is an ``nn.<Class>(...)`` constructor call, return
        ``<Class>``; otherwise ``None``."""
        if not isinstance(arg_node, ast.Call):
            return None
        chain = _attr_chain(arg_node.func) if isinstance(arg_node.func, ast.Attribute) else None
        if chain is None:
            return None
        parts = chain.split(".")
        if "nn" in parts[:-1]:
            return parts[-1]
        return None

    def _resolve_callback(self, arg_node):
        """Resolve a callback passed by name to its ``FunctionDef`` (for ``map`` /
        ``apply`` / similar higher-order calls), or ``None``."""
        if isinstance(arg_node, ast.Name):
            return self._lookup_func_by_name(arg_node.id)
        return None


    def _lookup_func_by_name(self, name: str) -> Optional[ast.FunctionDef]:
        for node in self.module.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def _cur_func_name(self) -> str:
        return self._frames[-1].func.name if self._frames else "<module>"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _known_truth(v: AbstractValue) -> Optional[bool]:
    if isinstance(v, BoolVal):
        return v.const
    if isinstance(v, NoneVal):
        return False
    if isinstance(v, IntVal) and v.sym is not None and v.sym.value is not None:
        return v.sym.value != 0
    return None


def _unpack_arity(value: AbstractValue) -> Optional[int]:
    """Return the fixed number of values ``value`` unpacks into, or ``None`` when
    that is not statically determined (in which case we must not report)."""
    if isinstance(value, NoneVal):
        return 0  # not iterable -> definitely wrong for any n>=1
    if isinstance(value, TupleVal) and value.exact_len:
        return len(value.elems)
    if isinstance(value, ListVal) and value.length is not None and value.exact_elems is not None:
        return value.length
    if isinstance(value, TensorVal):
        # a tensor unpacks along dim 0; only known when dim0 is a known constant
        if value.rank == 0:
            return 0
        if value.shape is not None and value.rank and value.shape[0] is not None:
            d0 = value.shape[0].value
            if d0 is not None:
                return d0
    return None


def _value_elems(value: AbstractValue, n: int) -> List[AbstractValue]:
    if isinstance(value, TupleVal) and value.exact_len and len(value.elems) == n:
        return list(value.elems)
    if isinstance(value, ListVal) and value.exact_elems is not None and len(value.exact_elems) == n:
        return list(value.exact_elems)
    return [TOP] * n


def _element_of(value: AbstractValue) -> AbstractValue:
    if isinstance(value, ListVal):
        return value.elem
    if isinstance(value, SetVal):
        return value.elem
    if isinstance(value, TupleVal) and value.elems:
        return join_many(list(value.elems))
    if isinstance(value, StrVal):
        return StrVal()  # iterating a str yields 1-char strs
    if isinstance(value, DictVal):
        # iterating a dict yields its keys; our ``known`` map only tracks string
        # keys, and ``exact_keys`` is only True when every key is such a string.
        return StrVal() if value.exact_keys else TOP
    if isinstance(value, TensorVal) and value.rank is not None and value.rank >= 1:
        return TensorVal(rank=value.rank - 1, dtype=value.dtype, device=value.device)
    return TOP


def _const_index(slc) -> Optional[int]:
    node = slc.value if isinstance(slc, ast.Index) else slc  # py<3.9 compat
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return None


def _const_str(slc) -> Optional[str]:
    node = slc.value if isinstance(slc, ast.Index) else slc
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_none_literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _shape_str(shape) -> str:
    parts = [str(d.value) if (d is not None and d.value is not None) else "?" for d in shape]
    return "(" + ", ".join(parts) + ")"


def _constraint_str(c) -> str:
    """Human-readable form of a ``smt_bridge.DimConstraint`` (e.g. ``a != b``,
    ``a % 4 == 0``)."""
    if c.op in ("%==0", "%!=0"):
        sign = "== 0" if c.op == "%==0" else "!= 0"
        return f"{c.lhs} % {c.rhs} {sign}"
    return f"{c.lhs} {c.op} {c.rhs}"


# In-place autograd-tracked mutators that raise when applied to a *leaf* tensor
# requiring grad.  Empirically verified against torch; ``requires_grad_`` and
# ``detach_`` are deliberately excluded (they are permitted on such leaves).
_INPLACE_AUTOGRAD_OPS = frozenset({
    "add_", "sub_", "mul_", "div_", "pow_", "neg_", "abs_", "relu_",
    "sigmoid_", "tanh_", "exp_", "log_", "sqrt_", "clamp_", "clamp_min_",
    "clamp_max_", "fill_", "zero_", "copy_", "t_",
})


# Pure (out-of-place) tensor transforms whose *only* effect is the returned
# tensor.  Calling one as a bare statement (discarding the result) is a no-op and
# almost always an intent bug — the author expected an in-place mutation.  These
# never raise, so the finding is heuristic-only (Step 86 ``heuristic`` mode).
_PURE_TENSOR_TRANSFORMS = frozenset({
    "to", "cuda", "cpu", "float", "double", "half", "long", "int", "bool",
    "detach", "clone", "contiguous", "view", "reshape", "flatten", "permute",
    "transpose", "t", "squeeze", "unsqueeze", "expand", "expand_as",
    "type", "type_as", "abs", "neg", "softmax", "log_softmax",
})


_NON_DIFF_DTYPES = frozenset({
    "uint8", "int8", "int16", "int32", "int64", "long", "short", "int",
    "uint16", "uint32", "uint64", "bool", "byte", "char",
})


def _is_module_base(base: ast.expr) -> bool:
    """Whether a class base expression denotes ``nn.Module`` — either the
    attribute form ``nn.Module`` / ``torch.nn.Module`` or a bare ``Module``."""
    if isinstance(base, ast.Attribute):
        return base.attr == "Module"
    if isinstance(base, ast.Name):
        return base.id == "Module"
    return False


def _is_super_call(node: ast.expr) -> bool:
    """Whether ``node`` is a ``super()`` call (the common zero-arg form, or the
    explicit ``super(Cls, self)`` form) — the receiver of a ``super().<attr>``
    delegation (Step 16)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
    )


def _known_numel(recv) -> Optional[int]:
    """Total element count of ``recv`` when every dim is a known constant."""
    shape = getattr(recv, "shape", None)
    if shape is None:
        return None
    numel = 1
    for d in shape:
        if d is None or d.value is None:
            return None
        numel *= d.value
    return numel


def _symdim_to_shape_entry(sym: Optional[SymDim]):
    """Map an affine ``SymDim`` to a reshape-theory dimension entry: an ``int``
    (a constant size), a ``str`` (a single dimension variable, coefficient 1), or
    ``None`` when the affine form (a sum / scaled variable) cannot be represented
    as one theory dim — in which case the caller must abstain."""
    if sym is None:
        return None
    if sym.is_const:
        return int(sym.const)
    if len(sym.terms) == 1 and sym.const == 0 and sym.terms[0][1] == 1:
        return sym.terms[0][0]
    return None


def _reshape_target_entries(pos: List[AbstractValue]):
    """Like :func:`_reshape_target_sizes` but preserves *symbolic* sizes as their
    dimension-variable name, so the reshape theory can reason about them.  Each
    entry is an ``int`` (constant, incl. ``-1`` infer) or a ``str`` (dim var);
    returns ``None`` if any size is neither."""
    if len(pos) == 1 and isinstance(pos[0], (TupleVal, ListVal)):
        container = pos[0]
        elems = container.elems if isinstance(container, TupleVal) else container.exact_elems
        if elems is None:
            return None
        pos = list(elems)
    out = []
    for v in pos:
        if not isinstance(v, IntVal):
            return None
        c = v.const
        if c is not None:
            out.append(c)
            continue
        ent = _symdim_to_shape_entry(v.sym)
        if ent is None:
            return None
        out.append(ent)
    return out or None


def _reshape_target_sizes(pos: List[AbstractValue]) -> Optional[List[int]]:
    """Extract the requested sizes from a ``view``/``reshape`` call when all are
    statically-known ints.  Supports a single tuple/list size argument.  Returns
    ``None`` if any size is unknown."""
    if len(pos) == 1 and isinstance(pos[0], (TupleVal, ListVal)):
        container = pos[0]
        elems = container.elems if isinstance(container, TupleVal) else container.exact_elems
        if elems is None:
            return None
        pos = list(elems)
    out: List[int] = []
    for v in pos:
        if isinstance(v, IntVal):
            c = v.const
            if c is None:
                return None
            out.append(c)
        else:
            return None
    return out or None


def _const_int_of(node: ast.expr) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


_CMP_NAMES = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}
_CMP_SWAP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
_CMP_NEG = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}


def _op_name(op) -> Optional[str]:
    return _CMP_NAMES.get(type(op))


def _swap_op(op) -> Optional[str]:
    name = _op_name(op)
    return _CMP_SWAP.get(name) if name else None


def _negate_op(name: str) -> str:
    return _CMP_NEG[name]


def _exclude_zero(iv: "IntVal"):
    """Return ``iv`` narrowed to exclude 0, ``CONTRA`` when that is infeasible
    (the value is provably exactly 0), or ``iv`` unchanged when 0 cannot be
    excluded from a single interval."""
    lo, hi = iv.lo(), iv.hi()
    if lo is not None and lo == 0:
        return _mk_int_range(1, hi)
    if hi is not None and hi == 0:
        return _mk_int_range(lo, -1)
    # 0 strictly interior or unbounded straddling 0: a single interval cannot
    # express the hole, so keep the value unchanged (sound).
    return iv


def _provably_nonzero(iv: "IntVal") -> bool:
    c = iv.const
    if c is not None:
        return c != 0
    lo, hi = iv.lo(), iv.hi()
    if lo is not None and lo > 0:
        return True
    if hi is not None and hi < 0:
        return True
    return False


def _is_definitely_non_none(v: "AbstractValue") -> bool:
    """True for values whose type rules out ``None`` (so ``x is None`` is
    infeasible).  ``Top``/``Bottom``/``NoneVal`` return False."""
    return isinstance(
        v, (IntVal, FloatVal, BoolVal, StrVal, TensorVal, TupleVal, ListVal, DictVal, ModuleVal)
    )


def _narrow_int(iv: "IntVal", op: str, c: int):
    """Narrow ``iv`` by ``op c``.  Returns a new ``IntVal``, ``CONTRA`` when the
    constraint is infeasible, or ``iv`` when no tightening is representable."""
    lo, hi = iv.lo(), iv.hi()
    if op == "==":
        if lo is not None and c < lo:
            return CONTRA
        if hi is not None and c > hi:
            return CONTRA
        return int_const(c)
    if op == "!=":
        if iv.const == c:
            return CONTRA
        if lo is not None and lo == c:
            return _mk_int_range(c + 1, hi)
        if hi is not None and hi == c:
            return _mk_int_range(lo, c - 1)
        return iv
    if op == "<":
        return _mk_int_range(lo, _min(hi, c - 1))
    if op == "<=":
        return _mk_int_range(lo, _min(hi, c))
    if op == ">":
        return _mk_int_range(_max(lo, c + 1), hi)
    if op == ">=":
        return _mk_int_range(_max(lo, c), hi)
    return iv


def _min(a: Optional[int], b: int) -> int:
    return b if a is None else min(a, b)


def _max(a: Optional[int], b: int) -> int:
    return b if a is None else max(a, b)


def _mk_int_range(lo: Optional[int], hi: Optional[int]):
    from .values import int_range

    if lo is not None and hi is not None and lo > hi:
        return CONTRA  # empty range ⇒ infeasible path
    return int_range(lo, hi)


def _slice_elts(slc):
    node = slc.value if isinstance(slc, ast.Index) else slc
    if isinstance(node, ast.Tuple):
        return list(node.elts)
    return [node]


def _num_index_dims(slc) -> Optional[int]:
    """Number of dimensions consumed by a subscript's index expression.

    ``Ellipsis`` makes the count unknown (returns ``None``) so we never
    over-count and false-report.  ``None`` (np.newaxis) does not consume a
    source dim, so it is excluded.
    """
    elts = _slice_elts(slc)
    count = 0
    for e in elts:
        if isinstance(e, ast.Constant) and e.value is Ellipsis:
            return None
        if isinstance(e, ast.Starred):
            return None
        # `None`/np.newaxis inserts, does not consume
        if isinstance(e, ast.Constant) and e.value is None:
            continue
        count += 1
    return count


def _index_rank_delta(slc) -> Tuple[Optional[int], int]:
    """(dims consumed by integer indices, dims inserted by None) for rank math."""
    elts = _slice_elts(slc)
    consumed = 0
    inserted = 0
    for e in elts:
        if isinstance(e, ast.Constant) and e.value is Ellipsis:
            return None, inserted
        if isinstance(e, ast.Constant) and e.value is None:
            inserted += 1
            continue
        if isinstance(e, ast.Slice):
            continue  # keeps the dim
        # integer index (Constant int, Name, UnaryOp) consumes a dim
        consumed += 1
    return consumed, inserted


def _attr_chain(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _has_loop_control(body) -> bool:
    """True if any statement in ``body`` contains a ``break``/``continue``/
    ``return``/``yield`` that would make a flat sequential unrolling unsound
    (the loop may not run its full constant trip count, or it produces values).
    Nested function/lambda bodies define their own scope; their ``return``/
    ``yield`` does not affect this loop, so they are skipped.  Nested *loops*
    are descended into (their break/continue bind to them, but conservatively
    treating any as control still only falls back to the sound fixpoint)."""
    stack: List[ast.AST] = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.Break, ast.Continue, ast.Return, ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue  # separate scope — its control flow is local
        stack.extend(ast.iter_child_nodes(n))
    return False


def _interval_binop(op, ia, ib):
    """Interval arithmetic for ``+ - * // %``; ``None`` when unknown/unsupported."""
    if ia is None or ib is None:
        return None
    try:
        if isinstance(op, ast.Add):
            return ia + ib
        if isinstance(op, ast.Sub):
            return ia - ib
        if isinstance(op, ast.Mult):
            return ia * ib
    except Exception:
        return None
    return None
