"""Mutation testing for the symbolic-execution engine (roadmap Step 80).

Where the fuzzer (Step 79) mutates *inputs* to find crashes, mutation testing
mutates the **engine's own source** and asks whether an oracle (the test
suite's invariants) *kills* each mutant.  A surviving mutant is a behaviour the
engine can change without any check noticing — i.e. a gap in the suite.  The
**mutation score** (killed / total) is the headline evidence that the suite
constrains the engine.

The harness is torch-free and fully in-process:

* :func:`generate_mutants` applies one classic source mutation at a time
  (comparison flip, arithmetic/boolean operator swap, constant off-by-one,
  boolean-constant flip) to a target module's AST, round-tripping through
  :func:`ast.unparse`.
* :func:`load_mutant_module` execs a mutated source string into a *fresh*
  module object whose ``__package__`` resolves relative imports to the real,
  unmutated siblings — so only the target module's behaviour changes.
* :func:`run_mutation_testing` loads each mutant and runs an **oracle**
  callable against it; the mutant is *killed* if the oracle returns ``False``
  or raises (including import/compile errors), and *survives* otherwise.

Two oracles are shipped:

* :func:`lattice_oracle` — checks the value-lattice laws on ``values.py``
  (a dense, pure target; most mutants die).
* :func:`make_corpus_oracle` — runs the curated corpus end-to-end through a
  freshly-built engine bound to the *mutant* ``interpreter.py`` and compares
  the reported bug kinds to ground truth.

Nothing here runs as part of normal analysis; it is a meta-test of the engine.
"""

from __future__ import annotations

import ast
import copy
import importlib
import sys
import types
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Mutant",
    "MutationReport",
    "generate_mutants",
    "load_mutant_module",
    "run_mutation_testing",
    "run_pytest_mutation_testing",
    "lattice_oracle",
    "make_corpus_oracle",
    "MUTATION_OPERATORS",
]


# --------------------------------------------------------------------------- #
# Mutation operators                                                          #
# --------------------------------------------------------------------------- #

# Each comparison operator maps to the mutation we apply to it.  We deliberately
# pick a *meaning-changing* counterpart (not merely the boundary variant) so a
# correct engine almost always behaves differently.
_CMP_FLIP = {
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}

_BINOP_SWAP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Mult,
}

_BOOLOP_FLIP = {ast.And: ast.Or, ast.Or: ast.And}


@dataclass(frozen=True)
class Mutant:
    """One single-point source mutation of a target module."""

    operator: str
    lineno: int
    detail: str
    source: str

    def describe(self) -> str:
        return f"{self.operator}@L{self.lineno}: {self.detail}"


def _comparison_sites(tree: ast.AST) -> List[Tuple[ast.Compare, int]]:
    sites: List[Tuple[ast.Compare, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in _CMP_FLIP:
                    sites.append((node, i))
    return sites


def _binop_sites(tree: ast.AST) -> List[ast.BinOp]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOP_SWAP
    ]


def _boolop_sites(tree: ast.AST) -> List[ast.BoolOp]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.BoolOp) and type(n.op) in _BOOLOP_FLIP
    ]


def _int_const_sites(tree: ast.AST) -> List[ast.Constant]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, int)
        and not isinstance(n.value, bool)
    ]


def _bool_const_sites(tree: ast.AST) -> List[ast.Constant]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, bool)
    ]


def _gen_comparison(tree: ast.AST):
    n_sites = len(_comparison_sites(tree))
    for idx in range(n_sites):
        new = copy.deepcopy(tree)
        node, opi = _comparison_sites(new)[idx]
        old = type(node.ops[opi])
        node.ops[opi] = _CMP_FLIP[old]()
        yield ("comparison", node, f"{old.__name__}->{_CMP_FLIP[old].__name__}", new)


def _gen_binop(tree: ast.AST):
    n_sites = len(_binop_sites(tree))
    for idx in range(n_sites):
        new = copy.deepcopy(tree)
        node = _binop_sites(new)[idx]
        old = type(node.op)
        node.op = _BINOP_SWAP[old]()
        yield ("arithmetic", node, f"{old.__name__}->{_BINOP_SWAP[old].__name__}", new)


def _gen_boolop(tree: ast.AST):
    n_sites = len(_boolop_sites(tree))
    for idx in range(n_sites):
        new = copy.deepcopy(tree)
        node = _boolop_sites(new)[idx]
        old = type(node.op)
        node.op = _BOOLOP_FLIP[old]()
        yield ("boolean", node, f"{old.__name__}->{_BOOLOP_FLIP[old].__name__}", new)


def _gen_int_const(tree: ast.AST):
    n_sites = len(_int_const_sites(tree))
    for idx in range(n_sites):
        new = copy.deepcopy(tree)
        node = _int_const_sites(new)[idx]
        old = node.value
        bumped = 1 if old == 0 else old + 1
        node.value = bumped
        yield ("constant", node, f"{old}->{bumped}", new)


def _gen_bool_const(tree: ast.AST):
    n_sites = len(_bool_const_sites(tree))
    for idx in range(n_sites):
        new = copy.deepcopy(tree)
        node = _bool_const_sites(new)[idx]
        old = node.value
        node.value = not old
        yield ("bool_constant", node, f"{old}->{not old}", new)


MUTATION_OPERATORS: Dict[str, Callable] = {
    "comparison": _gen_comparison,
    "arithmetic": _gen_binop,
    "boolean": _gen_boolop,
    "constant": _gen_int_const,
    "bool_constant": _gen_bool_const,
}


def generate_mutants(
    source: str,
    operators: Optional[Sequence[str]] = None,
    *,
    func_names: Optional[Sequence[str]] = None,
) -> List[Mutant]:
    """Return every single-point mutant of ``source`` for ``operators``.

    Output order is deterministic (operator order, then site order), so a
    seeded subsample is reproducible.  When ``func_names`` is given, mutation is
    restricted to AST nodes lexically inside one of the named
    ``def``/``async def`` bodies — useful for scoping a high-confidence campaign
    onto a specific, well-tested set of functions (e.g. the lattice core).
    """
    tree = ast.parse(source)
    line_ranges = _func_line_ranges(tree, func_names) if func_names else None
    names = list(operators) if operators is not None else list(MUTATION_OPERATORS)
    out: List[Mutant] = []
    for name in names:
        gen = MUTATION_OPERATORS[name]
        for op_label, node, detail, mutated_tree in gen(tree):
            ln = getattr(node, "lineno", 0)
            if line_ranges is not None and not any(
                lo <= ln <= hi for lo, hi in line_ranges
            ):
                continue
            try:
                mutated_src = ast.unparse(mutated_tree)
            except Exception:  # pragma: no cover - unparse should not fail
                continue
            out.append(
                Mutant(
                    operator=op_label,
                    lineno=ln,
                    detail=detail,
                    source=mutated_src,
                )
            )
    return out


def _func_line_ranges(
    tree: ast.AST, func_names: Sequence[str]
) -> List[Tuple[int, int]]:
    """``(start, end)`` line ranges of every selected ``def``/``async def``.

    Line numbers survive :func:`copy.deepcopy`, so range membership is a stable
    way to scope mutation onto specific functions even though the generators
    yield nodes from deep-copied trees.
    """
    wanted = set(func_names)
    ranges: List[Tuple[int, int]] = []
    for fn in ast.walk(tree):
        if (
            isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and fn.name in wanted
        ):
            start = getattr(fn, "lineno", 0)
            end = getattr(fn, "end_lineno", start)
            ranges.append((start, end))
    return ranges


# --------------------------------------------------------------------------- #
# In-process mutant loading                                                   #
# --------------------------------------------------------------------------- #

def load_mutant_module(qualname: str, source: str) -> types.ModuleType:
    """Exec ``source`` as a fresh module named ``qualname``.

    The new module's ``__package__`` is set so relative imports resolve, via
    ``sys.modules``, to the real unmutated sibling modules.  The module is
    **not** registered in ``sys.modules`` (so global state is untouched) unless
    a caller explicitly does so.
    """
    orig = sys.modules.get(qualname)
    mod = types.ModuleType(qualname)
    pkg = qualname.rpartition(".")[0]
    mod.__dict__["__name__"] = qualname
    mod.__dict__["__package__"] = pkg
    if orig is not None:
        if getattr(orig, "__file__", None):
            mod.__dict__["__file__"] = orig.__file__
        if getattr(orig, "__loader__", None) is not None:
            mod.__dict__["__loader__"] = orig.__loader__
    filename = mod.__dict__.get("__file__", f"<mutant {qualname}>")
    code = compile(source, filename, "exec")
    exec(code, mod.__dict__)
    return mod


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class MutationReport:
    """Outcome of a mutation campaign over one target module."""

    target: str
    total: int = 0
    killed: int = 0
    survivors: List[Mutant] = field(default_factory=list)
    # Mutants that even the *unmutated* loader could not run are excluded from
    # scoring as "invalid" so they neither flatter nor penalise the score.
    invalid: List[Mutant] = field(default_factory=list)

    @property
    def survived(self) -> int:
        return len(self.survivors)

    @property
    def score(self) -> float:
        """Killed fraction of the *scored* (valid) mutants; 1.0 if none."""
        return 1.0 if self.total == 0 else self.killed / self.total

    def summary(self) -> str:
        return (
            f"mutation[{self.target}]: score={self.score:.2f} "
            f"killed={self.killed}/{self.total} survived={self.survived} "
            f"invalid={len(self.invalid)}"
        )

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "total": self.total,
            "killed": self.killed,
            "survived": self.survived,
            "invalid": len(self.invalid),
            "score": round(self.score, 4),
            "survivors": [m.describe() for m in self.survivors],
        }


# --------------------------------------------------------------------------- #
# Campaign driver                                                             #
# --------------------------------------------------------------------------- #

def run_mutation_testing(
    qualname: str,
    oracle: Callable[[types.ModuleType], bool],
    *,
    operators: Optional[Sequence[str]] = None,
    func_names: Optional[Sequence[str]] = None,
    max_mutants: Optional[int] = None,
    seed: int = 0,
    mutants: Optional[Sequence["Mutant"]] = None,
) -> MutationReport:
    """Mutate ``qualname``'s source and score the ``oracle`` against each mutant.

    ``oracle(module) -> bool`` returns ``True`` when the (possibly mutant)
    module *behaves correctly*.  A mutant is **killed** when the oracle returns
    ``False`` or raises; it **survives** when the oracle returns ``True``.

    A deterministic subsample of size ``max_mutants`` (seeded) keeps campaigns
    fast.  Mutants that fail to even load (compile/import errors) are counted as
    killed — an engine that no longer imports is trivially caught.  Callers may
    pass an explicit ``mutants`` list to bypass generation entirely.
    """
    orig = sys.modules[qualname]
    if mutants is None:
        with open(orig.__file__, "r", encoding="utf-8") as fh:
            source = fh.read()
        mutants = generate_mutants(source, operators, func_names=func_names)
        if max_mutants is not None and len(mutants) > max_mutants:
            import random

            rng = random.Random(seed)
            mutants = sorted(
                rng.sample(mutants, max_mutants),
                key=lambda m: (m.operator, m.lineno, m.detail),
            )

    report = MutationReport(target=qualname, total=len(mutants))
    for mutant in mutants:
        try:
            mod = load_mutant_module(qualname, mutant.source)
        except Exception:
            # A mutant that breaks import/compile is detectable => killed.
            report.killed += 1
            continue
        try:
            ok = oracle(mod)
        except Exception:
            report.killed += 1
            continue
        if ok:
            report.survivors.append(mutant)
        else:
            report.killed += 1
    return report


# --------------------------------------------------------------------------- #
# Oracle: value lattice laws (pure target ``values.py``)                      #
# --------------------------------------------------------------------------- #

def _lattice_samples(mod: types.ModuleType) -> list:
    """A rich representative sample set drawn from the *mutant* module.

    Deliberately mirrors ``test_symexec_lattice._samples`` so the oracle
    exercises tensors, tuples and integers — not just the int facet — which is
    what lets it kill mutations in the container/tensor join/meet code too.
    """
    SymDim = mod.SymDim
    return [
        mod.BOTTOM,
        mod.TOP,
        mod.NoneVal(),
        mod.IntVal(sym=SymDim.const_dim(3)),
        mod.IntVal(sym=SymDim.const_dim(4)),
        mod.IntVal(sym=None),
        mod.BoolVal(const=True),
        mod.TensorVal(rank=2),
        mod.TensorVal(rank=3),
        mod.TensorVal(rank=2, shape=(SymDim.const_dim(4), SymDim.const_dim(8))),
        mod.TupleVal(elems=(mod.TensorVal(rank=2), mod.NoneVal()), exact_len=True),
        mod.TupleVal(elems=(mod.TensorVal(rank=2),), exact_len=True),
    ]


def lattice_oracle(mod: types.ModuleType) -> bool:
    """Return ``True`` iff the mutant ``values`` module obeys the lattice laws.

    Mirrors the invariants asserted by ``test_symexec_lattice.py`` over a rich
    sample set: idempotence/commutativity/associativity of join, ⊤/⊥
    absorption, join/meet as least-upper / greatest-lower bound, ``leq``
    reflexivity + consistency with join + antisymmetry, ``widen`` ⊒ join, and
    the documented behavioural facts (rank-disagreement loses rank, tuple
    length-disagreement loses ``exact_len``, incompatible ranks meet to ⊥).
    Any violation kills the mutant.
    """
    join, meet, leq = mod.join, mod.meet, mod.leq
    TOP, BOTTOM = mod.TOP, mod.BOTTOM
    samples = _lattice_samples(mod)

    for a in samples:
        if not leq(a, a):  # reflexivity
            return False
        if join(a, a) != a:  # idempotence
            return False
        if meet(a, a) != a:
            return False
        if join(a, BOTTOM) != a:  # ⊥ identity for join
            return False
        if join(a, TOP) != TOP:  # ⊤ absorbing for join
            return False
        if meet(a, TOP) != a:  # ⊤ identity for meet
            return False
        if meet(a, BOTTOM) != BOTTOM:  # ⊥ absorbing for meet
            return False
        if not leq(BOTTOM, a) or not leq(a, TOP):
            return False
        for b in samples:
            j = join(a, b)
            if j != join(b, a):  # commutativity
                return False
            if not leq(a, j) or not leq(b, j):  # upper bound
                return False
            m = meet(a, b)
            if not leq(m, a) or not leq(m, b):  # lower bound
                return False
            # leq consistent with join.
            if leq(a, b) != (join(a, b) == b):
                return False
            # widen over-approximates join (value lattice: equal).
            if not leq(j, a.widen(b)):
                return False
            for c in samples:
                if join(join(a, b), c) != join(a, join(b, c)):  # associativity
                    return False

    # Behavioural facts the suite pins down.
    Tensor, Tuple_, NoneV = mod.TensorVal, mod.TupleVal, mod.NoneVal
    if join(Tensor(rank=2), Tensor(rank=3)).rank is not None:
        return False
    if meet(Tensor(rank=2), Tensor(rank=3)) != BOTTOM and not meet(
        Tensor(rank=2), Tensor(rank=3)
    ).is_bottom():
        return False
    tj = join(
        Tuple_(elems=(NoneV(), NoneV()), exact_len=True),
        Tuple_(elems=(NoneV(),), exact_len=True),
    )
    if getattr(tj, "exact_len", True) is not False:
        return False
    return True


# --------------------------------------------------------------------------- #
# Oracle: end-to-end corpus via a mutant interpreter                          #
# --------------------------------------------------------------------------- #

def _build_engine_with_interpreter(interp_mod: types.ModuleType):
    """Exec ``engine.py`` against ``interp_mod`` as the interpreter.

    Temporarily substitutes ``sys.modules['src.symexec.interpreter']`` so the
    freshly-built engine's ``from .interpreter import Interpreter`` resolves to
    the supplied (mutant) interpreter, then restores the original.  Returns the
    fresh engine module (not registered in ``sys.modules``).
    """
    pkg = "src.symexec"
    interp_name = f"{pkg}.interpreter"
    engine_name = f"{pkg}.engine"
    engine_orig = sys.modules[engine_name]
    with open(engine_orig.__file__, "r", encoding="utf-8") as fh:
        engine_src = fh.read()

    saved_interp = sys.modules.get(interp_name)
    try:
        sys.modules[interp_name] = interp_mod
        mod = types.ModuleType(engine_name)
        mod.__dict__["__name__"] = engine_name
        mod.__dict__["__package__"] = pkg
        mod.__dict__["__file__"] = engine_orig.__file__
        code = compile(engine_src, engine_orig.__file__, "exec")
        exec(code, mod.__dict__)
        return mod
    finally:
        if saved_interp is not None:
            sys.modules[interp_name] = saved_interp
        else:  # pragma: no cover - interpreter is always pre-imported
            sys.modules.pop(interp_name, None)


def _kinds_of(result) -> Tuple[str, ...]:
    return tuple(sorted(b.kind.name.lower() for b in result.bugs))


def make_corpus_oracle(
    cases: Sequence[Tuple[str, Tuple[str, ...]]],
    *,
    budget_ms: Optional[float] = 4000.0,
) -> Callable[[types.ModuleType], bool]:
    """Build an oracle that runs ``cases`` through a mutant ``interpreter``.

    ``cases`` is a sequence of ``(source, expected_kind_multiset)`` pairs
    (kinds lower-cased & sorted).  The oracle builds a fresh engine bound to the
    candidate interpreter module, analyses each source, and returns ``True``
    iff every case still produces exactly its expected bug kinds.  A mismatch or
    an exception kills the mutant.

    ``budget_ms`` caps each per-file analysis (via the Step-78 resource guard)
    so a mutant whose altered logic causes runaway work is bounded rather than
    hanging the campaign; the early-stopped analysis simply yields a different
    (sound, abstaining) verdict, which the oracle reads as a kill.
    """

    def oracle(interp_mod: types.ModuleType) -> bool:
        engine = _build_engine_with_interpreter(interp_mod)
        for source, expected in cases:
            result = engine.analyze_source(
                source, filename="<mutant-probe>", budget_ms=budget_ms
            )
            if _kinds_of(result) != tuple(expected):
                return False
        return True

    return oracle


# --------------------------------------------------------------------------- #
# Faithful campaign: run the real pytest suite against each mutant            #
# --------------------------------------------------------------------------- #

def run_pytest_mutation_testing(
    qualname: str,
    test_paths: Sequence[str],
    *,
    operators: Optional[Sequence[str]] = None,
    func_names: Optional[Sequence[str]] = None,
    max_mutants: Optional[int] = None,
    seed: int = 0,
    timeout: float = 120.0,
    mutants: Optional[Sequence["Mutant"]] = None,
) -> MutationReport:
    """Score mutants of ``qualname`` against the **real** pytest suite.

    Unlike :func:`run_mutation_testing` (which uses an in-process oracle
    callable), this writes each mutant to the target module's file on disk and
    runs ``python -m pytest <test_paths>`` in an isolated subprocess.  A mutant
    is **killed** iff the suite fails (non-zero exit); it **survives** iff the
    suite still passes.  This is the most faithful measure of whether the
    committed tests constrain the engine, at the cost of a subprocess per
    mutant — so callers should keep ``max_mutants`` small.  An explicit
    ``mutants`` list bypasses generation.

    The original file is always restored (``try/finally``), even on timeout or
    keyboard interrupt.
    """
    import os
    import subprocess

    orig = sys.modules[qualname]
    target_file = orig.__file__
    with open(target_file, "r", encoding="utf-8") as fh:
        original_source = fh.read()

    if mutants is None:
        mutants = generate_mutants(original_source, operators, func_names=func_names)
        if max_mutants is not None and len(mutants) > max_mutants:
            import random

            rng = random.Random(seed)
            mutants = sorted(
                rng.sample(mutants, max_mutants),
                key=lambda m: (m.operator, m.lineno, m.detail),
            )

    repo_root = os.getcwd()
    report = MutationReport(target=qualname, total=len(mutants))
    try:
        for mutant in mutants:
            with open(target_file, "w", encoding="utf-8") as fh:
                fh.write(mutant.source)
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                *test_paths,
                "-q",
                "-x",
                "-p",
                "no:randomly",
                "--no-header",
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    capture_output=True,
                    timeout=timeout,
                )
                killed = proc.returncode != 0
            except subprocess.TimeoutExpired:
                # A mutant that makes the suite hang is detectable => killed.
                killed = True
            if killed:
                report.killed += 1
            else:
                report.survivors.append(mutant)
    finally:
        with open(target_file, "w", encoding="utf-8") as fh:
            fh.write(original_source)
    return report
