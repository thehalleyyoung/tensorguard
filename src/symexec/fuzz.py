"""AST-mutation fuzzing for the symbolic-execution engine (Step 79).

A static analyser is only trustworthy if it is *robust*: an arbitrary, even
nonsensical, but syntactically-valid Python file must never crash it, never make
it hang, and never make it give two different answers for the same input.  This
module is a torch-free, fully-deterministic fuzzer that stresses exactly those
properties by *mutating* a seed corpus at the AST level and re-running the
engine over each mutant.

The mutation operators come in two families:

* **Semantics-changing** (``MutateConstant``, ``MutateOperator``,
  ``DropArguments``, ``TupleUnpackTarget``, ``ShuffleBody``,
  ``DuplicateStatement``, ``DropStatement``) — these reshape the program to walk
  the engine into states the curated corpus never reaches.  They check
  *robustness*: no crash, deterministic, terminates under budget.
* **Semantics-preserving** (``RenameLocal``) — a consistent rename of a local
  name leaves the program's behaviour unchanged, so it is a *metamorphic* oracle
  for soundness: the engine must report the **same multiset of bug kinds** for a
  mutant as for its parent.  A divergence means a report depended on an
  incidental name, i.e. a brittle/unsound detector.

Every mutation is driven by a seeded :class:`random.Random`, so a fuzz run is
reproducible and any failure carries the exact offending source for a one-line
regression repro.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .engine import analyze_source

__all__ = [
    "MutationResult",
    "mutate",
    "rename_local",
    "FuzzCrash",
    "FuzzReport",
    "fuzz",
]

# A generous per-mutant analysis budget: the engine's iteration caps already
# bound cost deterministically, so any mutant that blows past this is a genuine
# performance/termination bug worth surfacing.
_FUZZ_BUDGET_MS = 5000.0

_BINOPS = [ast.Add, ast.Sub, ast.Mult, ast.MatMult, ast.Div, ast.Mod, ast.Pow,
           ast.FloorDiv]
_CMPOPS = [ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE]
_INT_POOL = [-1, 0, 1, 2, 3, 7, 16, 99]


@dataclass
class MutationResult:
    """The outcome of applying one mutation pass to a source string."""

    source: str
    changed: bool
    semantics_preserving: bool
    operator: str


# -- individual operators (each is a seeded NodeTransformer) ------------


class _ConstantMutator(ast.NodeTransformer):
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.changed = False

    def visit_Constant(self, node: ast.Constant):
        v = node.value
        if isinstance(v, bool):
            if self.rng.random() < 0.5:
                self.changed = True
                return ast.copy_location(ast.Constant(value=not v), node)
        elif isinstance(v, int):
            if self.rng.random() < 0.5:
                self.changed = True
                return ast.copy_location(
                    ast.Constant(value=self.rng.choice(_INT_POOL)), node
                )
        elif isinstance(v, float):
            if self.rng.random() < 0.5:
                self.changed = True
                return ast.copy_location(
                    ast.Constant(value=float(self.rng.choice(_INT_POOL))), node
                )
        return node


class _OperatorMutator(ast.NodeTransformer):
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.changed = False

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if self.rng.random() < 0.5:
            self.changed = True
            node.op = self.rng.choice(_BINOPS)()
        return node

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if node.ops and self.rng.random() < 0.5:
            self.changed = True
            node.ops = [self.rng.choice(_CMPOPS)() for _ in node.ops]
        return node


class _ArgDropper(ast.NodeTransformer):
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        pos = node.args.args
        # Keep ``self`` if present so methods stay structurally valid.
        head = pos[:1] if (pos and pos[0].arg == "self") else []
        tail = pos[len(head):]
        if tail and self.rng.random() < 0.6:
            keep = self.rng.randint(0, len(tail))
            node.args.args = head + tail[:keep]
            node.args.defaults = []
            self.changed = True
        return node


class _TupleUnpacker(ast.NodeTransformer):
    """Turn ``x = e`` into ``x_0, x_1 = e`` — directly stresses the engine's
    tuple-unpack arity detector (the titans-pytorch #60 class)."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.changed = False

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and self.rng.random() < 0.4
        ):
            base = node.targets[0].id
            n = self.rng.randint(2, 3)
            node.targets = [
                ast.Tuple(
                    elts=[ast.Name(id=f"{base}_{i}", ctx=ast.Store()) for i in range(n)],
                    ctx=ast.Store(),
                )
            ]
            self.changed = True
        return node


class _BodyShuffler(ast.NodeTransformer):
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.changed = False

    def _mutate_body(self, body: List[ast.stmt]) -> List[ast.stmt]:
        if len(body) <= 1:
            return body
        out = list(body)
        roll = self.rng.random()
        if roll < 0.34:
            self.rng.shuffle(out)
            self.changed = True
        elif roll < 0.67:
            i = self.rng.randrange(len(out))
            out.insert(i, out[i])  # duplicate
            self.changed = True
        else:
            del out[self.rng.randrange(len(out))]  # drop
            self.changed = True
        return out or [ast.Pass()]

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        node.body = self._mutate_body(node.body)
        return node


def _rename_in_function(func: ast.FunctionDef, rng: random.Random) -> bool:
    """Consistently rename one *purely-local* assigned name within ``func``,
    avoiding collisions.  Returns whether a rename happened.

    Strictly semantics-preserving: only names that are **assigned locally and are
    not parameters** are eligible.  Parameters are excluded because a callee's
    parameter name is part of its interface — callers may bind it by keyword
    (``f(param=…)``), so renaming it changes behaviour (a subtlety the fuzzer's
    own metamorphic oracle uncovered).  Names declared ``global``/``nonlocal`` are
    excluded too, since they reach outside the local scope.  Attributes, builtins
    and keyword-argument labels are never touched."""
    params = {a.arg for a in list(func.args.args) + list(func.args.kwonlyargs)}
    if func.args.vararg:
        params.add(func.args.vararg.arg)
    if func.args.kwarg:
        params.add(func.args.kwarg.arg)
    escaped: set = set()
    assigned: set = set()
    for n in ast.walk(func):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            escaped.update(n.names)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            assigned.add(n.id)
    candidates = assigned - params - escaped - {"self"}
    if not candidates:
        return False
    used = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}
    used |= {a.arg for a in ast.walk(func) if isinstance(a, ast.arg)}
    old = rng.choice(sorted(candidates))
    new = f"{old}_r"
    while new in used:
        new += "_"
    for n in ast.walk(func):
        if isinstance(n, ast.Name) and n.id == old:
            n.id = new
    return True


def rename_local(source: str, seed: int = 0) -> Optional[str]:
    """Return ``source`` with one local name consistently renamed (a
    semantics-preserving metamorphic mutation), or ``None`` if nothing was
    renamable / the result is unparseable."""
    rng = random.Random(seed)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return None
    rng.shuffle(funcs)
    changed = False
    for f in funcs:
        if _rename_in_function(f, rng):
            changed = True
            break
    if not changed:
        return None
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return None


_OPERATORS: List[Tuple[str, type, bool]] = [
    ("mutate_constant", _ConstantMutator, False),
    ("mutate_operator", _OperatorMutator, False),
    ("drop_arguments", _ArgDropper, False),
    ("tuple_unpack", _TupleUnpacker, False),
    ("shuffle_body", _BodyShuffler, False),
]


def mutate(source: str, seed: int = 0) -> MutationResult:
    """Apply one randomly-chosen mutation operator to ``source``.

    Deterministic in ``seed``.  When the chosen operator does not fire (or the
    result fails to round-trip through :func:`ast.unparse`) the original source
    is returned with ``changed=False`` so callers can skip no-op mutants."""
    rng = random.Random(seed)
    # A semantics-preserving rename is one of the operators, drawn with the rest.
    if rng.random() < 0.25:
        renamed = rename_local(source, seed=seed ^ 0x5EED)
        if renamed is not None and renamed != source:
            return MutationResult(renamed, True, True, "rename_local")
        # fall through to a structural operator if nothing was renamable
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return MutationResult(source, False, True, "parse_error")
    name, cls, preserving = rng.choice(_OPERATORS)
    mutator = cls(rng)
    new_tree = mutator.visit(tree)
    ast.fix_missing_locations(new_tree)
    if not getattr(mutator, "changed", False):
        return MutationResult(source, False, True, name)
    try:
        new_source = ast.unparse(new_tree)
    except Exception:
        return MutationResult(source, False, True, name)
    return MutationResult(new_source, new_source != source, preserving, name)


# -- fuzz driver --------------------------------------------------------


@dataclass(frozen=True)
class FuzzCrash:
    """A reproducible robustness failure found during fuzzing."""

    kind: str  # "crash" | "nondeterministic" | "metamorphic"
    operator: str
    seed: int
    detail: str
    source: str


@dataclass
class FuzzReport:
    """The aggregate result of a fuzz campaign."""

    runs: int = 0
    mutants: int = 0
    crashes: List[FuzzCrash] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.crashes

    def summary(self) -> str:
        return (
            f"fuzz: {self.runs} runs, {self.mutants} effective mutants, "
            f"{len(self.crashes)} failure(s)"
        )


def fuzz(
    seeds: Sequence[str],
    iterations: int = 50,
    base_seed: int = 0,
    budget_ms: Optional[float] = _FUZZ_BUDGET_MS,
) -> FuzzReport:
    """Run an AST-mutation fuzz campaign over ``seeds`` and return a report.

    For every mutant the driver asserts three robustness properties and one
    soundness property:

    * **no crash** — :func:`analyze_source` returns a result instead of raising;
    * **determinism** — two analyses of the same mutant share a fingerprint;
    * **termination** — the analysis stays within ``budget_ms`` (enforced via the
      engine's coarse budget guard, which is sound);
    * **metamorphic soundness** — a semantics-preserving mutant reports the same
      multiset of bug kinds as its parent.

    Any violation is recorded as a :class:`FuzzCrash` (with the offending source
    for a one-line repro) rather than raised, so a campaign always completes."""
    report = FuzzReport()
    for s_idx, seed_src in enumerate(seeds):
        try:
            parent = analyze_source(seed_src, budget_ms=budget_ms)
        except Exception as exc:
            report.crashes.append(
                FuzzCrash("crash", "seed", base_seed, f"{type(exc).__name__}: {exc}", seed_src)
            )
            continue
        parent_kinds = sorted(b.kind.name for b in parent.bugs)
        for it in range(iterations):
            report.runs += 1
            seed = base_seed + s_idx * 100003 + it
            res = mutate(seed_src, seed=seed)
            if not res.changed:
                continue
            report.mutants += 1
            # 1) no crash
            try:
                r1 = analyze_source(res.source, budget_ms=budget_ms)
            except Exception as exc:
                report.crashes.append(
                    FuzzCrash("crash", res.operator, seed,
                              f"{type(exc).__name__}: {exc}", res.source)
                )
                continue
            # 2) determinism
            try:
                r2 = analyze_source(res.source, budget_ms=budget_ms)
            except Exception as exc:
                report.crashes.append(
                    FuzzCrash("crash", res.operator, seed,
                              f"re-run {type(exc).__name__}: {exc}", res.source)
                )
                continue
            if r1.fingerprint() != r2.fingerprint():
                report.crashes.append(
                    FuzzCrash("nondeterministic", res.operator, seed,
                              f"{r1.fingerprint()} != {r2.fingerprint()}", res.source)
                )
                continue
            # 3) metamorphic soundness for semantics-preserving mutants
            if res.semantics_preserving:
                got = sorted(b.kind.name for b in r1.bugs)
                if got != parent_kinds:
                    report.crashes.append(
                        FuzzCrash("metamorphic", res.operator, seed,
                                  f"bug kinds changed {parent_kinds} -> {got}",
                                  res.source)
                    )
    return report
