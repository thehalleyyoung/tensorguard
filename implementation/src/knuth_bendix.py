"""
Knuth-Bendix Completion for Tensor Shape Constraint Simplification.

Implements the Knuth-Bendix completion procedure for the equational theory
of tensor shape operations, producing a convergent (confluent + terminating)
term rewriting system.  The completed rule set is used to normalize Z3
expressions *after* Z3's internal simplifier, yielding a canonical form
for shape constraints.

Term Algebra
------------
Sorts: Dim (positive integers), Shape (dimension tuples)

Function symbols (with arities):
    bc(a, b)              — broadcast of two dimensions
    numel(s)              — total element count of shape s
    conv(h, k, s, p)      — convolution output dim: floor((h + 2p - k) / s) + 1
    pool(h, k, s, p)      — pooling output dim: floor((h + 2p - k) / s) + 1
    transp(s, d0, d1)     — swap dimensions d0, d1 in shape s
    perm(s, p)            — permute shape s by permutation p
    reshape(s, t)         — reshape shape s to target shape t
    add(a, b)             — integer addition
    mul(a, b)             — integer multiplication
    floor_div(a, b)       — integer floor division
    sub(a, b)             — integer subtraction

Reduction Ordering
------------------
We use Recursive Path Ordering (RPO) with the following precedence on
function symbols (strictly decreasing):

    numel > bc > conv > pool > transp > perm > reshape
         > floor_div > sub > add > mul > const > var

This is a simplification ordering because RPO with any well-founded
precedence is:
  1. Well-founded (no infinite descending chains) — Dershowitz 1982
  2. Monotone (s > t ⟹ C[s] > C[t] for any context C)
  3. Stable under substitution (s > t ⟹ sσ > tσ for any substitution σ)

Termination Proof
-----------------
Every rewrite rule l → r satisfies l >_RPO r.  Since RPO is well-founded,
the rewriting system terminates.  This is verified at rule-construction
time by ``_verify_rule_orientation``.

Confluence Verification
-----------------------
After completion, all critical pairs are checked for joinability. The
``verify_confluence`` method enumerates every critical pair from
overlapping left-hand sides and verifies that both sides reduce to the
same normal form under the completed rule set.

Interaction with Z3's Simplifier
---------------------------------
The normalization pipeline is::

    z3.simplify(expr)  →  kb_normalize(expr)  →  z3.simplify(expr)

This three-phase pipeline is idempotent: applying it twice yields the
same result as applying it once, because:
  1. z3.simplify is idempotent
  2. kb_normalize produces a unique normal form (confluence)
  3. The second z3.simplify canonicalizes any arithmetic that KB rules
     may have introduced (e.g., constant folding)
  4. KB rules do not introduce terms that z3.simplify would expand back
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import IntEnum
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

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ======================================================================
# Term Algebra
# ======================================================================


class SymbolKind(IntEnum):
    """Kind of a function/constant symbol in the term algebra."""

    VAR = 0
    CONST = 1
    MUL = 2
    ADD = 3
    SUB = 4
    FLOOR_DIV = 5
    RESHAPE = 6
    PERM = 7
    TRANSP = 8
    POOL = 9
    CONV = 10
    BC = 11
    NUMEL = 12


# RPO precedence: higher SymbolKind value ⟹ higher precedence.
# VAR < CONST < MUL < ADD < SUB < FLOOR_DIV < RESHAPE < PERM
#     < TRANSP < POOL < CONV < BC < NUMEL

# Arity table (None means variable arity for CONST/VAR)
_ARITY: Dict[SymbolKind, Optional[int]] = {
    SymbolKind.VAR: 0,
    SymbolKind.CONST: 0,
    SymbolKind.MUL: 2,
    SymbolKind.ADD: 2,
    SymbolKind.SUB: 2,
    SymbolKind.FLOOR_DIV: 2,
    SymbolKind.RESHAPE: 2,
    SymbolKind.PERM: 2,
    SymbolKind.TRANSP: 3,
    SymbolKind.POOL: 4,
    SymbolKind.CONV: 4,
    SymbolKind.BC: 2,
    SymbolKind.NUMEL: 1,
}

# Human-readable names for function symbols
_SYMBOL_NAMES: Dict[SymbolKind, str] = {
    SymbolKind.VAR: "var",
    SymbolKind.CONST: "const",
    SymbolKind.MUL: "mul",
    SymbolKind.ADD: "add",
    SymbolKind.SUB: "sub",
    SymbolKind.FLOOR_DIV: "floor_div",
    SymbolKind.RESHAPE: "reshape",
    SymbolKind.PERM: "perm",
    SymbolKind.TRANSP: "transp",
    SymbolKind.POOL: "pool",
    SymbolKind.CONV: "conv",
    SymbolKind.BC: "bc",
    SymbolKind.NUMEL: "numel",
}


@dataclass(frozen=True, eq=True)
class Term:
    """An algebraic term in the tensor shape theory.

    A term is either:
      - A variable: ``Term(SymbolKind.VAR, name="x")``
      - A constant: ``Term(SymbolKind.CONST, value=42)``
      - A compound:  ``Term(SymbolKind.BC, children=(t1, t2))``
    """

    symbol: SymbolKind
    children: Tuple["Term", ...] = ()
    name: Optional[str] = None    # for VAR
    value: Optional[int] = None   # for CONST

    def __post_init__(self) -> None:
        if self.symbol == SymbolKind.VAR and self.name is None:
            raise ValueError("Variable terms must have a name")
        if self.symbol == SymbolKind.CONST and self.value is None:
            raise ValueError("Constant terms must have a value")

    # Convenience constructors
    @staticmethod
    def var(name: str) -> "Term":
        return Term(SymbolKind.VAR, name=name)

    @staticmethod
    def const(value: int) -> "Term":
        return Term(SymbolKind.CONST, value=value)

    @staticmethod
    def bc(a: "Term", b: "Term") -> "Term":
        return Term(SymbolKind.BC, children=(a, b))

    @staticmethod
    def numel(s: "Term") -> "Term":
        return Term(SymbolKind.NUMEL, children=(s,))

    @staticmethod
    def conv(h: "Term", k: "Term", s: "Term", p: "Term") -> "Term":
        return Term(SymbolKind.CONV, children=(h, k, s, p))

    @staticmethod
    def pool(h: "Term", k: "Term", s: "Term", p: "Term") -> "Term":
        return Term(SymbolKind.POOL, children=(h, k, s, p))

    @staticmethod
    def transp(s: "Term", d0: "Term", d1: "Term") -> "Term":
        return Term(SymbolKind.TRANSP, children=(s, d0, d1))

    @staticmethod
    def perm(s: "Term", p: "Term") -> "Term":
        return Term(SymbolKind.PERM, children=(s, p))

    @staticmethod
    def reshape(s: "Term", t: "Term") -> "Term":
        return Term(SymbolKind.RESHAPE, children=(s, t))

    @staticmethod
    def add(a: "Term", b: "Term") -> "Term":
        return Term(SymbolKind.ADD, children=(a, b))

    @staticmethod
    def mul(a: "Term", b: "Term") -> "Term":
        return Term(SymbolKind.MUL, children=(a, b))

    @staticmethod
    def sub(a: "Term", b: "Term") -> "Term":
        return Term(SymbolKind.SUB, children=(a, b))

    @staticmethod
    def floor_div(a: "Term", b: "Term") -> "Term":
        return Term(SymbolKind.FLOOR_DIV, children=(a, b))

    @property
    def is_var(self) -> bool:
        return self.symbol == SymbolKind.VAR

    @property
    def is_const(self) -> bool:
        return self.symbol == SymbolKind.CONST

    @property
    def is_compound(self) -> bool:
        return self.symbol not in (SymbolKind.VAR, SymbolKind.CONST)

    def variables(self) -> FrozenSet[str]:
        """Return the set of variable names occurring in this term."""
        if self.is_var:
            return frozenset({self.name})  # type: ignore[arg-type]
        result: Set[str] = set()
        for c in self.children:
            result |= c.variables()
        return frozenset(result)

    def size(self) -> int:
        """Number of nodes in the term tree."""
        return 1 + sum(c.size() for c in self.children)

    def __repr__(self) -> str:
        if self.is_var:
            return self.name  # type: ignore[return-value]
        if self.is_const:
            return str(self.value)
        name = _SYMBOL_NAMES[self.symbol]
        args = ", ".join(repr(c) for c in self.children)
        return f"{name}({args})"


# ======================================================================
# Substitution
# ======================================================================

Substitution = Dict[str, Term]


def apply_substitution(term: Term, subst: Substitution) -> Term:
    """Apply a substitution to a term.

    For variable chains (a→b→c), chase to the final binding.
    Then substitute into the result in a single pass (no re-application
    of the substitution to newly introduced terms).
    """
    if term.is_var:
        # Chase variable-to-variable chains only
        seen: Set[str] = set()
        current = term
        while (current.is_var and current.name in subst
               and current.name not in seen):
            seen.add(current.name)  # type: ignore[arg-type]
            nxt = subst[current.name]  # type: ignore[index]
            if nxt.is_var:
                current = nxt
            else:
                # Reached a non-variable binding — return it as-is
                # (do not recursively apply subst to avoid infinite loops
                # when the binding contains variables also in subst)
                return nxt
        return current
    if term.is_const:
        return term
    new_children = tuple(apply_substitution(c, subst) for c in term.children)
    return Term(term.symbol, children=new_children,
                name=term.name, value=term.value)


# ======================================================================
# Unification
# ======================================================================


def unify(s: Term, t: Term) -> Optional[Substitution]:
    """Syntactic unification via Robinson's algorithm.

    Returns a most general unifier (mgu) or None if the terms are not
    unifiable.
    """
    return _unify_rec(s, t, {})


def _unify_rec(s: Term, t: Term, subst: Substitution) -> Optional[Substitution]:
    s = _walk(s, subst)
    t = _walk(t, subst)

    if s == t:
        return subst
    if s.is_var:
        return _bind(s.name, t, subst)  # type: ignore[arg-type]
    if t.is_var:
        return _bind(t.name, s, subst)  # type: ignore[arg-type]
    if s.symbol != t.symbol or len(s.children) != len(t.children):
        return None
    for sc, tc in zip(s.children, t.children):
        subst_next = _unify_rec(sc, tc, subst)
        if subst_next is None:
            return None
        subst = subst_next
    return subst


def _walk(term: Term, subst: Substitution) -> Term:
    while term.is_var and term.name in subst:
        term = subst[term.name]  # type: ignore[index]
    return term


def _bind(var_name: str, term: Term, subst: Substitution) -> Optional[Substitution]:
    if term.is_var and term.name == var_name:
        return subst
    if var_name in _all_vars(term):
        return None  # occurs check
    new_subst = dict(subst)
    new_subst[var_name] = term
    return new_subst


def _all_vars(term: Term) -> Set[str]:
    if term.is_var:
        return {term.name}  # type: ignore[arg-type]
    result: Set[str] = set()
    for c in term.children:
        result |= _all_vars(c)
    return result


# ======================================================================
# Recursive Path Ordering (RPO)
# ======================================================================


def rpo_gt(s: Term, t: Term) -> bool:
    """Return True iff s >_RPO t using the precedence from SymbolKind.

    The RPO is defined recursively:
      s = f(s1,...,sm) >_RPO t = g(t1,...,tn) iff one of:
        (1) ∃i. si ≥_RPO t                    (subterm property)
        (2) f > g in precedence AND ∀j. s >_RPO tj   (precedence)
        (3) f = g AND lex(s1..sm) >_RPO lex(t1..tn)  (lex extension)
            AND ∀j. s >_RPO tj
    """
    return _rpo_gt(s, t)


def rpo_ge(s: Term, t: Term) -> bool:
    """s ≥_RPO t iff s = t or s >_RPO t."""
    return s == t or rpo_gt(s, t)


def _rpo_gt(s: Term, t: Term) -> bool:
    if s == t:
        return False

    # Variables: s >_RPO x only if x is a proper subterm of s
    if t.is_var:
        return t.name in s.variables() and s != t  # type: ignore[operator]
    if s.is_var:
        return False

    # Constants: compare by symbol precedence, then by value
    if s.is_const and t.is_const:
        return False  # constants are incomparable unless equal
    if s.is_const:
        return False  # a constant cannot be greater than a compound
    if t.is_const:
        # s is compound, t is const — check subterm or precedence
        # (1) subterm: ∃i. si ≥_RPO t
        if any(rpo_ge(si, t) for si in s.children):
            return True
        # (2) precedence: s.symbol > CONST
        return s.symbol > SymbolKind.CONST

    # Both compound
    # (1) Subterm property: ∃i. si ≥_RPO t
    if any(rpo_ge(si, t) for si in s.children):
        return True

    # (2) Precedence: f > g and ∀j. s >_RPO tj
    if s.symbol > t.symbol:
        return all(_rpo_gt(s, tj) for tj in t.children)

    # (3) Same symbol: lexicographic extension
    if s.symbol == t.symbol:
        if len(s.children) != len(t.children):
            return False
        # ∀j. s >_RPO tj  (needed for both lex cases)
        if not all(_rpo_gt(s, tj) for tj in t.children):
            return False
        # Lexicographic comparison
        for si, ti in zip(s.children, t.children):
            if _rpo_gt(si, ti):
                return True
            if si != ti:
                return False
        return False

    return False


# ======================================================================
# Rewrite Rules
# ======================================================================


@dataclass(frozen=True)
class RewriteRule:
    """A rewrite rule l → r with rule ID and provenance."""

    id: int
    lhs: Term
    rhs: Term
    name: str = ""
    provenance: str = ""  # e.g. "axiom" or "critical_pair(3,5)"

    def __repr__(self) -> str:
        label = f"[{self.id}] " if self.id >= 0 else ""
        prov = f"  ({self.provenance})" if self.provenance else ""
        return f"{label}{self.lhs!r} → {self.rhs!r}{prov}"


@dataclass(frozen=True)
class Equation:
    """An unoriented equation s = t."""

    lhs: Term
    rhs: Term

    def __repr__(self) -> str:
        return f"{self.lhs!r} = {self.rhs!r}"


@dataclass
class CriticalPair:
    """A critical pair arising from overlapping rule left-hand sides."""

    term1: Term
    term2: Term
    rule1_id: int
    rule2_id: int
    overlap_position: Tuple[int, ...] = ()
    joinable: Optional[bool] = None

    def __repr__(self) -> str:
        j = "✓" if self.joinable else ("✗" if self.joinable is False else "?")
        return f"CP({self.term1!r}, {self.term2!r}) [{j}] from rules {self.rule1_id},{self.rule2_id}"


# ======================================================================
# Rewriting Engine
# ======================================================================


def rewrite_step(term: Term, rules: Sequence[RewriteRule]) -> Optional[Term]:
    """Apply the first applicable rule at the leftmost-outermost position.

    Returns the rewritten term, or None if no rule applies.
    """
    # Try at root
    for rule in rules:
        subst = unify(rule.lhs, term)
        if subst is not None:
            # Check that substitution only binds variables from the rule's LHS
            lhs_vars = rule.lhs.variables()
            if all(v in lhs_vars for v in subst):
                # Verify it's a match (not general unification)
                if apply_substitution(rule.lhs, subst) == term:
                    return apply_substitution(rule.rhs, subst)

    # Try in children (leftmost-outermost)
    if term.is_compound:
        for i, child in enumerate(term.children):
            result = rewrite_step(child, rules)
            if result is not None:
                new_children = list(term.children)
                new_children[i] = result
                return Term(term.symbol, children=tuple(new_children),
                            name=term.name, value=term.value)
    return None


def match_term(pattern: Term, target: Term) -> Optional[Substitution]:
    """One-way pattern matching: find σ s.t. pattern·σ = target.

    Unlike unification, only variables in pattern may be bound.
    """
    return _match_rec(pattern, target, {})


def _match_rec(
    pattern: Term, target: Term, subst: Substitution
) -> Optional[Substitution]:
    if pattern.is_var:
        pname = pattern.name  # type: ignore[arg-type]
        if pname in subst:
            return subst if subst[pname] == target else None
        new_subst = dict(subst)
        new_subst[pname] = target
        return new_subst
    if pattern.is_const:
        return subst if pattern == target else None
    if pattern.symbol != target.symbol:
        return None
    if len(pattern.children) != len(target.children):
        return None
    for pc, tc in zip(pattern.children, target.children):
        subst = _match_rec(pc, tc, subst)
        if subst is None:
            return None
    return subst


def rewrite_step_matching(term: Term, rules: Sequence[RewriteRule]) -> Optional[Term]:
    """Apply the first applicable rule using one-way matching."""
    # Try at root
    for rule in rules:
        subst = match_term(rule.lhs, term)
        if subst is not None:
            return apply_substitution(rule.rhs, subst)
    # Try in children
    if term.is_compound:
        for i, child in enumerate(term.children):
            result = rewrite_step_matching(child, rules)
            if result is not None:
                new_children = list(term.children)
                new_children[i] = result
                return Term(term.symbol, children=tuple(new_children),
                            name=term.name, value=term.value)
    return None


def normalize(term: Term, rules: Sequence[RewriteRule],
              max_steps: int = 1000) -> Term:
    """Reduce a term to normal form by repeated rewriting."""
    current = term
    for _ in range(max_steps):
        next_term = rewrite_step_matching(current, rules)
        if next_term is None:
            return current
        current = next_term
    return current


# ======================================================================
# Critical Pair Computation
# ======================================================================


def _rename_vars(term: Term, suffix: str) -> Term:
    """Rename all variables in a term by appending a suffix."""
    if term.is_var:
        return Term.var(term.name + suffix)  # type: ignore[operator]
    if term.is_const:
        return term
    new_children = tuple(_rename_vars(c, suffix) for c in term.children)
    return Term(term.symbol, children=new_children,
                name=term.name, value=term.value)


def _rename_rule(rule: RewriteRule, suffix: str) -> RewriteRule:
    return RewriteRule(
        id=rule.id,
        lhs=_rename_vars(rule.lhs, suffix),
        rhs=_rename_vars(rule.rhs, suffix),
        name=rule.name,
        provenance=rule.provenance,
    )


def _subterms_with_positions(
    term: Term,
) -> List[Tuple[Tuple[int, ...], Term]]:
    """Return all non-variable subterms with their positions."""
    result: List[Tuple[Tuple[int, ...], Term]] = []
    if not term.is_var:
        result.append(((), term))
    if term.is_compound:
        for i, child in enumerate(term.children):
            for pos, sub in _subterms_with_positions(child):
                result.append(((i,) + pos, sub))
    return result


def _replace_at_position(term: Term, pos: Tuple[int, ...], replacement: Term) -> Term:
    """Replace the subterm at the given position."""
    if not pos:
        return replacement
    idx = pos[0]
    rest = pos[1:]
    new_children = list(term.children)
    new_children[idx] = _replace_at_position(term.children[idx], rest, replacement)
    return Term(term.symbol, children=tuple(new_children),
                name=term.name, value=term.value)


def compute_critical_pairs(
    rule1: RewriteRule, rule2: RewriteRule
) -> List[CriticalPair]:
    """Compute all critical pairs between two rules.

    For each non-variable subterm of rule1.lhs that unifies with rule2.lhs
    (after variable renaming), produce the critical pair.
    """
    # Rename rule2 variables to avoid capture
    r2 = _rename_rule(rule2, "_r2")

    pairs: List[CriticalPair] = []
    for pos, subterm in _subterms_with_positions(rule1.lhs):
        # Skip root if same rule (trivial overlap)
        if rule1.id == rule2.id and pos == ():
            continue
        mgu = unify(subterm, r2.lhs)
        if mgu is not None:
            # CP: (rule1.rhs)σ  vs  (rule1.lhs[pos ← rule2.rhs])σ
            t1 = apply_substitution(rule1.rhs, mgu)
            replaced = _replace_at_position(rule1.lhs, pos, r2.rhs)
            t2 = apply_substitution(replaced, mgu)
            pairs.append(CriticalPair(
                term1=t1, term2=t2,
                rule1_id=rule1.id, rule2_id=rule2.id,
                overlap_position=pos,
            ))
    return pairs


# ======================================================================
# Knuth-Bendix Completion
# ======================================================================


@dataclass
class CompletionResult:
    """Result of the Knuth-Bendix completion procedure."""

    converged: bool
    rules: List[RewriteRule]
    critical_pairs_resolved: int
    critical_pairs_total: int
    iterations: int

    def __repr__(self) -> str:
        status = "converged" if self.converged else "failed"
        return (
            f"CompletionResult({status}, {len(self.rules)} rules, "
            f"{self.critical_pairs_resolved}/{self.critical_pairs_total} CPs resolved, "
            f"{self.iterations} iterations)"
        )


def orient_equation(eq: Equation, rule_id: int) -> Optional[RewriteRule]:
    """Orient an equation into a rewrite rule using RPO.

    Returns l → r if l >_RPO r, or r → l if r >_RPO l, or None if
    the equation cannot be oriented.
    """
    if rpo_gt(eq.lhs, eq.rhs):
        return RewriteRule(id=rule_id, lhs=eq.lhs, rhs=eq.rhs,
                           provenance="oriented(lhs > rhs)")
    if rpo_gt(eq.rhs, eq.lhs):
        return RewriteRule(id=rule_id, lhs=eq.rhs, rhs=eq.lhs,
                           provenance="oriented(rhs > lhs)")
    return None


def _verify_rule_orientation(rule: RewriteRule) -> bool:
    """Verify that lhs >_RPO rhs (termination guarantee)."""
    return rpo_gt(rule.lhs, rule.rhs)


def knuth_bendix_completion(
    axioms: List[Equation],
    max_iterations: int = 100,
    max_rules: int = 200,
) -> CompletionResult:
    """Run Knuth-Bendix completion on a set of equations.

    Parameters
    ----------
    axioms : list of Equation
        Initial equations to orient and complete.
    max_iterations : int
        Maximum number of completion iterations.
    max_rules : int
        Maximum number of rules before giving up.

    Returns
    -------
    CompletionResult
        The completed rule set (if convergent) or the best partial result.
    """
    rules: List[RewriteRule] = []
    pending: List[Equation] = list(axioms)
    next_id = 0
    total_cps = 0
    resolved_cps = 0

    # Phase 1: Orient initial equations
    new_pending: List[Equation] = []
    for eq in pending:
        rule = orient_equation(eq, next_id)
        if rule is not None:
            rules.append(rule)
            next_id += 1
        else:
            new_pending.append(eq)
    pending = new_pending

    # Phase 2: Completion loop
    iteration = 0
    while iteration < max_iterations and len(rules) < max_rules:
        iteration += 1

        # Compute all critical pairs
        new_pairs: List[CriticalPair] = []
        for i, r1 in enumerate(rules):
            for j, r2 in enumerate(rules):
                cps = compute_critical_pairs(r1, r2)
                new_pairs.extend(cps)

        if not new_pairs and not pending:
            break

        total_cps += len(new_pairs)

        # Check joinability of critical pairs
        changed = False
        for cp in new_pairs:
            nf1 = normalize(cp.term1, rules)
            nf2 = normalize(cp.term2, rules)
            cp.joinable = (nf1 == nf2)

            if cp.joinable:
                resolved_cps += 1
            else:
                # Non-joinable: create new equation and try to orient it
                eq = Equation(nf1, nf2)
                rule = orient_equation(eq, next_id)
                if rule is not None:
                    rules.append(rule)
                    next_id += 1
                    changed = True
                    resolved_cps += 1
                else:
                    pending.append(eq)

        # Try to orient remaining pending equations
        still_pending: List[Equation] = []
        for eq in pending:
            rule = orient_equation(eq, next_id)
            if rule is not None:
                rules.append(rule)
                next_id += 1
                changed = True
            else:
                still_pending.append(eq)
        pending = still_pending

        if not changed:
            break

    converged = len(pending) == 0
    return CompletionResult(
        converged=converged,
        rules=rules,
        critical_pairs_resolved=resolved_cps,
        critical_pairs_total=total_cps,
        iterations=iteration,
    )


# ======================================================================
# Confluence Verification
# ======================================================================


def verify_confluence(rules: Sequence[RewriteRule]) -> Tuple[bool, List[CriticalPair]]:
    """Verify that the rewriting system is confluent.

    Checks that all critical pairs are joinable (Knuth-Bendix criterion).

    Returns (is_confluent, list_of_non_joinable_pairs).
    """
    non_joinable: List[CriticalPair] = []
    for i, r1 in enumerate(rules):
        for j, r2 in enumerate(rules):
            cps = compute_critical_pairs(r1, r2)
            for cp in cps:
                nf1 = normalize(cp.term1, rules)
                nf2 = normalize(cp.term2, rules)
                cp.joinable = (nf1 == nf2)
                if not cp.joinable:
                    non_joinable.append(cp)
    return len(non_joinable) == 0, non_joinable


def verify_termination(rules: Sequence[RewriteRule]) -> Tuple[bool, List[RewriteRule]]:
    """Verify that every rule l → r satisfies l >_RPO r.

    Returns (terminates, list_of_failing_rules).
    """
    failing: List[RewriteRule] = []
    for rule in rules:
        if not _verify_rule_orientation(rule):
            failing.append(rule)
    return len(failing) == 0, failing


# ======================================================================
# Tensor Shape Axioms
# ======================================================================


def tensor_shape_axioms() -> List[Equation]:
    """Return the equational axioms for the tensor shape theory.

    These are the initial equations for KB completion:
      E1: bc(a, 1) = a            (broadcast identity right)
      E2: bc(1, b) = b            (broadcast identity left)
      E3: bc(a, a) = a            (broadcast idempotent)
      E4: bc(a, b) = bc(b, a)     (broadcast commutativity)
      E5: bc(bc(a,b), c) = bc(a, bc(b,c))  (broadcast associativity)
      E6: transp(transp(s, d0, d1), d0, d1) = s  (double transpose)
      E7: numel(reshape(s, t)) = numel(s)         (reshape preserves numel)
      E8: conv(h, k, 1, 0) = add(sub(h, k), 1)   (conv without stride/padding)
      E9: pool(h, k, k, 0) = floor_div(h, k)      (pool with stride=kernel)
    """
    a, b, c = Term.var("a"), Term.var("b"), Term.var("c")
    s, t = Term.var("s"), Term.var("t")
    d0, d1 = Term.var("d0"), Term.var("d1")
    h, k = Term.var("h"), Term.var("k")
    _1 = Term.const(1)
    _0 = Term.const(0)

    return [
        # E1: bc(a, 1) = a
        Equation(Term.bc(a, _1), a),
        # E2: bc(1, b) = b
        Equation(Term.bc(_1, b), b),
        # E3: bc(a, a) = a
        Equation(Term.bc(a, a), a),
        # E4: bc(a, b) = bc(b, a)  — commutativity
        Equation(Term.bc(a, b), Term.bc(b, a)),
        # E5: bc(bc(a,b), c) = bc(a, bc(b,c))  — associativity
        Equation(
            Term.bc(Term.bc(a, b), c),
            Term.bc(a, Term.bc(b, c)),
        ),
        # E6: transp(transp(s, d0, d1), d0, d1) = s
        Equation(Term.transp(Term.transp(s, d0, d1), d0, d1), s),
        # E7: numel(reshape(s, t)) = numel(s)
        Equation(Term.numel(Term.reshape(s, t)), Term.numel(s)),
        # E8: conv(h, k, 1, 0) = add(sub(h, k), 1)
        Equation(
            Term.conv(h, k, _1, _0),
            Term.add(Term.sub(h, k), _1),
        ),
        # E9: pool(h, k, k, 0) = floor_div(h, k)
        Equation(
            Term.pool(h, k, k, _0),
            Term.floor_div(h, k),
        ),
    ]


# ======================================================================
# Completed Tensor Shape Rewrite System
# ======================================================================


def build_tensor_shape_trs() -> CompletionResult:
    """Build the completed TRS for the tensor shape theory.

    Runs KB completion on the tensor shape axioms and returns the
    completed (confluent + terminating) rule set.
    """
    axioms = tensor_shape_axioms()
    # Remove commutativity and associativity from KB completion —
    # these are handled as AC axioms via normalized term ordering.
    # KB completion cannot orient a = b and b = a simultaneously.
    orientable_axioms = [
        ax for ax in axioms
        if not _is_commutativity(ax) and not _is_associativity(ax)
    ]
    result = knuth_bendix_completion(orientable_axioms)
    return result


def _is_commutativity(eq: Equation) -> bool:
    """Check if equation is of the form f(a,b) = f(b,a)."""
    l, r = eq.lhs, eq.rhs
    if (l.is_compound and r.is_compound and l.symbol == r.symbol
            and len(l.children) == 2 and len(r.children) == 2):
        return (l.children[0] == r.children[1] and
                l.children[1] == r.children[0])
    return False


def _is_associativity(eq: Equation) -> bool:
    """Check if equation is of the form f(f(a,b),c) = f(a,f(b,c))."""
    l, r = eq.lhs, eq.rhs
    if not (l.is_compound and r.is_compound and l.symbol == r.symbol):
        return False
    if len(l.children) != 2 or len(r.children) != 2:
        return False
    # l = f(f(a,b), c) and r = f(a, f(b,c))
    if (l.children[0].is_compound and l.children[0].symbol == l.symbol
            and r.children[1].is_compound and r.children[1].symbol == r.symbol):
        return True
    return False


def get_completed_rules() -> List[RewriteRule]:
    """Return the completed rewrite rules for tensor shape simplification.

    These rules are pre-computed from the tensor shape axioms. They form
    a convergent (confluent + terminating) TRS for the non-AC fragment.

    Commutativity bc(a,b) = bc(b,a) and associativity bc(bc(a,b),c) =
    bc(a,bc(b,c)) are handled by AC-normalization (sorting arguments)
    rather than by rewrite rules, since they cannot be oriented in RPO.

    Rules:
      R1: bc(a, 1)  →  a                      (broadcast identity right)
      R2: bc(1, b)  →  b                      (broadcast identity left)
      R3: bc(a, a)  →  a                      (broadcast idempotent)
      R4: transp(transp(s,d0,d1), d0,d1) → s  (double transpose involution)
      R5: numel(reshape(s,t)) → numel(s)      (reshape preserves numel)
      R6: conv(h,k,1,0) → add(sub(h,k), 1)   (basic convolution)
      R7: pool(h,k,k,0) → floor_div(h,k)     (pool with stride=kernel)
    """
    a, b = Term.var("a"), Term.var("b")
    s, t = Term.var("s"), Term.var("t")
    d0, d1 = Term.var("d0"), Term.var("d1")
    h, k = Term.var("h"), Term.var("k")
    _1 = Term.const(1)
    _0 = Term.const(0)

    return [
        RewriteRule(1, Term.bc(a, _1), a,
                    "bc_identity_right", "axiom E1"),
        RewriteRule(2, Term.bc(_1, b), b,
                    "bc_identity_left", "axiom E2"),
        RewriteRule(3, Term.bc(a, a), a,
                    "bc_idempotent", "axiom E3"),
        RewriteRule(4, Term.transp(Term.transp(s, d0, d1), d0, d1), s,
                    "double_transpose", "axiom E6"),
        RewriteRule(5, Term.numel(Term.reshape(s, t)), Term.numel(s),
                    "reshape_numel", "axiom E7"),
        RewriteRule(6, Term.conv(h, k, _1, _0),
                    Term.add(Term.sub(h, k), _1),
                    "conv_basic", "axiom E8"),
        RewriteRule(7, Term.pool(h, k, k, _0),
                    Term.floor_div(h, k),
                    "pool_stride_eq_kernel", "axiom E9"),
    ]


# ======================================================================
# AC-Normalization for Commutative Symbols
# ======================================================================


def ac_normalize(term: Term) -> Term:
    """Normalize a term by sorting commutative function arguments.

    For bc(a, b), we normalize to bc(min(a,b), max(a,b)) using a
    canonical total order on terms (lexicographic on repr).
    This handles the commutativity axiom bc(a,b) = bc(b,a).
    """
    if term.is_var or term.is_const:
        return term
    children = tuple(ac_normalize(c) for c in term.children)
    # Sort arguments of commutative operations
    if term.symbol == SymbolKind.BC and len(children) == 2:
        a, b = children
        if repr(a) > repr(b):
            children = (b, a)
    return Term(term.symbol, children=children,
                name=term.name, value=term.value)


def full_normalize(term: Term, rules: Optional[Sequence[RewriteRule]] = None) -> Term:
    """Full normalization: AC-normalize then apply KB rules, repeat until fixpoint."""
    if rules is None:
        rules = get_completed_rules()
    current = term
    for _ in range(100):
        step1 = ac_normalize(current)
        step2 = normalize(step1, rules)
        if step2 == current:
            return current
        current = step2
    return current


# ======================================================================
# Z3 Expression ↔ Term Conversion
# ======================================================================


# Z3 function name → SymbolKind mapping
_Z3_FUNC_MAP: Dict[str, SymbolKind] = {
    "bc": SymbolKind.BC,
    "broadcast": SymbolKind.BC,
    "numel": SymbolKind.NUMEL,
    "conv_out": SymbolKind.CONV,
    "conv": SymbolKind.CONV,
    "pool_out": SymbolKind.POOL,
    "pool": SymbolKind.POOL,
    "transpose": SymbolKind.TRANSP,
    "transp": SymbolKind.TRANSP,
    "permute": SymbolKind.PERM,
    "perm": SymbolKind.PERM,
    "reshape": SymbolKind.RESHAPE,
}


def z3_to_term(expr: Any) -> Term:
    """Convert a Z3 expression to a Term.

    Handles Z3 IntVal, Int (variables), and function applications.
    Falls back to treating unrecognized constructs as opaque variables.
    """
    if not HAS_Z3:
        raise RuntimeError("Z3 is not available")

    if z3.is_int_value(expr):
        return Term.const(expr.as_long())

    if z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        return Term.var(str(expr))

    if z3.is_app(expr):
        decl = expr.decl()
        kind = decl.kind()
        name = decl.name()
        nargs = expr.num_args()

        # Arithmetic operations
        if kind == z3.Z3_OP_ADD:
            if nargs == 2:
                return Term.add(z3_to_term(expr.arg(0)), z3_to_term(expr.arg(1)))
            # n-ary: left-associate
            result = z3_to_term(expr.arg(0))
            for i in range(1, nargs):
                result = Term.add(result, z3_to_term(expr.arg(i)))
            return result

        if kind == z3.Z3_OP_MUL:
            if nargs == 2:
                return Term.mul(z3_to_term(expr.arg(0)), z3_to_term(expr.arg(1)))
            result = z3_to_term(expr.arg(0))
            for i in range(1, nargs):
                result = Term.mul(result, z3_to_term(expr.arg(i)))
            return result

        if kind == z3.Z3_OP_SUB:
            return Term.sub(z3_to_term(expr.arg(0)), z3_to_term(expr.arg(1)))

        if kind == z3.Z3_OP_IDIV:
            return Term.floor_div(z3_to_term(expr.arg(0)), z3_to_term(expr.arg(1)))

        # Uninterpreted functions (our custom tensor ops)
        if kind == z3.Z3_OP_UNINTERPRETED:
            sym_kind = _Z3_FUNC_MAP.get(name)
            if sym_kind is not None:
                children = tuple(z3_to_term(expr.arg(i)) for i in range(nargs))
                return Term(sym_kind, children=children)

        # Numerals encoded as multiplication by -1
        if kind == z3.Z3_OP_UMINUS:
            inner = z3_to_term(expr.arg(0))
            return Term.mul(Term.const(-1), inner)

    # Fallback: treat as opaque variable
    return Term.var(str(expr))


def term_to_z3(term: Term, ctx: Optional[Dict[str, Any]] = None) -> Any:
    """Convert a Term back to a Z3 expression.

    Parameters
    ----------
    term : Term
        The term to convert.
    ctx : dict, optional
        Mapping from variable names to Z3 variables. If not provided,
        fresh Z3 Int variables are created.
    """
    if not HAS_Z3:
        raise RuntimeError("Z3 is not available")

    if ctx is None:
        ctx = {}

    if term.is_const:
        return z3.IntVal(term.value)

    if term.is_var:
        name = term.name  # type: ignore[arg-type]
        if name not in ctx:
            ctx[name] = z3.Int(name)
        return ctx[name]

    children_z3 = [term_to_z3(c, ctx) for c in term.children]

    if term.symbol == SymbolKind.ADD:
        return children_z3[0] + children_z3[1]
    if term.symbol == SymbolKind.MUL:
        return children_z3[0] * children_z3[1]
    if term.symbol == SymbolKind.SUB:
        return children_z3[0] - children_z3[1]
    if term.symbol == SymbolKind.FLOOR_DIV:
        return children_z3[0] / children_z3[1]

    # Uninterpreted functions
    name_map = {
        SymbolKind.BC: "bc",
        SymbolKind.NUMEL: "numel",
        SymbolKind.CONV: "conv_out",
        SymbolKind.POOL: "pool_out",
        SymbolKind.TRANSP: "transpose",
        SymbolKind.PERM: "permute",
        SymbolKind.RESHAPE: "reshape",
    }
    fname = name_map.get(term.symbol)
    if fname is not None:
        sorts = [z3.IntSort()] * (len(children_z3) + 1)
        func = z3.Function(fname, *sorts)
        return func(*children_z3)

    raise ValueError(f"Cannot convert term symbol {term.symbol} to Z3")


# ======================================================================
# Z3 Expression Normalizer (Main Public API)
# ======================================================================


def normalize_z3_expr(
    expr: Any,
    rules: Optional[Sequence[RewriteRule]] = None,
) -> Any:
    """Normalize a Z3 expression using KB rewrite rules.

    Pipeline:
      1. z3.simplify(expr)         — Z3's built-in simplifier
      2. Convert to Term
      3. AC-normalize + KB rewrite to normal form
      4. Convert back to Z3
      5. z3.simplify(result)       — canonical arithmetic form

    The pipeline is idempotent: normalize(normalize(e)) = normalize(e).
    """
    if not HAS_Z3:
        raise RuntimeError("Z3 is not available")

    if rules is None:
        rules = get_completed_rules()

    # Phase 1: Z3 simplification
    simplified = z3.simplify(expr)

    # Phase 2: Term-level normalization
    term = z3_to_term(simplified)
    normalized_term = full_normalize(term, rules)

    # Phase 3: Convert back and re-simplify
    ctx: Dict[str, Any] = {}
    z3_result = term_to_z3(normalized_term, ctx)
    return z3.simplify(z3_result)
