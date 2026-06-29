"""Operational semantics of the modeled Python fragment (SYMEXEC_100_STEPS Step 91).

This module is the **importable source of truth** for the small-step operational
semantics that the symbolic-execution engine (`src.symexec.interpreter`)
implements.  The companion document ``docs/symexec/semantics.md`` is generated
from this module (:func:`render_markdown`) and kept in sync by
``tests/test_symexec_semantics.py``.

Why an *abstract* small-step semantics
--------------------------------------
The engine is an abstract interpreter.  It does not run the program on concrete
tensors; it executes a **small-step semantics over an abstract store** σ that
maps names to :class:`~src.symexec.values.AbstractValue` lattice elements (and
``self.<attr>`` slots to the same).  We give the semantics in two layers:

  * the **concrete** small-step relation ``⟨s, σ_c⟩ → ⟨s', σ_c'⟩`` over a
    concrete store σ_c (the meaning a real Python execution would have, for the
    modeled forms only); and
  * the **abstract** small-step relation ``⟨s, σ⟩ ⇒ ⟨s', σ'⟩`` that the engine
    actually computes, where σ is an abstract store.

Soundness (Step 92) is the statement that ⇒ over-approximates → under the Galois
connection (α, γ) of Step 93: every concrete step is matched by an abstract step
on the abstracted store, so a *reported* bug corresponds to a real failing
concretization and a SAFE verdict (within the modeled fragment) admits no
modeled-class violation.

Nothing here is aspirational: every syntactic form is tagged with the concrete
``ast`` node the engine dispatches on, and the test pins that those nodes are
actually handled in ``interpreter.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = [
    "Configuration",
    "StoreComponent",
    "SyntaxForm",
    "STORE",
    "STATEMENT_FORMS",
    "EXPRESSION_FORMS",
    "ABSTRACTION_NOTES",
    "render_markdown",
]


# --------------------------------------------------------------------------- #
# Abstract store / configuration                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StoreComponent:
    """One component of the abstract store σ."""

    name: str
    description: str
    code: str


@dataclass(frozen=True)
class Configuration:
    """A small-step configuration ``⟨code, σ⟩``."""

    description: str


CONFIG = Configuration(
    "A configuration is ⟨c, σ⟩ where c is the residual code (a list of "
    "statements with a program counter) and σ is the abstract store.  A method/"
    "function body is a sequence of statements executed left-to-right; the "
    "engine analyses the `__main__` harness, free top-level functions, and "
    "class `forward`/`__call__` methods (bare module-level statements are NOT "
    "executed)."
)

STORE: List[StoreComponent] = [
    StoreComponent(
        "env : Name ⇀ AbstractValue",
        "the variable environment; a missing name reads as ⊤ (unknown).",
        "src/symexec/state.py:State.env, State.get/set",
    ),
    StoreComponent(
        "store : Obj × Attr ⇀ AbstractValue",
        "the object/attribute store; `self` is the canonical key for the module "
        "instance under analysis, so `self.fc = nn.Linear(...)` set in `__init__` "
        "is visible to `forward`.",
        "src/symexec/state.py:State.store, get_attr/set_attr",
    ),
    StoreComponent(
        "reachable : Bool",
        "the path reachability flag; an unreachable configuration takes no step "
        "and emits no bug.",
        "src/symexec/state.py:State.reachable",
    ),
    StoreComponent(
        "dim_facts : Constraint*",
        "path constraints over symbolic dimensions accumulated by guard "
        "refinement along the current path; a report whose failing condition is "
        "provably UNSAT under dim_facts is suppressed (feasibility gate).",
        "src/symexec/state.py:State.dim_facts; src/symexec/smt_bridge.py",
    ),
]


# --------------------------------------------------------------------------- #
# Syntactic forms + their reduction rules                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyntaxForm:
    """A modeled syntactic form, its small-step rule, and the dispatching node."""

    form: str
    ast_node: str  # the ``ast`` class name the engine dispatches on
    rule: str
    code: str
    notes: str = ""


STATEMENT_FORMS: List[SyntaxForm] = [
    SyntaxForm(
        "x = e   /   x, y = e   /   *xs = e",
        "Assign",
        "⟨x = e, σ⟩ ⇒ ⟨skip, σ[x ↦ E⟦e⟧σ]⟩.  Tuple/list targets destructure a "
        "TupleVal/ListVal componentwise; an arity mismatch is reported "
        "(UNPACK_ARITY_MISMATCH) and the targets bind ⊤.",
        "src/symexec/interpreter.py:_exec (ast.Assign), _bind_target",
    ),
    SyntaxForm(
        "x: T = e   /   x: T",
        "AnnAssign",
        "Same transfer as Assign when a value is present; a bare annotation is a "
        "no-op on σ.",
        "src/symexec/interpreter.py (ast.AnnAssign)",
    ),
    SyntaxForm(
        "x op= e",
        "AugAssign",
        "⟨x op= e, σ⟩ ⇒ ⟨x = x op e, σ⟩: desugars to the binary transfer on the "
        "current value of x.",
        "src/symexec/interpreter.py (ast.AugAssign)",
    ),
    SyntaxForm(
        "e   (expression statement)",
        "Expr",
        "⟨e, σ⟩ ⇒ ⟨skip, σ'⟩ where E⟦e⟧ is evaluated for its bug-checking side "
        "effects (e.g. a bare `a @ b`) and σ' carries any attribute writes; the "
        "value is discarded.",
        "src/symexec/interpreter.py (ast.Expr)",
    ),
    SyntaxForm(
        "pass",
        "Pass",
        "⟨pass, σ⟩ ⇒ ⟨skip, σ⟩.",
        "src/symexec/interpreter.py (ast.Pass)",
    ),
    SyntaxForm(
        "return e",
        "Return",
        "Evaluates E⟦e⟧σ, records it as the function's return value (checked "
        "against the declared/contracted arity, RETURN_ARITY_CONTRACT), and marks "
        "the continuation unreachable.",
        "src/symexec/interpreter.py (ast.Return)",
    ),
    SyntaxForm(
        "if e: S1 else: S2",
        "If",
        "Evaluates the guard; when its truth value is unknown the engine explores "
        "BOTH branches from refined copies of σ (σ ⊓ e, σ ⊓ ¬e — guard refinement "
        "adds to dim_facts) and JOINS the resulting stores at the merge point: "
        "⟨if e …, σ⟩ ⇒ ⟨skip, σ1 ⊔ σ2⟩.  A statically decidable guard prunes the "
        "dead branch (reachable=False).",
        "src/symexec/interpreter.py (ast.If); State.join; src/symexec/relational.py",
    ),
    SyntaxForm(
        "for x in e: S",
        "For",
        "Bounded unrolling toward a fixpoint: the loop body is re-executed on the "
        "joined store until σ stabilises, applying WIDENING after a bounded number "
        "of iterations to guarantee termination; the iterate value binds the "
        "element abstraction of E⟦e⟧σ (⊤ if unknown).",
        "src/symexec/interpreter.py (ast.For); State.widen; ITERATION_CAPS",
    ),
    SyntaxForm(
        "while e: S",
        "While",
        "Same fixpoint/widening treatment as `for`, iterating on the guard-refined "
        "store until a post-fixpoint is reached or the iteration cap triggers "
        "widening to ⊤ on the unstable slots.",
        "src/symexec/interpreter.py (ast.While); State.widen; ITERATION_CAPS",
    ),
    SyntaxForm(
        "with e as x: S",
        "With",
        "Binds x to E⟦e⟧σ (the context value, ⊤ if unmodeled) and executes the "
        "body; no special enter/exit effect is modeled.",
        "src/symexec/interpreter.py (ast.With)",
    ),
    SyntaxForm(
        "assert e",
        "Assert",
        "Refines σ by the asserted condition (adds to dim_facts) on the "
        "continuation; never used to emit a bug.",
        "src/symexec/interpreter.py (ast.Assert)",
    ),
    SyntaxForm(
        "def f(...): S   (nested)",
        "FunctionDef",
        "A def binds a callable summary in σ; calls are handled by E⟦·⟧ via "
        "inlining/summary application (interprocedural).",
        "src/symexec/interpreter.py (ast.FunctionDef); _analyze_function",
    ),
]


EXPRESSION_FORMS: List[SyntaxForm] = [
    SyntaxForm(
        "literal / None",
        "Constant",
        "E⟦c⟧σ = the corresponding IntVal/FloatVal/StrVal/BoolVal/NoneVal.",
        "src/symexec/interpreter.py (ast.Constant); values.py",
    ),
    SyntaxForm(
        "x",
        "Name",
        "E⟦x⟧σ = σ.env(x), or ⊤ if unbound.",
        "src/symexec/interpreter.py (ast.Name)",
    ),
    SyntaxForm(
        "obj.attr",
        "Attribute",
        "E⟦self.a⟧σ = σ.store(self, a); `.shape`/`.dtype`/`.T`/`.ndim` etc. are "
        "modeled projections of a TensorVal.",
        "src/symexec/interpreter.py (ast.Attribute)",
    ),
    SyntaxForm(
        "(e1, …)",
        "Tuple",
        "Construct a TupleVal of the evaluated elements; a `*e` element splices a "
        "sequence value.",
        "src/symexec/interpreter.py (ast.Tuple/Starred)",
    ),
    SyntaxForm(
        "[e1, …]",
        "List",
        "Construct a ListVal of the evaluated elements.",
        "src/symexec/interpreter.py (ast.List)",
    ),
    SyntaxForm(
        "{k: v, …}",
        "Dict",
        "Construct a DictVal of the evaluated key/value pairs.",
        "src/symexec/interpreter.py (ast.Dict)",
    ),
    SyntaxForm(
        "e1 op e2  (incl. @)",
        "BinOp",
        "Applies the operator transfer function: arithmetic on IntVal/FloatVal "
        "(with DIVISION_BY_ZERO when the divisor is provably 0), and `@` invokes "
        "the matmul shape transfer (MATMUL_DIM_MISMATCH) propagating the result "
        "shape.",
        "src/symexec/interpreter.py (ast.BinOp); transfer.py",
    ),
    SyntaxForm(
        "op e  (-e, not e, ~e)",
        "UnaryOp",
        "Unary transfer on the operand abstraction (e.g. USub on a known IntVal).",
        "src/symexec/interpreter.py (ast.UnaryOp)",
    ),
    SyntaxForm(
        "e1 and e2  /  e1 or e2",
        "BoolOp",
        "Short-circuit boolean abstraction over operand truth values; result is "
        "BoolVal or ⊤.",
        "src/symexec/interpreter.py (ast.BoolOp)",
    ),
    SyntaxForm(
        "e1 < e2  (and other comparisons)",
        "Compare",
        "Yields a BoolVal when both sides are concrete/related, else ⊤; feeds "
        "guard refinement in `if`/`while`/`assert`.",
        "src/symexec/interpreter.py (ast.Compare)",
    ),
    SyntaxForm(
        "e1 if c else e2",
        "IfExp",
        "Joins the two branch abstractions: E⟦e1 if c else e2⟧σ = E⟦e1⟧σ ⊔ "
        "E⟦e2⟧σ (refined by c when decidable).",
        "src/symexec/interpreter.py (ast.IfExp)",
    ),
    SyntaxForm(
        "e[i]  /  e[a:b]",
        "Subscript",
        "Index/slice transfer: integer indexing checks bounds against a known "
        "length (TENSOR_INDEX_OOB / RANK_INDEX_ERROR) and projects the element/"
        "subshape; symbolic indices abstain.",
        "src/symexec/interpreter.py (ast.Subscript); _eval_subscript",
    ),
    SyntaxForm(
        "f(e1, …, *args, **kw)",
        "Call",
        "The interprocedural / library transfer: torch ops, `nn` constructors and "
        "their `forward` application, einops/einsum, builtins (`len`, `range`, "
        "`map`/`filter`), user functions (inlined), and stubbed library calls all "
        "dispatch here, each with its own sound shape transfer + bug check.",
        "src/symexec/interpreter.py:_eval_call; stubs.py; transfer.py",
    ),
    SyntaxForm(
        "[e for x in it if c]",
        "ListComp",
        "Models the comprehension as a bounded loop producing a ListVal of the "
        "element abstraction; unbounded/opaque iterables yield a ListVal of ⊤.",
        "src/symexec/interpreter.py (ast.ListComp)",
    ),
]


# --------------------------------------------------------------------------- #
# Abstraction / soundness notes (forward references to Steps 92-93)           #
# --------------------------------------------------------------------------- #
ABSTRACTION_NOTES: List[str] = [
    "Out-of-fragment forms (data-dependent control flow that the guard model "
    "cannot refine, dynamic eval/exec, `*`-imports of opaque modules, "
    "comprehensions over opaque iterables, etc.) are handled by abstracting the "
    "affected values to ⊤ and, where a sound result cannot be guaranteed, by "
    "ABSTAINING (recording an AbstainReason and emitting no bug) rather than "
    "guessing — this is what keeps the reported set free of false positives.",
    "Control-flow joins use the store join ⊔ (pointwise value join); loops use "
    "widening ▽ after a bounded iteration count (ITERATION_CAPS) to force "
    "termination at a sound post-fixpoint.",
    "The abstract relation ⇒ is designed so that for every concrete step "
    "⟨s, σ_c⟩ → ⟨s', σ_c'⟩ there is an abstract step ⟨s, α(σ_c)⟩ ⇒ ⟨s', σ#⟩ "
    "with α(σ_c') ⊑ σ# (local soundness of each transfer function); composing "
    "these steps gives the whole-program soundness theorem of Step 92.",
    "Each per-operator transfer function is the abstraction of the operator's "
    "concrete shape semantics; Step 93 formalises (α, γ) as a Galois connection "
    "so that 'abstraction of the concrete transfer ⊑ abstract transfer' is the "
    "single proof obligation per operator.",
]


# --------------------------------------------------------------------------- #
# Markdown rendering                                                          #
# --------------------------------------------------------------------------- #
def _form_table(forms: List[SyntaxForm]) -> str:
    rows = [
        "| Form | `ast` node | Small-step rule (over the abstract store σ) | Code |",
        "| --- | --- | --- | --- |",
    ]
    for f in forms:
        rule = f.rule.replace("\n", " ")
        note = f" _{f.notes}_" if f.notes else ""
        rows.append(
            f"| `{f.form}` | `ast.{f.ast_node}` | {rule}{note} | `{f.code}` |"
        )
    return "\n".join(rows)


def render_markdown() -> str:
    """Render the operational-semantics specification as Markdown."""
    lines: List[str] = []
    lines.append("# TensorGuard symbolic-execution — Operational Semantics")
    lines.append("")
    lines.append(
        "> Generated from `src/symexec/semantics.py` — the single source of "
        "truth. Do not edit by hand; run `python -m src.symexec.semantics > "
        "docs/symexec/semantics.md` and it is pinned by "
        "`tests/test_symexec_semantics.py`."
    )
    lines.append("")
    lines.append("## Configurations")
    lines.append("")
    lines.append(CONFIG.description)
    lines.append("")
    lines.append("## The abstract store σ")
    lines.append("")
    lines.append("σ = (env, store, reachable, dim_facts):")
    lines.append("")
    lines.append("| Component | Meaning | Code |")
    lines.append("| --- | --- | --- |")
    for c in STORE:
        lines.append(f"| `{c.name}` | {c.description} | `{c.code}` |")
    lines.append("")
    lines.append(
        "Stores form a lattice: `⊥ ⊑ … ⊑ ⊤`, with a pointwise join `⊔` used at "
        "control-flow merges and a widening `▽` used to terminate loops."
    )
    lines.append("")
    lines.append("## Statements — small-step rules")
    lines.append("")
    lines.append(_form_table(STATEMENT_FORMS))
    lines.append("")
    lines.append("## Expressions — evaluation `E⟦e⟧σ`")
    lines.append("")
    lines.append(_form_table(EXPRESSION_FORMS))
    lines.append("")
    lines.append("## Abstraction & soundness notes")
    lines.append("")
    for n in ABSTRACTION_NOTES:
        lines.append(f"* {n}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(render_markdown())
