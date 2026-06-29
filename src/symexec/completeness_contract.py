"""Completeness characterization of the modeled fragment (SYMEXEC_100_STEPS Step 97).

Soundness (Steps 91-93, machine-checked in `lean/TensorGuard/Symexec/`) answers
*"if we report, is it real?"* — yes, always.  This module is the companion
**importable source of truth** for the dual question, *"which real bugs are we
guaranteed to find?"*  The companion document ``docs/symexec/completeness.md`` is
generated from here (:func:`render_markdown`) and pinned by
``tests/test_symexec_completeness.py``.

Relative completeness, not absolute
-----------------------------------
A whole-program shape analyser cannot be both sound and absolutely complete
(reachability and integer arithmetic are undecidable).  We therefore state a
precise **relative-completeness** theorem — completeness *modulo the precision of
the abstract store*:

    COMPLETENESS THEOREM (per detector).  Let O be a modeled operation that the
    engine reaches on some analysed path π.  Suppose every operand of O on which
    its runtime well-formedness depends is pinned to a *known* (non-⊤) value in
    the abstract store at O along π, and O's runtime precondition (the named
    predicate in `src.symexec.certificate.PRECONDITIONS`) is violated on those
    values.  Then the engine **reports** the corresponding bug.

In other words: within the modeled operator set, the *only* reasons a genuine
forced failure goes unreported are (i) an operand the detector needs is ⊤
(unknown / abstracted away), or (ii) the operation is not reached under the path
model (e.g. guarded by a condition the engine cannot refine, so the branch is
conservatively pruned or merged to ⊤).  Both are precision limits, not detector
gaps — and both are exactly the situations in which the engine *abstains* rather
than guessing, which is what preserves soundness.  Completeness and the
no-false-positive guarantee are thus two sides of the same "report iff a forced
failure is provable on known operands" contract.

This module enumerates, per :class:`~src.symexec.bugs.SymBugKind`, the *witness
condition* (which operands must be known for the guarantee to bite) and, for the
kinds where no completeness guarantee is offered, why.  Nothing here is
aspirational: every clause names a real bug kind, and every cited precondition is
a real entry in the certificate vocabulary, both pinned by the test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    "CompletenessClause",
    "COMPLETE_FOR",
    "NO_GUARANTEE",
    "COMPLETENESS_NOTES",
    "render_markdown",
]


@dataclass(frozen=True)
class CompletenessClause:
    """One detector's relative-completeness clause.

    ``kind`` is a :class:`SymBugKind` value; ``predicate`` is the violated
    runtime precondition (an entry in ``certificate.PRECONDITIONS``) or ``None``
    when the kind is excluded from the guarantee; ``condition`` is the witness
    condition that must hold for the guarantee to bite; ``code`` cites the
    detector site."""

    kind: str
    predicate: Optional[str]
    condition: str
    code: str


# --------------------------------------------------------------------------- #
# Kinds with a relative-completeness guarantee.                               #
# Each guarantee bites once the listed operands are *known* (non-⊤) and the   #
# operation is reached on the analysed path.                                  #
# --------------------------------------------------------------------------- #
COMPLETE_FOR: List[CompletenessClause] = [
    CompletenessClause(
        "matmul_dim_mismatch", "dims_equal",
        "both contracted dimensions (the last dim of the lhs and the second-to-"
        "last dim of the rhs) are known and unequal.",
        "src/symexec/interpreter.py:_check_matmul",
    ),
    CompletenessClause(
        "broadcast_mismatch", "broadcast_compat",
        "two aligned trailing dimensions are both known, both ≠ 1, and unequal.",
        "src/symexec/interpreter.py:_check_broadcast",
    ),
    CompletenessClause(
        "layer_dim_mismatch", "feature_match",
        "the layer's declared in-features and the input's last dim are both "
        "known and unequal (nn.Linear and the modeled conv/norm layers).",
        "src/symexec/interpreter.py:_check_layer / _apply_layer_call",
    ),
    CompletenessClause(
        "reshape_size_mismatch", "numel_match",
        "the source element count and the fully-specified target element count "
        "(no -1 inferred dim) are both known and unequal.",
        "src/symexec/interpreter.py:_check_reshape",
    ),
    CompletenessClause(
        "cat_shape_mismatch", "dims_equal",
        "two inputs have known shapes that disagree on a non-concatenation axis.",
        "src/symexec/interpreter.py:_check_cat / _check_stack",
    ),
    CompletenessClause(
        "einsum_dim_mismatch", "dims_equal",
        "a shared einsum index is bound to two known, unequal sizes.",
        "src/symexec/interpreter.py:_check_einsum",
    ),
    CompletenessClause(
        "axis_out_of_range", "index_in_range",
        "the reduction/permutation axis is a known constant and the operand's "
        "rank is known, with the axis outside [-rank, rank).",
        "src/symexec/interpreter.py:_check_axis",
    ),
    CompletenessClause(
        "tensor_index_oob", "index_in_range",
        "the index is a known constant and the indexed dimension's extent is "
        "known, with the index outside [-extent, extent).",
        "src/symexec/interpreter.py:_report_index_oob",
    ),
    CompletenessClause(
        "rank_index_error", "index_in_range",
        "the index is a known constant and the list/tuple length is known, with "
        "the index outside [-length, length).",
        "src/symexec/interpreter.py:_check_index / _len_of",
    ),
    CompletenessClause(
        "negative_dimension", "dim_nonneg",
        "a tensor-constructor dimension argument is a known negative constant.",
        "src/symexec/interpreter.py:_check_negative_dim",
    ),
    CompletenessClause(
        "division_by_zero", "divisor_nonzero",
        "the divisor of a `/`, `//` or `%` is the known constant 0 on the path.",
        "src/symexec/interpreter.py:_check_div_by_zero",
    ),
    CompletenessClause(
        "unpack_arity_mismatch", "arity_match",
        "the right-hand side is a known fixed-arity tuple/list and the number of "
        "unpack targets differs (no starred target).",
        "src/symexec/interpreter.py:_check_unpack",
    ),
    CompletenessClause(
        "return_arity_contract", "arity_match",
        "a called function returns a known fixed arity on every path and the "
        "call site unpacks a different number of targets.",
        "src/symexec/interpreter.py:_check_return_arity",
    ),
    CompletenessClause(
        "einops_pattern_mismatch", "arity_match",
        "the rearrange/reduce/repeat pattern is a literal, ellipsis-free "
        "`lhs -> rhs` (so the LHS top-level group count is known) and the input "
        "rank is known and differs from it — a structural conflict that also "
        "covers a duplicated axis name or an output axis absent from a known "
        "literal LHS. (Decomposition-divisibility within a group stays "
        "best-effort and is not part of this witness condition.)",
        "src/symexec/interpreter.py:_check_einops",
    ),
    CompletenessClause(
        "none_propagation", "not_none",
        "the dereferenced or unpacked value is pinned to the abstract None (a "
        "known, non-⊤ NoneVal) at the use site — attribute access, indexing, or "
        "tuple-unpacking a None raises AttributeError/TypeError. (Values that "
        "first pass through an opaque call become ⊤ and are abstained on, not "
        "flagged — that is the precision limit, not a detector gap.)",
        "src/symexec/interpreter.py:_report_none_deref / _report_unpack",
    ),
]


# --------------------------------------------------------------------------- #
# Kinds explicitly OUTSIDE the completeness guarantee (best-effort only).     #
# These detectors are sound (no false positives) but make no claim to fire on #
# every genuine failure, because the triggering property is not fully modeled #
# in the abstract domain.                                                     #
# --------------------------------------------------------------------------- #
NO_GUARANTEE: List[CompletenessClause] = [
    CompletenessClause(
        "channel_axis_mismatch", None,
        "requires tracking a semantic channel axis through reshapes/permutes; "
        "when the channel position becomes ⊤ the check abstains rather than "
        "guessing.",
        "src/symexec/interpreter.py (channel-axis handler)",
    ),
    CompletenessClause(
        "axis_name_construction", None,
        "diagnostic for constructing an axis from a name that cannot exist; "
        "depends on name-resolution heuristics that are intentionally partial.",
        "src/symexec/interpreter.py (axis-name handler)",
    ),
]


# --------------------------------------------------------------------------- #
# Notes tying completeness to soundness, abstain and the path model.          #
# --------------------------------------------------------------------------- #
COMPLETENESS_NOTES: List[str] = [
    "Completeness is *relative* to the abstract store's precision: the only ways "
    "a genuine forced failure in a covered kind goes unreported are (i) a needed "
    "operand is ⊤ (unknown), or (ii) the operation is not reached under the path "
    "model. Both are precision limits where the engine ABSTAINS, never a missing "
    "detector.",
    "Reachability is approximated by the path model: a branch the guard analysis "
    "cannot refine is either explored or merged with widening, so an operation "
    "under a feasible-but-unrefinable guard may be analysed with ⊤ operands "
    "(losing the completeness guarantee) but is never falsely reported.",
    "The witness condition of each clause is exactly the operand set the matching "
    "certificate (`src.symexec.certificate`) records; when those operands are "
    "known, certify+replay reproduce the failure, so 'reported' and 'replayably "
    "certified' coincide on the complete fragment.",
    "Soundness and completeness are the two directions of one contract: the engine "
    "reports a covered bug IFF a forced failure is provable on known operands. "
    "The machine-checked Lean `witness` lemmas give the ⇐ direction (report ⇒ "
    "failure); these clauses give the ⇒ direction (failure on known operands ⇒ "
    "report) for the enumerated kinds.",
    "Modes interact with completeness, not soundness: `sound` mode may drop "
    "reports that fail the positive-feasibility gate (trading completeness for an "
    "even stronger soundness story), while `heuristic` mode reports additional "
    "*suspicions* outside this proven fragment and labels them as such.",
]


# --------------------------------------------------------------------------- #
# Markdown rendering                                                          #
# --------------------------------------------------------------------------- #
def _clause_table(clauses: List[CompletenessClause], *, with_pred: bool) -> str:
    if with_pred:
        rows = [
            "| Bug kind | Violated precondition | Witness condition (when the guarantee bites) | Code |",
            "| --- | --- | --- | --- |",
        ]
        for c in clauses:
            cond = c.condition.replace("\n", " ")
            rows.append(f"| `{c.kind}` | `{c.predicate}` | {cond} | `{c.code}` |")
    else:
        rows = [
            "| Bug kind | Why no completeness guarantee | Code |",
            "| --- | --- | --- |",
        ]
        for c in clauses:
            cond = c.condition.replace("\n", " ")
            rows.append(f"| `{c.kind}` | {cond} | `{c.code}` |")
    return "\n".join(rows)


def render_markdown() -> str:
    """Render the completeness characterization as Markdown."""
    lines: List[str] = []
    lines.append("# TensorGuard symbolic-execution — Completeness Characterization")
    lines.append("")
    lines.append(
        "> Generated from `src/symexec/completeness_contract.py` — the single "
        "source of truth. Do not edit by hand; run `python -m "
        "src.symexec.completeness_contract > docs/symexec/completeness.md` and it "
        "is pinned by `tests/test_symexec_completeness.py`."
    )
    lines.append("")
    lines.append("## Relative-completeness theorem")
    lines.append("")
    lines.append(
        "Let `O` be a modeled operation the engine reaches on an analysed path. "
        "If every operand of `O` on which its runtime well-formedness depends is "
        "pinned to a **known** (non-`⊤`) value in the abstract store, and `O`'s "
        "runtime precondition is violated on those values, then the engine "
        "**reports** the corresponding bug. Completeness is thus *relative to the "
        "precision of the abstract store*: the only ways a real forced failure in "
        "a covered kind escapes are an unknown (`⊤`) operand or an unreached "
        "operation — exactly where the engine abstains."
    )
    lines.append("")
    lines.append("## The complete fragment")
    lines.append("")
    lines.append(
        "For these bug kinds the engine is relatively complete under the witness "
        "condition shown:"
    )
    lines.append("")
    lines.append(_clause_table(COMPLETE_FOR, with_pred=True))
    lines.append("")
    lines.append("## Outside the guarantee (sound, best-effort)")
    lines.append("")
    lines.append(
        "These detectors are sound (no false positives) but make **no** "
        "completeness claim: the triggering property is only partially modeled, "
        "so a genuine failure may be missed (the engine abstains rather than "
        "guessing)."
    )
    lines.append("")
    lines.append(_clause_table(NO_GUARANTEE, with_pred=False))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for n in COMPLETENESS_NOTES:
        lines.append(f"* {n}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(render_markdown())
