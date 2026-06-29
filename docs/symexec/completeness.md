# TensorGuard symbolic-execution — Completeness Characterization

> Generated from `src/symexec/completeness_contract.py` — the single source of truth. Do not edit by hand; run `python -m src.symexec.completeness_contract > docs/symexec/completeness.md` and it is pinned by `tests/test_symexec_completeness.py`.

## Relative-completeness theorem

Let `O` be a modeled operation the engine reaches on an analysed path. If every operand of `O` on which its runtime well-formedness depends is pinned to a **known** (non-`⊤`) value in the abstract store, and `O`'s runtime precondition is violated on those values, then the engine **reports** the corresponding bug. Completeness is thus *relative to the precision of the abstract store*: the only ways a real forced failure in a covered kind escapes are an unknown (`⊤`) operand or an unreached operation — exactly where the engine abstains.

## The complete fragment

For these bug kinds the engine is relatively complete under the witness condition shown:

| Bug kind | Violated precondition | Witness condition (when the guarantee bites) | Code |
| --- | --- | --- | --- |
| `matmul_dim_mismatch` | `dims_equal` | both contracted dimensions (the last dim of the lhs and the second-to-last dim of the rhs) are known and unequal. | `src/symexec/interpreter.py:_check_matmul` |
| `broadcast_mismatch` | `broadcast_compat` | two aligned trailing dimensions are both known, both ≠ 1, and unequal. | `src/symexec/interpreter.py:_check_broadcast` |
| `layer_dim_mismatch` | `feature_match` | the layer's declared in-features and the input's last dim are both known and unequal (nn.Linear and the modeled conv/norm layers). | `src/symexec/interpreter.py:_check_layer / _apply_layer_call` |
| `reshape_size_mismatch` | `numel_match` | the source element count and the fully-specified target element count (no -1 inferred dim) are both known and unequal. | `src/symexec/interpreter.py:_check_reshape` |
| `cat_shape_mismatch` | `dims_equal` | two inputs have known shapes that disagree on a non-concatenation axis. | `src/symexec/interpreter.py:_check_cat / _check_stack` |
| `einsum_dim_mismatch` | `dims_equal` | a shared einsum index is bound to two known, unequal sizes. | `src/symexec/interpreter.py:_check_einsum` |
| `axis_out_of_range` | `index_in_range` | the reduction/permutation axis is a known constant and the operand's rank is known, with the axis outside [-rank, rank). | `src/symexec/interpreter.py:_check_axis` |
| `tensor_index_oob` | `index_in_range` | the index is a known constant and the indexed dimension's extent is known, with the index outside [-extent, extent). | `src/symexec/interpreter.py:_report_index_oob` |
| `rank_index_error` | `index_in_range` | the index is a known constant and the list/tuple length is known, with the index outside [-length, length). | `src/symexec/interpreter.py:_check_index / _len_of` |
| `negative_dimension` | `dim_nonneg` | a tensor-constructor dimension argument is a known negative constant. | `src/symexec/interpreter.py:_check_negative_dim` |
| `division_by_zero` | `divisor_nonzero` | the divisor of a `/`, `//` or `%` is the known constant 0 on the path. | `src/symexec/interpreter.py:_check_div_by_zero` |
| `unpack_arity_mismatch` | `arity_match` | the right-hand side is a known fixed-arity tuple/list and the number of unpack targets differs (no starred target). | `src/symexec/interpreter.py:_check_unpack` |
| `return_arity_contract` | `arity_match` | a called function returns a known fixed arity on every path and the call site unpacks a different number of targets. | `src/symexec/interpreter.py:_check_return_arity` |
| `einops_pattern_mismatch` | `arity_match` | the rearrange/reduce/repeat pattern is a literal, ellipsis-free `lhs -> rhs` (so the LHS top-level group count is known) and the input rank is known and differs from it — a structural conflict that also covers a duplicated axis name or an output axis absent from a known literal LHS. (Decomposition-divisibility within a group stays best-effort and is not part of this witness condition.) | `src/symexec/interpreter.py:_check_einops` |
| `none_propagation` | `not_none` | the dereferenced or unpacked value is pinned to the abstract None (a known, non-⊤ NoneVal) at the use site — attribute access, indexing, or tuple-unpacking a None raises AttributeError/TypeError. (Values that first pass through an opaque call become ⊤ and are abstained on, not flagged — that is the precision limit, not a detector gap.) | `src/symexec/interpreter.py:_report_none_deref / _report_unpack` |

## Outside the guarantee (sound, best-effort)

These detectors are sound (no false positives) but make **no** completeness claim: the triggering property is only partially modeled, so a genuine failure may be missed (the engine abstains rather than guessing).

| Bug kind | Why no completeness guarantee | Code |
| --- | --- | --- |
| `channel_axis_mismatch` | requires tracking a semantic channel axis through reshapes/permutes; when the channel position becomes ⊤ the check abstains rather than guessing. | `src/symexec/interpreter.py (channel-axis handler)` |
| `axis_name_construction` | diagnostic for constructing an axis from a name that cannot exist; depends on name-resolution heuristics that are intentionally partial. | `src/symexec/interpreter.py (axis-name handler)` |

## Notes

* Completeness is *relative* to the abstract store's precision: the only ways a genuine forced failure in a covered kind goes unreported are (i) a needed operand is ⊤ (unknown), or (ii) the operation is not reached under the path model. Both are precision limits where the engine ABSTAINS, never a missing detector.
* Reachability is approximated by the path model: a branch the guard analysis cannot refine is either explored or merged with widening, so an operation under a feasible-but-unrefinable guard may be analysed with ⊤ operands (losing the completeness guarantee) but is never falsely reported.
* The witness condition of each clause is exactly the operand set the matching certificate (`src.symexec.certificate`) records; when those operands are known, certify+replay reproduce the failure, so 'reported' and 'replayably certified' coincide on the complete fragment.
* Soundness and completeness are the two directions of one contract: the engine reports a covered bug IFF a forced failure is provable on known operands. The machine-checked Lean `witness` lemmas give the ⇐ direction (report ⇒ failure); these clauses give the ⇒ direction (failure on known operands ⇒ report) for the enumerated kinds.
* Modes interact with completeness, not soundness: `sound` mode may drop reports that fail the positive-feasibility gate (trading completeness for an even stronger soundness story), while `heuristic` mode reports additional *suspicions* outside this proven fragment and labels them as such.

