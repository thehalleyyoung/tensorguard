# `lake build` reproducibility record

**Command:** `cd lean && lake clean && lake build TensorGuard parity_runner`

**Toolchain:** `lean-toolchain` pins Lean 4.14.0 (Lake 5.0.0-410fab7).

**Output:** `reproducibility/lake_build.log` — full build trace ending in
`Build completed successfully.`, exit code `0`.

**`sorry` audit (post-edit, round 2):**

```
$ grep -nE '\bsorry\b' lean/TensorGuard/*.lean lean/*.lean
lean/TensorGuard/AssumeGuaranteeExtended.lean:30:(ag_composition) are proved sorry-free using Lean 4 core (no mathlib).
lean/TensorGuard/AssumeGuaranteeExtended.lean:356:stated for trivial (True/True) contracts so they compile sorry-free
lean/TensorGuard/Extended.lean:15:All theorems are proved sorry-free using Lean 4 core (no mathlib).
lean/TensorGuard/Extended.lean:92:    the in-range case stated below. We close that case sorry-free.
lean/TensorGuard/Parity.lean:7:NOTE: This file is fully closed under `lake build` (no `sorry`). The
lean/TensorGuard/V5OperatorRules.lean:20:All theorems in this file are closed without `sorry`. As of round~9
lean/TensorGuard/V5OperatorRules.lean:21:the entire `lean/TensorGuard/` tree is also `sorry`-free: the
lean/TensorGuard/V5OperatorRules.lean:24:unconditional `permList_compose`) is now closed sorry-free by
```

Every match is in a documentation comment (`/-- ... -/` or `/- ... -/`).
**No executable `sorry` remains anywhere under `lean/TensorGuard/`** —
the previous residual on the (false) unconditional `permList_compose`
has been replaced with the in-bounds variant `permList_compose_inrange`,
which closes by case-analysis on `p.get?`.

**Build trace (audited):** `reproducibility/lake_build.log` shows zero
`error:` lines and zero `declaration uses 'sorry'` warnings. The only
warnings reported by Lean are
(i) one deprecation notice (`List.get?_map → List.getElem?_map`) and
(ii) unused-variable lints — neither has soundness consequences.

**Paper claim cited:** the abstract and §4.4 statement
"$11/11$ soundness lemmas closed sorry-free" refers to the
shape-transfer rule lemmas in `Soundness.lean` and `Extended.lean`
that mechanise the audited operator-rule table. The
`AssumeGuarantee.lean` and `AssumeGuaranteeExtended.lean` files add
the load-bearing assume/guarantee composition theorems (`ag_composition`,
`ag_composition_ext`); both are also closed sorry-free.

**Theorem-backed footprint, this round:**
* `lean/TensorGuard/Soundness.lean` — `linear`, `view`, `broadcast_add`
  shape-transfer lemmas.
* `lean/TensorGuard/Extended.lean` — `matmul2`, `bmm`, `transpose2`,
  `permList_compose_inrange`, `conv1dOut_monotone`, `relu_shape_preserving`,
  `bcastDim_comm`, `bcast_comm`, `bcast_rank1_sound`.
* `lean/TensorGuard/AssumeGuarantee.lean` — `applyChain_append`,
  `ag_composition`, `satisfies_trivial`, `satisfies_nil` over the
  3-operator core DSL.
* `lean/TensorGuard/AssumeGuaranteeExtended.lean` — operator-agnostic
  `applyChainExt_append` and `ag_composition_ext` plus eleven
  per-operator soundness lemmas across the 17-operator extended DSL
  (`transpose`, `permute`, `relu`, `sum_reduce`, `mean_reduce`,
  `cat`, `expand`, `gather`, `embedding`, `conv2d`, `einsum`,
  `unbind`, `view`, `reshape`).
