# `reshape` divisibility / inferred-dim Lean proofs

## Command that produced the artifact

```
cd lean && lake build
```

## Inputs / configuration

* Lean toolchain: Lean 4.14.0 (lake 5.0.0-410fab7).
* No mathlib dependency.
* Source file: `lean/TensorGuard/V5OperatorRules.lean`.

## What the new artifact closes

Three new sorry-free theorems were added next to the existing
`reshape` shape-transfer rule:

* `reshape_sound_zero_unknowns (input out r) (hzero h)` — when the
  reshape spec contains no `none` slot (`hzero`), success implies
  the explicit-knowns product equals the input total and the result
  is exactly the knowns list.
* `reshape_sound_one_unknown (input out r) (hone h)` — when the
  reshape spec contains exactly one `none` (the `-1` slot,
  `hone`), success implies (i) the product of the explicitly
  given dims is non-zero, (ii) it divides the input total, and
  (iii) the result is the spec with the `none` slot filled by the
  integer quotient.
* `reshape_rejects_multi_unknown (input out hmulti)` — when the
  spec contains two or more `none` slots, `reshape` returns
  `none`. This pins the multi-`-1` rejection behaviour explicitly.

## Numbers / status

* `lake build` returns `Build completed successfully.`
* No new `sorry` introduced; whole `lean/TensorGuard/` tree
  remains sorry-free.

## Paper claims that cite this artifact

* Theorem 1 (proof-grade soundness on `Cat_sound`): the
  pen-and-paper-derived `reshape` divisibility obligation is now a
  closed Lean theorem, named in the proof body.
* Theorem 2 (fragment-level soundness on `Cat_sound`): the
  preservation step for the `reshape` rule cites the three theorem
  names directly.
* Side-conditions paragraph in the calculus: the multi-`-1`
  rejection clause cites `reshape_rejects_multi_unknown`.
* Abstract: the Lean-audited count remains 28 of 79 handlers
  with the additional reshape divisibility / inferred-dim /
  multi-`-1`-rejection cases stated as closed Lean theorems.
