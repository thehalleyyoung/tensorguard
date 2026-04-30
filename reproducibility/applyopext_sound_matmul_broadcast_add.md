# Closed Lean lemmas: `applyOpExt_sound_matmul` and `applyOpExt_sound_broadcast_add`

## What changed
Two operators that the previous revision discharged inside
`Theorem (Compositional soundness; mechanised fragment)` by an
operator-agnostic composition witness — `broadcast_add` and `matmul` —
are now closed as explicit per-operator soundness lemmas in Lean 4.

The new lemmas live in
`lean/TensorGuard/AssumeGuaranteeExtended.lean` and read:

```
theorem applyOpExt_sound_broadcast_add
    (s s' : Shape)
    (h : applyOpExt .broadcast_add s = some s') :
    s' = s

theorem applyOpExt_sound_matmul
    (s s' : Shape)
    (h : applyOpExt .matmul s = some s') :
    s.length ≥ 3 ∧ s' = s
```

The `broadcast_add` lemma states that the simplified single-shape
`broadcast_add` rule is the identity on the input shape (the two-input
broadcast contract is discharged compositionally via the rule for
shape-pair `add` already in `app:ag-proofs`).

The `matmul` lemma states that the single-shape `matmul` rule
succeeds only on inputs of rank ≥ 3 and that the verdict shape is the
input shape (the rank-2 contraction is delegated to the two-input
`matmul2`/`V5OperatorRules.matmul` lemma already mechanised).

## How to reproduce
```bash
cd lean
lake build
```
The build completes with `EXIT=0` and no `sorry` is introduced; see
`lean/build_round01_improver.log`. The total operator count theorem
`operator_count` (asserting `all_operators.length = 17`) still holds
by `rfl`, and 17 of 17 operators now carry an `applyOpExt_sound_*`
or directly-equivalent verdict lemma.

## Which paper claim this discharges
- `\Cref{thm:ag-sound}` (Compositional soundness; mechanised
  fragment) is now stated as: 17 of 17 operators with closed
  `applyOpExt_sound_*` lemmas; the previous
  `\Cref{ax:operator-agnostic-witness}` is now restated as a remark
  (`\Cref{rem:operator-agnostic-witness}`) recording the historical
  obligation and its discharge.
- The abstract claim "$17/17$ per-operator soundness lemmas,
  including `matmul` and `broadcast_add`, closed sorry-free in this
  revision" is backed by these two lemmas.

## Verification command
```
grep -nE "(:= sorry|by sorry|^[[:space:]]*sorry$)" lean/TensorGuard/*.lean lean/TheoryCombination.lean
```
returns no matches.
