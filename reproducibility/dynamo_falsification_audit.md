# Theorem 5 falsifiability audit (round-5 W5 / Q4)

**Reviewer ask (round 5).**
> Theorem 5 is a one-directional, mostly-tautological statement, and
> the empirical "audit" does not test it. (...) Can you specify a
> measurement that would have *falsified* the theorem, and confirm
> none of the 48 in-contract recompiles trace to a shape/dtype/rank
> bit that is not a refinement variable in any rule?

**Command.**
```
python3 experiments_v5/v8/dynamo_falsification_audit.py
```

**Falsification predicate (added round 5; this is the new artifact).**

```
exists in-contract recompile r in
  experiments_v5/dynamo_correspondence_v5.json::modules[*].
    in_contract_recompiles ::
  r.guard_kind in {SHAPE, DTYPE, RANK}
  and r.guard_var not in catalogue_refinement_vars(M).
```

In English: Theorem 5 is falsified iff Dynamo recompiles inside the
TG contract on a shape, dtype, or rank bit that the TG operator
catalogue does not declare as a refinement variable.  The theorem
already excludes recompiles on non-shape metadata (Python ints
captured at trace time, list lengths, integer scalar parameters,
tracer-id changes); these are not in the SHAPE/DTYPE/RANK bucket and
do not falsify the theorem.

**Result.**

| Guard kind        | n recompiles |   In-catalogue refinement var?  | Falsifies Thm 5? |
|-------------------|-------------:|:-------------------------------:|:----------------:|
| SHAPE             | 0            | yes                             | --- (vacuous)    |
| DTYPE             | 0            | yes                             | --- (vacuous)    |
| RANK              | 0            | yes                             | --- (vacuous)    |
| INT               | 48           | no (excluded by Thm 5)          | no               |
| LIST_LEN          | 0            | no (excluded by Thm 5)          | no               |
| TRACER            | 0            | no (excluded by Thm 5)          | no               |
| OTHER             | 0            | no (excluded by Thm 5)          | no               |
| **Total**         | **48**       |                                 | **0 / 48**       |

All 48 in-contract recompiles classify as `INT` (Python int captured
at trace time, integer scalar parameter, soft recompile from cache
size or int specialisation).  Zero recompiles fall in the
SHAPE/DTYPE/RANK bucket where Theorem 5 would have made a positive
claim that could be falsified.

**Per-module breakdown.**  See
`experiments_v5/v8/dynamo_falsification_audit.json` for the full
11-row table (`tv_resnet18` 5; `tv_resnet50` 4; `tv_squeezenet1_1`
26; etc.; `tg_verified_TinyMLP` 1, the documented
counterexample-witness for one-directionality).

**What this measurement would have caught.**  If e.g.
`tv_resnet18` had recompiled on `x.size(2) == 224`-style guards
inside the contract, the row for `tv_resnet18` would carry
`guard_kind = SHAPE` and `falsifies_theorem_5 = true`.  We
report the SHAPE bucket as zero; the necessary direction of
Theorem 5 is not falsified by the 17-module dataset.

**Caveat.**  Zero on a 17-module audit is consistent with Theorem 5
but is not a proof of necessary correspondence; a genuinely
adversarial example (a custom op that reads a shape bit not declared
by any rule) could in principle fire the falsifier.  The
catalogue-coverage table at `app:ops-extended` is the static check
against this failure mode; the audit here is the dynamic check.

**Paper claim cited.**  §4.3 (`eval_v6.tex`, "Empirical audit"
paragraph): "in-contract recompile rate is 48/544 (8.8%)".  The
falsification predicate and 0/48 SHAPE-class recompile result are
new in round 5.
