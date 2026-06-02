# Statistical power & sample-size justification (Step 120)

Exact-binomial power analysis at α = 0.05 over **12** headline claims, with observed counts read straight from the committed regeneration artifacts.

## Zero-failure claims (false alarms / false positives)

| claim | n | 95% upper bound | power@1% | power@5% | n needed (≤5%) | powered? |
| --- | --- | --- | --- | --- | --- | --- |
| fp_stress_sound_zero_fa | 101 | 0.0292 | 0.6376 | 0.9944 | 59 | True |
| natural_sound_zero_fa | 29 | 0.0981 | 0.2528 | 0.7741 | 59 | False |
| corpus_ext_zero_fp | 74 | 0.0397 | 0.5247 | 0.9775 | 59 | True |
| differential_zero_fa | 1235 | 0.0024 | 1.0 | 1.0 | 59 | True |

Pooled across every clean trial (1439 clean models, zero observed false alarms) the aggregate one-sided 95% upper bound on the false-alarm rate is **0.0021**.

## Perfect-recall claims

| claim | n | 95% lower bound | power vs r=0.95 | n needed (≥0.95) | powered? |
| --- | --- | --- | --- | --- | --- |
| corpus_ext_recall | 153 | 0.9806 | 0.9996 | 59 | True |
| blind_recall | 138 | 0.9785 | 0.9992 | 59 | True |
| differential_unsafe_recall | 765 | 0.9961 | 1.0 | 59 | True |
| mutation_kill_rate | 376 | 0.9921 | 1.0 | 59 | True |

## Paired McNemar comparisons

Minimum all-one-sided discordant pairs for significance at α=0.05: **6**.

| comparison | discordant | exact p | needed | powered? |
| --- | --- | --- | --- | --- |
| mcnemar_vs_runtime_forward | 1 | 1.0 | 6 | False |
| mcnemar_vs_runtime_backward | 0 | 1.0 | 6 | False |
| mcnemar_vs_pytea | 2 | 0.5 | 6 | False |
| mcnemar_vs_noop | 8 | 0.0078 | 6 | True |

## Summary

- every zero-failure claim is powered to exclude a 5% rate: **False**
- every perfect-recall claim is powered to certify a 95% floor: **True**
- McNemar comparisons that are significant *and* adequately powered: **1**
