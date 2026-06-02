# Paired effect sizes & dual multiple-comparison correction (Step 121)

Consumer of `evaluation/confusion_matrices.json` (n = 16 items, family of 4 usable baselines). Every comparison carries a paired effect size and is corrected under both a family-wise (Holm-Bonferroni) and a false-discovery (Benjamini-Hochberg) procedure.

| baseline | b | c | Cohen's g | magnitude | Haldane OR | risk diff | McNemar p | Holm p | Holm? | BH p | BH? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runtime_forward | 1 | 0 | 0.5 | large | 3.0 | 0.0625 | 1.0 | 1.0 | False | 1.0 | False |
| runtime_backward | 0 | 0 | 0.0 | negligible | 1.0 | 0.0 | 1.0 | 1.0 | False | 1.0 | False |
| pytea | 2 | 0 | 0.5 | large | 5.0 | 0.125 | 0.5 | 1.0 | False | 1.0 | False |
| noop | 8 | 0 | 0.5 | large | 17.0 | 0.5 | 0.007812 | 0.03125 | True | 0.03125 | True |

## Summary

- comparisons carrying a paired effect size: **True** (all)
- significant after Holm-Bonferroni (FWER): **1**
- significant after Benjamini-Hochberg (FDR): **1**
- the two corrections agree on the count: **True**
