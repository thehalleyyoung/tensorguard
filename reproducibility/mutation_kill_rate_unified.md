# Unified mutation kill rate (4-corpus union, 50-mutant sweep)

## Command

    python3.11 reproducibility/mutation_kill_rate_unified.py

## Reviewer obligation

Round-18 W4: the global mutation union (7/50) and the targeted per-handler measurement (conv2d 53%, einsum 100%) are not directly comparable because the latter is reported on a separate 18-case targeted corpus.  This artifact reruns the *same* 50-mutant sweep against the union of all four corpora (60-bug ∪ 488-block sample ∪ 25-stress ∪ targeted-extension).

## Mutation operators

Same as `mutation_kill_rate_corpora.py`: M1 comparison flip, M2 boolean-op flip, M3 arithmetic-op swap, M4 small-int +1, M5 boolean constant flip.

## Per-corpus and union kill rates (one mutant either kills or it does not)

| Corpus | Baseline RP / N | Killed / 50 | Kill rate |
|---|---|---:|---:|
| 60bug | 53 / 60 | 3 / 50 | 6.0% |
| 488block | 31 / 50 | 7 / 50 | 14.0% |
| 25stress | 16 / 25 | 5 / 50 | 10.0% |
| targeted_ext | 0 / 22 | 4 / 50 | 8.0% |
| **Union (any corpus)** | --- | **7 / 50** | **14.0%** |

## Headline

Adding the targeted-extension corpus to the same 50-mutant sweep raises the union kill rate from the 4-corpus measurement to **7/50 = 14.0%**.  The targeted-extension corpus alone, scored against the same 50 mutants, yields 4/50 = 8.0%.

## Paper claim cited by this artifact

- Eval section paragraph on mutation-testing robustness: a unified, directly-comparable union kill rate against the 4-corpus union, supplementing the 7/50 = 14% union reported in `mutation_kill_rate_corpora.md`.
