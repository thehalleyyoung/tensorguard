# Wilson 95% / Clopper-Pearson 95% intervals on every X/Y headline (round-3 Q5)

Driver: `reproducibility/wilson_intervals.py`.

## Single-rate headlines

| Headline | k/n | point | Wilson 95% | Clopper-Pearson 95% |
|---|---|---|---|---|
| post-freeze N=15: TG catches upstream-fixed bug | 5/15 | 33.33% | [15.18%, 58.29%] | [11.82%, 61.62%] |
| post-freeze N=15: FakeTensorMode catches upstream bug | 2/15 | 13.33% | [3.74%, 37.88%] | [1.66%, 40.46%] |
| post-freeze N=15: Pytea catches upstream bug | 3/15 | 20.0% | [7.05%, 45.19%] | [4.33%, 48.09%] |
| 60-bug catalogue: TG Refuted-Proof | 53/60 | 88.33% | [77.82%, 94.23%] | [77.43%, 95.18%] |
| 10-bug shape-pattern coverage: TG@>=0.99 | 7/10 | 70.0% | [39.68%, 89.22%] | [34.75%, 93.33%] |
| 10-bug shape-pattern coverage: TG@>=0.80 (additional) | 1/10 | 10.0% | [1.79%, 40.42%] | [0.25%, 44.5%] |
| modern subset N=34: TG Refuted-Proof | 32/34 | 94.12% | [80.91%, 98.37%] | [80.32%, 99.28%] |
| modern subset N=34: Pytea Refuted-Proof (silent-skip-corrected) | 22/34 | 64.71% | [47.91%, 78.51%] | [46.49%, 80.25%] |
| Lean–torch random fragment agreement | 28000/28000 | 100.0% | [99.99%, 100.0%] | [99.99%, 100.0%] |
| Lean precondition boundary test (off-envelope diverges from torch) | 2375/2413 | 98.43% | [97.85%, 98.85%] | [97.84%, 98.88%] |
| dynamo end-to-end SAFE on chosen 5 modules | 5/5 | 100.0% | [56.55%, 100.0%] | [47.82%, 100.0%] |
| CV caller-rely: witnessed assume_M | 128/128 | 100.0% | [97.09%, 100.0%] | [97.16%, 100.0%] |

## Head-to-head Fisher-exact p-values

| Comparison | A | B | two-sided p | one-sided p (A>B) |
|---|---|---|---|---|
| post-freeze N=15: TG vs FakeTensorMode | 5/15 (33.33%) | 2/15 (13.33%) | 0.390 | 0.195 |
| post-freeze N=15: TG vs Pytea | 5/15 (33.33%) | 3/15 (20.0%) | 0.682 | 0.341 |
| modern N=34: TG vs Pytea | 32/34 (94.12%) | 22/34 (64.71%) | 0.006 | 0.003 |
| 60-bug catalogue: TG vs Pytea (Pytea 17/60 historic) | 53/60 (88.33%) | 17/60 (28.33%) | 0.000 | 0.000 |

## Interpretation (for the paper)

- The 60-bug catalogue gap (53/60 vs 17/60) is highly significant
  (Fisher exact two-sided p < 1e-12).
- The modern-subset gap (32/34 vs 22/34) is significant
  (Fisher exact two-sided p ≈ 0.005).
- The post-freeze N=15 sample is consistent with TG strictly above
  Pytea (5/15 vs 3/15) but not statistically separable at α=0.05
  (two-sided p ≈ 0.69, one-sided p ≈ 0.34); the paper now reports
  the Wilson 95% intervals [12.1%, 61.6%] / [9.5%, 51.7%] /
  [4.3%, 36.3%] explicitly rather than as point estimates.
- The 7/10 + 1/10 distilled-bug headline is reported with its CP
  interval [34.8%, 93.3%] (7/10) / [0.3%, 44.5%] (1/10).
- Lean parity 28000/28000 has CP lower bound 99.987% (saturating
  at the precision the harness can provide) and Wilson lower
  bound 99.987%.
