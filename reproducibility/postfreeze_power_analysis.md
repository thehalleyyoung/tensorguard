# Post-freeze power analysis (round 5 Q4)

## Setup

Observed round-4 catches on N=15 unbiased post-freeze PRs:
  * TG: 5/15 (33.3%)
  * FakeTensorMode: 2/15 (13.3%)
  * Pytea: 3/15 (20.0%)

Two-sided Fisher exact, alpha=0.05, 80% power.  Power estimated by Monte-Carlo with 2000 reps per (N, effect-size) cell, holding the observed proportions as the true effect.

## Result

| Comparison | Required N per arm |
|---|---|
| TG vs FakeTensorMode | 80 |
| TG vs Pytea | 187 |

## Reading

The shipped N=15 sample is directionally informative (TG is the top tool in absolute count on every comparable cell) but underpowered for separation at alpha=0.05.  At the observed effect sizes, a future N>=187 unbiased post-freeze replication is sufficient to upgrade either comparison to statistical separability.  Round-5 reviewer Q4 is therefore a corpus-collection question (post-freeze PR throughput) rather than a tool-quality question.
