# Post-freeze second-wave results (wave 1 measured; wave 2 pre-registered)

## Wave-1 catch counts (N=15)

| Tool | Catches of upstream bug | RP fire (any axis) | Off-axis FP |
|---|---:|---:|---:|
| TensorGuard      | **5/15** (33.3%) | 6/15 | 1/15 |
| FakeTensorMode   | 2/15 (13.3%) | 2/15 | 0/15 |
| Pytea (silent-skip-corrected) | 3/15 (20.0%) | 3/15 | 0/15 |

Wilson 95% CI on TG catch-rate: **[15.17%, 58.32%]**.

## Wave-1 statistical tests

| Comparison | Test | p-value |
|---|---|---:|
| TG vs FakeTensorMode | Fisher exact (two-sided) | 0.39 |
| TG vs Pytea          | Fisher exact (two-sided) | 0.68 |
| TG vs FakeTensorMode | McNemar exact (two-sided) | 0.219 |
| TG vs Pytea          | McNemar exact (two-sided) | 0.625 |

Neither pair reaches `α=0.05` on N=15.  We report this as a
directional finding (TG point estimate strictly above both
execution-based baselines), not a separation claim.

## Power calculation

Conditioning on the wave-1 point estimates and assuming wave-2
catches are drawn at the same per-tool Bernoulli rates, the
smallest second wave at which the pooled-sample Fisher exact
falls below `α=0.05`:

| Pair | `N_new` for `p<0.05` (one-sided) | Total N |
|---|---:|---:|
| TG vs FakeTensorMode | **26** | 41 |
| TG vs Pytea          | 77 | 92 |
| Either pair to BF₁₀ ≥ 10 (uniform prior) | 56 | 71 |

## Wave-2 pre-registration

* Window: `2026-04-08..2026-08-31`.
* Query: identical to wave 1
  (`experiments_v5/v8/REAL_BUG_PREREG_QUERY_WAVE2.md`).
* Inclusion rule: next `N_new ≥ 26` PRs in chronological order;
  ceiling `N_new = 100`; no further filtering.
* Pre-registered primary tests: Fisher exact two-sided
  on TG vs FakeTensorMode and TG vs Pytea on the pooled
  wave-1 ∪ wave-2 sample; Wilson 95% CI on TG catch-rate.
* Wave-2 collection scheduled for the camera-ready window.

## Paper claims cited by this artifact

* Abstract sentence on the unfiltered N=15 post-freeze sample.
* Eval section paragraph on the unfiltered pre-registered
  post-freeze sample (Table on per-tool verdict triple).
* Limitations paragraph on N=15 non-separability.
