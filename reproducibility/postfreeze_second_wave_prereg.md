# Post-freeze second-wave pre-registration

Mirror of `experiments_v5/v8/REAL_BUG_PREREG_QUERY_WAVE2.md`, kept
under `reproducibility/` for cross-reference from the body.

## Protocol summary

* **Wave 1**: 15 PRs sampled by the pre-registered query in the
  window `created:>2026-04-07` (wave-1 protocol), already
  reported in the paper.
* **Wave 2**: extends the same query to
  `created:2026-04-08..2026-08-31` and takes the next
  `N_new ≥ 26` PRs in chronological order.  Ceiling
  `N_new = 100`.  Inclusion rule: identical mechanical rule as
  wave 1 (no filter for TG-handleability, no filter for
  self-contained repro, no filter for shape-only root cause).

## Freeze dates

| Marker | Date | Notes |
|---|---|---|
| Catalogue freeze (analyser pin) | 2026-04-07 | commit `040f6f3` |
| Wave-1 inclusion window opens   | 2026-04-08 | first PR after freeze |
| Wave-1 inclusion window closes  | wave-1 first 15 PRs ordered by `created_at` |
| Wave-2 inclusion window opens   | 2026-04-08 | immediately after wave-1 |
| Wave-2 inclusion window closes  | 2026-08-31 | camera-ready cutoff |

## Power calculation (target N)

At the observed wave-1 point estimates (TG 5/15, FakeTensorMode
2/15, Pytea 3/15), assuming the per-tool catch rates extrapolate,
the smallest second wave reaching Fisher-exact `p<0.05` is:

* TG vs FakeTensorMode: `N_new = 26` (total `N = 41`).
* TG vs Pytea:          `N_new = 77` (total `N = 92`).
* Either pair to reach Bayes factor `BF_10 ≥ 10` under uniform
  prior: `N_new = 56`.

The wave-2 collection target is therefore `N_new ≥ 26`, with
hard ceiling `N_new = 100`.

## Statistical analysis plan (pre-registered)

Pre-specified primary tests on the pooled wave-1 ∪ wave-2 sample:

1. Two-sided Fisher exact on TG vs FakeTensorMode catch-counts.
2. Two-sided Fisher exact on TG vs Pytea catch-counts.
3. Wilson 95% CI on the TG catch-rate.

Secondary, descriptive: per-tool RP-fire vs catch-of-upstream-bug,
off-axis-fire (false-positive) rate, Abstain breakdown by reason.

## Reference to wave-1 data

Wave-1 cached verdicts and per-PR breakdown are at
`experiments_v5/v8/real_bugs_unfiltered/manifest.json`.

## Status

Wave-2 collection is pre-registered for the camera-ready window;
the present submission reports the wave-1 N=15 result with
explicit non-separability at `α=0.05` and the wave-2 protocol
committed to this file.
