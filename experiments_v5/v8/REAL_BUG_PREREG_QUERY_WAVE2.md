# Pre-registered query for the unfiltered post-freeze real-bug sample — Wave 2

This document is the wave-2 pre-registration that extends the existing
wave-1 protocol (`REAL_BUG_PREREG_QUERY.md`) to a new freeze date,
camera-ready bound.  The scientific motivation is a power calculation
on the wave-1 result: at the observed point estimates (TG 5/15,
FakeTensorMode 2/15, Pytea 3/15), the smallest second wave that
reaches Fisher-exact `p<0.05` against `FakeTensorMode` is
`N_new = 26` additional PRs (total `N = 41`); against Pytea,
`N_new = 77` (total `N = 92`).  The wave-2 collection target is
therefore `N_new ≥ 26`, with a hard ceiling of `N_new = 100`.

## Pre-registration date and freeze

* Wave-1 freeze date (already in record): **2026-04-07**
  (commit `040f6f3`; freeze hash recorded in
  `reproducibility/postfreeze_catalogue_hash.txt`).
* Wave-2 collection window opens: **2026-04-08**
  (immediately after the wave-1 inclusion window).
* Wave-2 freeze date (collection cutoff): **2026-08-31**.
* Pre-registration date: written and committed at the point of this
  round's revision; the analyser binary is pinned to the wave-1
  catalogue freeze so wave-2 verdicts are produced with identical
  rules to wave-1 verdicts.

## Query Q (unchanged from wave 1)

GitHub Search API (REST, `/search/issues`) with the literal query
string:

```
repo:huggingface/transformers repo:huggingface/diffusers repo:huggingface/peft
is:pr is:merged
created:2026-04-08..2026-08-31
( "shape mismatch" OR "size mismatch" OR
  "is invalid for input of size" OR
  "RuntimeError" "shape" OR "RuntimeError" "size" OR
  "RuntimeError" "dim" OR
  "view" OR "reshape" OR "matmul" OR "linear" OR "conv" OR
  "expand" OR "permute" OR "transpose" OR
  "broadcast" OR "embedding" )
```

Sort: `created` ascending (oldest matching PR after wave-1 freeze
first).

## Sampling rule (mechanical, identical to wave 1)

Take the **next `N_new`** PRs returned by Q after the 15 PRs already
included in wave 1, where `N_new` is the smallest integer
`≥ 26` for which the chronologically-ordered query returns
`N_new` non-duplicate PRs by the wave-2 freeze date, capped at
`N_new = 100`.  No further filtering, in particular:

* No filter for "TG-handleable" / static-integer view arithmetic.
* No filter for "self-contained ≤ 60-line CPU repro".
* No filter for "fix PR is shape-related as opposed to autograd /
  dtype / control-flow".

If a PR's bug surface cannot fit in a single self-contained CPU
repro, the entry is still recorded with `repro_class =
"out_of_fragment"` and `expected_verdict = "Abstain"`; the upstream
PR title and link are kept so a future reviewer can audit the
inclusion.

## Statistical analysis plan (pre-registered)

Pre-specified primary tests on the pooled wave-1 ∪ wave-2 sample:

1. Two-sided Fisher exact on TG vs FakeTensorMode catch-counts.
2. Two-sided Fisher exact on TG vs Pytea catch-counts.
3. Wilson 95% CI on the TG catch-rate.

Secondary, descriptive: per-tool RP-fire vs catch-of-upstream-bug
breakdown; off-axis-fire (false-positive) rate; Abstain breakdown
by reason.  No further hypothesis tests are pre-registered.

## Outputs

* Per-PR repros land under `experiments_v5/v8/real_bugs_unfiltered/`
  with a per-bug provenance entry in
  `experiments_v5/v8/real_bugs_unfiltered/manifest.json` (PR url,
  `created_at`, root-cause class, `expected_verdict`, in-fragment).
* All three tools (TG, Pytea, FakeTensorMode) are scored end-to-end
  with no rule edits via the wave-1 verifier.
* Cached results land in `reproducibility/postfreeze_second_wave_results.{json,md}`.

## Status at submission time

Wave 2 data collection is pre-registered for the camera-ready
window.  At the time of initial submission, the wave-1 result
(N=15) is reported in the body with the explicit statement that
the unfiltered head-to-head is not separable at `α=0.05` on
`N=15`, and the wave-2 protocol is committed to the repository at
this filename.
