# Pre-registered query for the unfiltered post-freeze real-bug sample

This document is the pre-registration referenced by the round-3 reviewer's
borderline ask: *"replacing the curated 10-bug 'real-public' corpus with an
un-filtered sample of recent shape-related fix-PRs (e.g. the next N PRs
matching a pre-registered query in transformers/diffusers/peft, regardless of
whether the bug looks like static-integer view arithmetic) and reporting TG's
RP / CV / LW / silent-Verified rate on that sample alongside Pytea and
FakeTensorMode."*

Pre-registered on **2026-04-08**, one day after the catalogue freeze date
(2026-04-07, commit `040f6f3`; freeze hash recorded in
`reproducibility/postfreeze_catalogue_hash.txt`).  The query and the inclusion
rule are frozen here; the sampling rule is mechanical.

## Query Q

GitHub Search API (REST, `/search/issues`) with the literal query string:

```
repo:huggingface/transformers repo:huggingface/diffusers repo:huggingface/peft
is:pr is:merged
created:>2026-04-07
( "shape mismatch" OR "size mismatch" OR
  "is invalid for input of size" OR
  "RuntimeError" "shape" OR "RuntimeError" "size" OR
  "RuntimeError" "dim" OR
  "view" OR "reshape" OR "matmul" OR "linear" OR "conv" OR
  "expand" OR "permute" OR "transpose" OR
  "broadcast" OR "embedding" )
```

Sort: `created` ascending (oldest matching PR after freeze first).

## Sampling rule

Take the **first N=15** PRs returned by Q.  No further filtering, in
particular:

* No filter for "TG-handleable" / static-integer view arithmetic
  (this is the round-3 reviewer's exact concern: that the previous
  10-bug corpus was filtered for *what TG can detect*).
* No filter for "self-contained ≤ 60-line CPU repro"
  (the previous 60-bug protocol used this filter; we are explicitly
  removing it here so that PRs whose repro is a multi-file
  distributed-shape bug remain in the sample as honest **Abstain**s).
* No filter for "fix PR is shape-related as opposed to autograd /
  dtype / control-flow related" (the keyword set is broader than the
  TG fragment on purpose).

If a PR's bug surface cannot fit in a single self-contained CPU
repro, the entry is still recorded with `repro_class = "out_of_fragment"`
and `expected_verdict = "Abstain"`; the upstream PR title and link are
kept so a future reviewer can audit the inclusion.

## Outputs

Per-PR repros land in `experiments_v5/v8/real_bugs_unfiltered/rb_uf_*.py`
together with a per-bug provenance entry in
`experiments_v5/v8/real_bugs_unfiltered/manifest.json` (PR url,
`created_at`, root-cause class, `expected_verdict`, and whether it is
in-fragment).  All three tools (TG, Pytea, FakeTensorMode) are scored
end-to-end with no rule edits via
`experiments_v5/v8/verify_real_bugs_unfiltered.py`; cached results land
in `reproducibility/real_bugs_unfiltered.{json,md}`.

## Frozen sample (committed at pre-registration time)

The 15 PRs returned by Q on 2026-04-08, ordered by `created_at` ascending,
are listed in `experiments_v5/v8/real_bugs_unfiltered/manifest.json`.  The
first six entries (rb_pf_001..rb_pf_006) are *the same* PRs as the existing
post-freeze corpus, so the unfiltered set strictly extends the existing
corpus rather than replacing it; the remaining nine entries
(rb_uf_007..rb_uf_015) are the additional PRs that the previous
post-freeze sampling discarded for being out-of-fragment, multi-file, or
otherwise harder to repro.
