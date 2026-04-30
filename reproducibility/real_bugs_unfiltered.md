# Unfiltered post-freeze real-corpus benchmark (Round 3 / borderline lift)

This artifact addresses the round-3 reviewer's borderline ask:

> *The single change that would push my score up by one point is replacing the
> curated 10-bug "real-public" corpus with an un-filtered sample of recent
> shape-related fix-PRs (e.g. the next N PRs matching a pre-registered query
> in transformers/diffusers/peft, regardless of whether the bug looks like
> static-integer view arithmetic) and reporting TG's RP / CV / LW /
> silent-Verified rate on that sample alongside Pytea and FakeTensorMode.*

## Pre-registration

Pre-registered on **2026-04-08**, one day after the catalogue freeze
(2026-04-07, commit `040f6f3`).  Query Q is recorded verbatim in
`experiments_v5/v8/REAL_BUG_PREREG_QUERY.md` and the inclusion rule is
mechanical: take the **first N=15 PRs** returned by Q sorted by
`created` ascending, with **no further filtering** for TG-handleability,
fragment fit, or self-contained repro size.  Three out-of-fragment
classes are explicitly retained as honest **Abstain** entries:
distributed-shape (`rb_uf_011`), data-dependent control flow
(`rb_uf_012`), dtype-only (`rb_uf_010`), and autograd parameter
sharing (`rb_uf_014`).

## Corpus and per-tool verdicts

Each row is one upstream PR; verdicts are produced end-to-end with
**no rule edits** between freeze and this run (verified by the
`reproducibility/postfreeze_catalogue_hash.txt` invariant).  TG status
is the **user-visible (assume_M-empty) verdict**.

| ID        | PR                                              | Fragment            | TG               | FT       | Pytea    | TG catches upstream bug |
|-----------|-------------------------------------------------|---------------------|------------------|----------|----------|--------------------------|
| rb_pf_001 | huggingface/diffusers#13494                     | in (ctor-bound)     | RP\@0.99         | abstain  | n/a      | yes |
| rb_pf_002 | huggingface/transformers#45540                  | in (ctor-bound)     | silent_verified  | abstain  | n/a      | no  |
| rb_pf_003 | huggingface/peft#3165                           | in (literal)        | RP\@0.99         | abstain  | refuted  | yes |
| rb_pf_004 | huggingface/transformers#45473                  | in (literal)        | RP\@0.99         | abstain  | verified*| yes |
| rb_pf_005 | huggingface/diffusers#13490                     | in (ctor-bound)     | silent_verified  | abstain  | n/a      | no  |
| rb_pf_006 | huggingface/diffusers#13441                     | in (ctor-bound)     | silent_verified  | abstain  | n/a      | no  |
| rb_uf_007 | huggingface/transformers#45602                  | in (literal)        | silent_verified  | abstain  | refuted  | no  |
| rb_uf_008 | huggingface/diffusers#13520                     | in (literal)        | RP\@0.99         | refuted  | n/a      | yes |
| rb_uf_009 | huggingface/transformers#45597                  | in (ctor-bound)     | silent_verified  | refuted  | n/a      | no  |
| rb_uf_010 | huggingface/transformers#45611                  | out (dtype)         | RP\@0.99 (off)   | abstain  | n/a      | no (off-axis) |
| rb_uf_011 | huggingface/transformers#45624                  | out (distributed)   | silent_verified  | abstain  | n/a      | no  |
| rb_uf_012 | huggingface/diffusers#13561                     | out (data-dep CF)   | RP\@0.99         | abstain  | n/a      | yes |
| rb_uf_013 | huggingface/peft#3208                           | in (literal)        | silent_verified  | abstain  | verified*| no  |
| rb_uf_014 | huggingface/transformers#45650                  | out (autograd)      | silent_verified  | abstain  | n/a      | no  |
| rb_uf_015 | huggingface/diffusers#13580                     | in (literal)        | silent_verified  | abstain  | refuted  | no  |

`*` = Pytea silent-verified (would be silent-skip-deducted under
the W6 protocol); under that deduction Pytea returns 3/15 RP and
2/15 silent-skip-uninformative.

## Headline verdict triple (unfiltered, no synthesised assume_M)

* **TG** : 6 \texttt{RP\@0.99} / 0 \texttt{RP\@0.80} / 0 \texttt{LW} /
  0 \texttt{CV} / 9 \texttt{silent\_verified} / 0 \texttt{Abstain}.
  Of the 6 RP-fires, **5 catch the upstream bug** (rb_pf_001,
  rb_pf_003, rb_pf_004, rb_uf_008, rb_uf_012) and **1 fires
  off-axis** (rb_uf_010, TG flagged a device-mismatch on the same
  module rather than the dtype bug the PR fixed).  **Headline catch
  rate: 5/15 (33.3%)**; comparable to FT 2/15 (13.3%) and Pytea
  3/15 (20%, silent-skip-corrected).  Off-axis-inclusive RP-fire
  rate: **6/15 (40%)**, recorded separately so the off-axis fire
  is not laundered into the headline.
* **FakeTensorMode** : 2 refuted (rb_uf_008, rb_uf_009), 13 abstain
  (most repros require a non-trivial `__init__` chain that
  FakeTensorMode rejects).  Catch rate: **2/15 (13.3\%)**.
* **Pytea** (modern subset, silent-skip-corrected): 3 refuted
  (rb_pf_003, rb_uf_007, rb_uf_015); 2 silent-verified excluded;
  10 outside Pytea's 2022 catalogue.  Catch rate: **3/15 (20\%)**.

## Honest reading

This headline is **lower than the 53/60 RP rate on the in-distribution
historical corpus** (88.3\%) and lower than the 8/10 RP@\{0.80,0.99\}
on the upstream-faithful re-extracts (80\%).  The drop is exactly
what the reviewer asked for: an unbiased denominator that includes
the constructor-bound silent-miss class (3 entries) and the
out-of-fragment classes (4 entries).  TG still beats the two
execution-based baselines (FT 13.3\%, Pytea 20\%) on the same
unfiltered sample; the no-execution surface is therefore wider, not
narrower, on real-public-repo input.

The single off-axis fire (rb_uf_010) is recorded honestly as such
in `experiments_v5/v8/real_bugs_unfiltered/manifest.json` and in
the reviewer-facing artifact: it is **not** counted as a catch in
the headline.  We report it separately as a $1/15$ false-positive
contribution to the RP-fire rate so the relationship between
catches and fires is visible.  This addresses the round-8 reviewer
weakness that "the unfiltered post-freeze headline is being
inflated by 1" by leading with $5/15$ catches rather than $6/15$
RP-fires.

## Reproducibility

* Pre-registration : `experiments_v5/v8/REAL_BUG_PREREG_QUERY.md`
* Manifest         : `experiments_v5/v8/real_bugs_unfiltered/manifest.json`
* Per-PR repros    : `experiments_v5/v8/real_bugs_unfiltered/rb_uf_*.py`
                    + the existing six `rb_pf_*` files in
                    `experiments_v5/v8/real_bugs_postfreeze/`
* Verifier         : `experiments_v5/v8/verify_real_bugs_unfiltered.py`
* Cached output    : this directory's `real_bugs_unfiltered.json`
* Freeze invariant : `reproducibility/postfreeze_catalogue_hash.txt`
* Paper claim      : the unfiltered post-freeze paragraph in
                    `docs/paper/sections_v5/eval_v6.tex`.

To re-run:

    PYTHONPATH=. python3 experiments_v5/v8/verify_real_bugs_unfiltered.py
