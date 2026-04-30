# 60-Bug Historical Corpus — Rule-Development Holdout Protocol

This document responds to round-1 reviewer Weakness 2 and Question 2.

## Claim

The TG v5 rule catalogue was assembled by enumerating
documented PyTorch operators from `torch.nn.modules` and
`torch._refs`, **not** by examining the 60-bug historical
corpus's `forward` bodies.  No bug class's `forward` was
inspected at the AST level before the operator-handler set in
`src/v5/` was frozen.

This claim is supported by:

1. The list of operators implemented in `src/v5/*.py` is a
   strict subset of the operators returned by
   `dir(torch.nn.modules)` ∪ `dir(torch._refs)` as of
   torch 2.9.1 — it contains no operator that occurs only in
   the 60-bug corpus.
2. The 60-bug corpus was assembled by keyword search over
   `pytorch/pytorch` issues (see `bug_corpus_protocol.md` §1)
   *before* the v5 catalogue was finalised.
3. The corpus is content-addressed by SHA-256 of its
   manifest at `experiments_v5/bug_corpus_manifest.json`;
   any rule added in response to a TG-miss on this corpus
   would invalidate the manifest and is detectable by
   `experiments_v5/v8/verify_corpus_freeze.py`.

## Leave-one-category-out (LOO) holdout

The 60 bugs span 10 categories
(`attention_dim`, `broadcasting`, `view_reshape_total_size`,
`conv_channel_mismatch`, `linear_inout_mismatch`, `einsum_dim`,
`transpose_axes`, `batchnorm_features`, `embedding_index`,
`other`).  For each category, we disable all
`src/v5/*.py` rules whose module name *or* operator name
contains the category label, then re-score the full 60-bug
corpus with TG.  The marginal contribution of each category's
rules is the drop in RP-count.

The harness is `experiments_v5/bug_corpus_loo.py`; outputs are
cached in `experiments_v5/bug_corpus_loo.json`.

This is a calibrated, weaker form of "true" rule-development
holdout (which would require time-machining the TG rule set
back to a state predating each bug's filing date — not
feasible given that `src/v5/` is a fresh repo subtree).  It is
sufficient to bound the maximum overfit at the per-category
granularity.

## What this protocol does **not** establish

* It does not prove that no specific predicate inside a rule
  (e.g.\ a divisibility envelope) was tuned in response to a
  bug repro inspected during development.  We have no
  cryptographic proof of authoring history below the rule-file
  level.
* It does not extend the holdout to the upstream-faithful
  re-extracts of the 10-bug real corpus
  (`experiments_v5/v8/real_bugs_upstream/`); those are
  governed by `REAL_BUG_SELECTION_PROTOCOL.md`.

## Reproducibility

```
PYTHONPATH=. python3 experiments_v5/bug_corpus_loo.py
```

The script writes `experiments_v5/bug_corpus_loo.json` with
per-category disabled-rule lists, RP-counts, and silent-miss
deltas.  See `reproducibility/bug_corpus_loo.md`.
