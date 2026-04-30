# Real Public-Repo Bug Corpus — Selection Protocol (rb_001 … rb_010)

This document specifies the selection protocol for the 10-bug
"real public" corpus shipped at
`experiments_v5/v8/real_bug_corpus.json` and accompanying repros
at `experiments_v5/v8/real_bugs/rb_*.py`.  It exists to address
the round-1 reviewer's W2: *"the 10-bug real-public corpus … is
small, hand-curated, and possibly trained-on … no protocol is
given for blind selection (e.g., commit-time freeze, holdout from
rule-development)."*

## 1. Source pool

The pool of candidate bugs is GitHub fix-PRs and issue tracker
items with the `bug` label, restricted to:

* `huggingface/transformers`
* `huggingface/diffusers`
* `huggingface/peft`
* `EleutherAI/gpt-neox`

filtered with the same shape-error keyword set used for the
60-bug historical PyTorch corpus
(`experiments_v5/bug_corpus_protocol.md` §2), namely:

```
"shape mismatch"     "size mismatch"
"is invalid for input of size"
"view"  "reshape"   "matmul"   "linear"   "conv"
"expand"   "permute"   "transpose"
RuntimeError +  one of {"shape", "size", "dim"}
```

dated **after** 2022-04-26 (the freeze date of the Pytea snapshot
we compare against, so the bugs are out-of-distribution for
Pytea's static catalogue) and **before** 2025-12-01 (the day the
corpus JSON was committed to this repo).

## 2. Inclusion criteria (applied **before** any TG run)

A candidate was included iff:

1. The bug is reproducible from a single `nn.Module` class as it
   shipped in the parent commit of the fix PR.
2. The bug surface is a static shape/divisibility/grad-flag
   mismatch (i.e.\ in scope for the calculus of §3 of the paper);
   data-dependent control-flow bugs are excluded.
3. A minimal CPU-only repro of $\le 60$ lines exists.
4. The fix PR was merged or accepted (so the bug is canonically
   confirmed by upstream maintainers, not just by us).

Criteria (1) – (4) were applied by inspecting the PR / issue
diffs **without running TG on the bug**.

## 3. Holdout from rule development

* No file in `src/v5/` was modified after the commit that landed
  `experiments_v5/v8/real_bug_corpus.json`.  A bisection harness
  (`experiments_v5/v8/verify_corpus_freeze.py`, see §5) re-runs
  each `rb_*` repro against the pre-corpus tree and confirms the
  same RP-at-conf-0.99 verdict on $10/10$ items.
* No rule was added in response to a TG miss on these bugs.  In
  particular, the `view_total_size`, `linear_view_chain`, and
  `view_groups` rules used to produce the RPs all predate the
  corpus by at least one tagged release.

## 4. Per-bug provenance

The corpus JSON records, for each item:

* `github_url` — the originating issue (open or closed)
* `pr_url` — the fix PR (when available)
* `source_repo`, `model` — the canonical name of the buggy class
* `filed_date` — date the issue was filed (so the reader can
  verify *temporal precedence* over our rule-set)
* `description` — narrative root cause
* `substitution_note` — concrete integer dims used in the repro,
  taken verbatim from the upstream error message where present
* `expected_error_substring` — the runtime error the repro
  triggers under PyTorch 2.9.1, used as the ground-truth oracle

## 5. Reproducibility script

`experiments_v5/v8/verify_real_bugs.py` re-runs all 10 repros
and asserts (a) the upstream `RuntimeError` is raised at runtime
and (b) TG returns RP at confidence $\ge 0.99$ on the static
class.  `experiments_v5/v8/verify_corpus_freeze.py` performs (b)
against the pre-corpus tree as a second, independent check that
the corpus is not rule-tuned.

## 6. Honest caveats

* **Sample size.** Ten bugs is small.  We have flagged
  "expand to $\ge 30$ bugs from independent fix-PR mining" in
  `.comet_neurips/self_obligations.md`.
* **Hand-minimised repros.** The repros were minimised by us, not
  by upstream.  We have preserved every `view`/`reshape` chain
  on the path between input and the failing op.
* **Selection bias.** Bugs that are obvious to an experienced
  maintainer at PR review time are over-represented; bugs that
  only manifest under exotic configurations (`bf16`,
  `accelerate` zero-2/3, `torch.compile`) are
  under-represented.  This is a real limitation of the corpus,
  not of TG.
