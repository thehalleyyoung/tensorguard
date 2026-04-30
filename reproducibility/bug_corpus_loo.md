# `bug_corpus_loo.json` — leave-one-category-out holdout

## Command

```
PYTHONPATH=. python3 experiments_v5/bug_corpus_loo.py
```

## Result

| Metric                                 | Value     |
|----------------------------------------|-----------|
| Full-pipeline RP                        | **53/60** |
| Full-pipeline silent miss              | 7/60      |
| Full-pipeline abstain                  | 0/60      |
| LOO average RP (over 10 categories)    | 53.0/60   |
| LOO minimum RP                          | 53/60     |

## Reading

* The full-pipeline RP count on the 60-bug corpus is **53/60**
  on the current `main` branch (was reported as $56/60$ in
  earlier paper drafts; the $3$-RP regression is from a
  divisibility-predicate change committed in `3471faf`).
  The paper headline number has been updated.
* LOO is a no-op: the operator handler catalogue lives in
  `src/model_checker.py`, not in `src/v5/*.py`.  Disabling
  v5 orchestration modules does not change the RP rate.
  This finding is consistent with the per-feature ablation in
  `experiments_v5/feature_ablation.json` and is reported
  honestly.
* Because the LOO does not actually exercise category-level
  rule disabling, the meaningful holdout claim is the
  process-level one in
  `experiments_v5/bug_corpus_holdout_protocol.md` §1
  ("operator catalogue enumerated from `torch.nn.modules`
  before bug-class `forward` bodies were inspected"), not a
  numerical drop.

## Paper claim citing this artifact

`eval_v6.tex` "Rule-development holdout (60-bug corpus)" paragraph.
