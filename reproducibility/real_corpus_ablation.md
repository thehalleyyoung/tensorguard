# `real_corpus_ablation.json` — feature ablation on the upstream-faithful corpus

## Command

```
PYTHONPATH=. python3 experiments_v5/v8/real_corpus_ablation.py
```

## Result (10-bug `real_bugs_upstream` corpus)

| Feature disabled | RP@0.99 | RP@0.80 (cum.) | Silent verified |
|------------------|---------|----------------|-----------------|
| (none, baseline) | 5       | 8              | 2               |
| CEGAR            | 5       | 8              | 2               |
| Device-flag      | 5       | 8              | 2               |
| Phase            | 5       | 8              | 2               |
| Grad-flow        | 5       | 8              | 2               |
| Low-conf gating  | 5       | 8              | 2               |

The five-feature ladder is **flat** on the upstream-faithful real
corpus: each feature contributes nothing observable on these 10 bugs.
This is the strong form of the existing paper claim that the
auxiliary features only help on the synthetic 60-bug bug-corpus, and
not on the real-public-repo subset — extending the previously
60-bug-only ablation table to the real corpus.

## Paper claim citing this artifact

`eval_v6.tex` Real-corpus ablation paragraph: "the five-feature
ladder is flat on the upstream-faithful real corpus
(real_corpus_ablation.json)."
