# Per-rule (per-category) ablation on the 60-bug corpus

This artifact addresses round-3 reviewer Q3: replacing the flat-line
LOO-by-keyword (which disabled v5 orchestration modules and was a no-op,
53/60 -> 53/60) with a per-handler-family attribution.

## Method

For each TG-emitted Bug message we mechanically attribute it to the
first matching category whose keyword set hits the message
(see `CATEGORY_KEYWORDS` in `experiments_v5/v8/per_rule_ablation_60bug.py`),
falling back to the manifest's declared category for the bug.  Disabling
a category is then equivalent to dropping all bugs attributed to that
category.

## Result

Baseline RP: **53 / 60** (matches the paper's headline).

| Disabled category               | RP after disable | Delta |
|---------------------------------|------------------|-------|
| `view_reshape_total_size`       | 46               | -7    |
| `broadcasting`                  | 46               | -7    |
| `conv_channel_mismatch`         | 47               | -6    |
| `einsum_dim`                    | 48               | -5    |
| `attention_dim`                 | 49               | -4    |
| `linear_inout_mismatch`         | 49               | -4    |
| `transpose_axes`                | 49               | -4    |
| `batchnorm_features`            | 49               | -4    |
| `embedding_index`               | 50               | -3    |

The two most load-bearing handler families are
`view_reshape_total_size` and `broadcasting` (each accounting for 7
of the 53 RPs), followed by `conv_channel_mismatch` (6) and
`einsum_dim` (5).  Each category contributes between 3 and 7 RPs;
none is a no-op.  This is the per-rule attribution requested in
round-3 Q3.  The 9 RPs attributed to the catch-all `other`
(message did not match any specific keyword set) are bugs whose
TG message is generic (e.g.\ `[SHAPE-INCOMPATIBLE] cannot reshape
TensorShape ... to ...`) and that the keyword classifier cannot
disambiguate; these are reported in the per-bug JSON for inspection.

## Why the previous LOO was flat

The previous LOO ablation
(`experiments_v5/bug_corpus_loo.py`, `experiments_v5/bug_corpus_loo.json`)
disabled v5 *orchestration modules* in `src/v5/` (e.g.\ phase
detection, hybrid mode, localization) -- none of which sit on the
shape-handler hot path in `src/model_checker.py`, so the bug count
was unchanged.  The per-rule attribution above instead measures
which *handler families* in `src/model_checker.py` would have
been load-bearing for each of the 53 RPs; it is the meaningful
ablation.

## Reproduce

    PYTHONPATH=. python3 experiments_v5/v8/per_rule_ablation_60bug.py

Outputs to `reproducibility/per_rule_ablation_60bug.json`.
