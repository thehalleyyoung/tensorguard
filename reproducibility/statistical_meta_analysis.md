# Cross-corpus statistical meta-analysis (Step 265)

This artifact summarizes heterogeneous TensorGuard evidence without naively pooling raw cases across real, synthetic, fuzzed, mutation, and stress-test distributions. The primary interval is a deterministic suite-level cluster bootstrap within each distribution.

- suites analyzed: **13**
- bootstrap resamples: **5000**
- naive global pooling allowed: **False**
- resampling unit: **suite, not individual cases**

## Distribution-stratified summaries

| distribution | suites | suite mean | bootstrap CI | suite range | case-weighted diagnostic |
| --- | ---: | ---: | --- | --- | ---: |
| clean_stress | 2 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 1.000 |
| mixed_decidable | 1 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 1.000 |
| natural_clean | 1 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 1.000 |
| real_minimized | 5 | 0.950 | [0.850, 1.000] | [0.750, 1.000] | 0.996 |
| real_semantic | 1 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 1.000 |
| synthetic_fuzz | 2 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 1.000 |
| synthetic_mutation | 1 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 1.000 |

## Metric summaries (still suite-level)

| metric | suites | suite mean | bootstrap CI | case-weighted diagnostic |
| --- | ---: | ---: | --- | ---: |
| bug_detection | 6 | 0.958 | [0.875, 1.000] | 0.999 |
| clean_acceptance | 6 | 1.000 | [1.000, 1.000] | 1.000 |
| decision | 1 | 1.000 | [1.000, 1.000] | 1.000 |

## Source suites

| suite | distribution | metric | success/trials | rate | source |
| --- | --- | --- | ---: | ---: | --- |
| extended-corpus bug recall | real_minimized | bug_detection | 153/153 | 1.000 | `reproducibility/corpus_extended_score.json` |
| extended-corpus clean specificity | real_minimized | clean_acceptance | 74/74 | 1.000 | `reproducibility/corpus_extended_score.json` |
| latent hard-recall bugs | real_minimized | bug_detection | 6/8 | 0.750 | `evaluation/hard_recall.json` |
| runtime-silent semantic bugs | real_semantic | bug_detection | 15/15 | 1.000 | `reproducibility/silent_bug_benchmark.json` |
| clean-model mutation kill rate | synthetic_mutation | bug_detection | 756/756 | 1.000 | `reproducibility/mutation_clean_models.json` |
| negative fuzz injected faults | synthetic_fuzz | bug_detection | 281/281 | 1.000 | `evaluation/neg_fuzz.json` |
| expected-decidable false-UNKNOWN corpus | mixed_decidable | decision | 86/86 | 1.000 | `evaluation/false_unknowns.json` |
| sound-mode clean false-positive hunt | clean_stress | clean_acceptance | 80/80 | 1.000 | `evaluation/sound_mode_fp.json` |
| 100+ clean-model false-alarm stress | clean_stress | clean_acceptance | 101/101 | 1.000 | `reproducibility/fp_stress_eval.json` |
| natural clean public-model sample | natural_clean | clean_acceptance | 174/174 | 1.000 | `reproducibility/natural_distribution_study.json` |
| differential clean fuzz | synthetic_fuzz | clean_acceptance | 200/200 | 1.000 | `evaluation/diff_fuzz.json` |
| same-case baseline full-corpus recall | real_minimized | bug_detection | 153/153 | 1.000 | `reproducibility/baseline_head_to_head.json` |
| same-case baseline full-corpus clean specificity | real_minimized | clean_acceptance | 74/74 | 1.000 | `reproducibility/baseline_head_to_head.json` |

real bugs, natural clean models, fuzzed modules, mutation tests, and stress corpora are different sampling distributions; a single raw pooled denominator would overweight the largest synthetic suite.
