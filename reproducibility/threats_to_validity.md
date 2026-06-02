# Threats to validity (Step 124, generated from abstention + FP data)

Every threat below is quantified by the committed abstention and false-positive artifacts and regenerated from them; residual-risk levels are computed from thresholds on those figures, not asserted.

Covering all four validity categories: **True** (3 low / 1 medium / 0 high residual risk).

## Construct validity — construct_abstention_masking

**Threat.** The three-valued verdict lets the tool abstain (UNKNOWN); if buggy models were disproportionately abstained on, recall would be inflated by silently dropping the hard cases.

**Evidence.**

- buggy_items_abstained_extended_corpus: `0`
- buggy_items_decided_extended_corpus: `153`
- max_abstention_rate_any_mode: `0.0`

**Mitigation.** Recall is reported on the full buggy set, not the decided subset, and the corpus records zero abstained buggy items, so no hard case is hidden behind UNKNOWN.

**Residual risk: LOW.**

## Conclusion validity — conclusion_false_alarm_undercount

**Threat.** A 'zero false positives' headline is only as strong as the clean corpus behind it; too few clean models, or a lenient oracle, would let real false alarms go uncounted.

**Evidence.**

- clean_models_false_positive_stress: `101`
- false_positives_observed_total: `0.0`
- false_alarm_rate_sound_mode: `0.0`

**Mitigation.** False alarms are counted across multiple independent clean corpora and cross-checked against a live eager-PyTorch differential oracle; the exact-binomial power analysis (Step 120) bounds the residual rate even at zero observed alarms.

**Residual risk: LOW.**

## External validity — external_synthetic_generalisation

**Threat.** Much of the corpus is programmatically generated; results might not transfer to hand-written, naturally-distributed models.

**Evidence.**

- natural_models_evaluated: `29`
- natural_coverage_sound_mode: `1.0`
- natural_recall_via_blind_split: `1.0`
- smallest_clean_sample: `29`
- smallest_clean_sample_individually_powered: `False`

**Mitigation.** A natural-distribution study and a pre-registered held-out blind split corroborate the synthetic results on hand-written models; where a single natural sample is small it is backed by the pooled clean-model bound rather than read in isolation.

**Residual risk: MEDIUM.**

## Internal validity — internal_overfitting

**Threat.** Detector thresholds and operator rules could be tuned to the development corpus, overstating performance on it.

**Evidence.**

- blind_split_overfitting_gap: `0.0`
- blind_split_recall: `1.0`

**Mitigation.** Hypotheses and the held-out split were pre-registered before evaluation; the observed dev-vs-blind overfitting gap is zero.

**Residual risk: LOW.**
