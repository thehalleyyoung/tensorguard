# Pre-registration: held-out blind split evaluation

This document records the hypotheses we commit to **before** scoring TensorGuard
on the held-out blind split. It exists to rule out overfitting and post-hoc
metric selection: the split is generated from parameter grids **disjoint** from
the development corpus (`corpus_extended/generators.py`), frozen and
content-addressed, and the predictions below were registered prior to running
`reproducibility/blind_split_eval.py`.

## Frozen artifact under test

- Split: `tensorguard-blind-split` v1.0.0
- Definition: `corpus_extended/blind_split.py` (186 cases, 138 buggy / 48 clean,
  9 families)
- Materialized: `corpus_extended/blind_cases/<id>.py`
- Manifest: `corpus_extended/blind_manifest.json`
- **Manifest SHA-256 (commitment):**
  `df881add26871538d6a5e8d552e8839af44240b9b79478a892dc7c6802e65dc3`
- Disjointness: every blind case id is prefixed `blind_` and uses parameter
  values absent from the dev grids, so the dev/blind id intersection is empty
  (checked in `tests/test_blind_split.py`).

## Tool under test (frozen, not tuned on this split)

- `src.api.verify_architecture` at the repository commit that introduces this
  split. The verifier and its stubs were authored against the dev corpus only;
  the blind split is scored exactly once.

## Pre-registered hypotheses

For both `balanced` and `sound` soundness modes, scored over all 186 blind
cases:

- **H1 (soundness).** Zero false positives on the 48 clean blind modules: no
  clean module is reported `UNSAFE`.
- **H2 (recall).** Recall on the 138 buggy blind cases, computed over cases on
  which a definite verdict is issued, is at least **0.90**.
- **H3 (no overfitting gap).** The blind-split recall-on-decided is within
  **0.10** (absolute) of the dev-corpus recall-on-decided reported by
  `reproducibility/corpus_extended_score.py`, i.e. performance does not collapse
  on held-out parameters.

The evaluation harness reports each hypothesis as confirmed or refuted from the
frozen split; results are recorded in `reproducibility/blind_split_eval.json`.
