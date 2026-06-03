# Labeling rubric for the mined PyTorch bug agreement audit

## Scope and honesty contract

This artifact measures agreement on a **dual-pass repository audit** over a
deterministic 32-record stratified sample from the frozen 2,704-record
GitHub-mined PyTorch bug corpus. It is not a human-subjects study and does not
claim independent external annotators. The purpose is narrower and auditable:
given only committed metadata (`source_url`, repository/title, matched PyTorch
runtime signature, category/domain, and provenance fields), can two review passes
apply the same inclusion and root-cause rubric consistently?

The original category label in `github_bug_mining` is mechanical: it is derived
from the matched runtime signature. This audit therefore does **not** report that
mechanical category as the main inter-rater result. The measured axes are the
judgment calls reviewers actually need before promoting mined rows into a
paper-facing gold split:

1. `include_decision`
   - `include`: enough committed evidence to treat the row as an in-scope
     TensorGuard-positive mined bug signal.
   - `defer`: plausible positive signal, but a gold-label split should refetch
     third-party context or inspect a fix before using it.
   - `exclude`: title/metadata suggests the signature match is generic,
     non-tensor, dependency-only, or otherwise out of scope.
2. `root_cause_family`
   - `shape_contract`: tensor rank, dimension, broadcast, concat, matmul,
     convolution, reshape, or indexing contract.
   - `device_placement`: CPU/GPU/MPS/XPU placement mismatch.
   - `dtype_device_contract`: dtype/device compatibility in a tensor kernel.
   - `data_or_preprocess`: input data shape, image size, tokenizer/preprocess, or
     batch construction is the visible driver.
   - `out_of_scope`: not a TensorGuard tensor bug from committed evidence.
   - `unknown_from_metadata`: committed metadata is insufficient to localize root
     cause.
3. `evidence_strength`
   - `strong`: title or committed metadata is enough for inclusion.
   - `moderate`: matched signature is strong, but title context is incomplete.
   - `weak`: generic or sparse metadata; defer/exclude unless later context
     justifies inclusion.

Every row must also carry at least one ambiguity code from
`ambiguity_taxonomy.json`. Disagreements are resolved by the adjudication fields
embedded in `annotations.jsonl` and rendered in `adjudication_log.md`.

## Reproduction

```bash
PYTHONPATH=. python3 experiments_v5/labeling_agreement/agreement.py
PYTHONPATH=. python3 experiments_v5/labeling_agreement/agreement.py --check
PYTHONPATH=. python3 -m pytest -q tests/test_labeling_agreement.py --no-header
```
