# Ambiguity taxonomy for mined PyTorch bug labels

This taxonomy is scoped to the offline metadata committed in
`experiments_v5/github_bug_mining/` and
`experiments_v5/provenance_bug_corpus/`. It deliberately does **not** assume that
third-party issue bodies, patches, or source files are redistributed in this
repository.

| Code | Meaning |
| --- | --- |
| `CLEAR_SIGNATURE` | The committed title or metadata contains an unambiguous tensor runtime signature. |
| `BODY_SIGNATURE_ONLY` | The frozen miner recorded the PyTorch runtime signature, but the committed title alone does not carry the full error text. |
| `NON_ENGLISH_OR_MINIMAL_TITLE` | The title is non-English, very short, or otherwise too sparse for root-cause adjudication. |
| `PR_METADATA_AMBIGUOUS` | The PR metadata is too generic to show the bug without the frozen signature match. |
| `GENERIC_INPUT_TYPE_PHRASE` | The phrase `Input type` is overloaded and may refer to non-tensor APIs. |
| `DEPENDENCY_OR_DOCS` | The title suggests dependency churn, documentation, API/schema text, or workflow metadata rather than a tensor execution bug. |
| `DATA_OR_PREPROCESS` | The title suggests input formatting, preprocessing, or dataset shape as the proximate cause. |
| `ENVIRONMENT_DEVICE` | The title ties the failure to device placement or accelerator environment. |
| `OVERLAP_SHAPE_DEVICE_DTYPE` | Multiple TensorGuard domains could plausibly explain the record. |
| `FIX_LINK_ONLY` | A PR or fix-style title indicates a possible repair but not enough standalone evidence for a gold positive label. |

The executable checker requires every ambiguity code used in
`annotations.jsonl` to be defined in `ambiguity_taxonomy.json`, and every defined
code to be exercised by the sample or explicitly remain unused in future
versions.
