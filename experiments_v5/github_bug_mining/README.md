# GitHub-Mined PyTorch Shape/Device Bug Dataset

A **frozen, labeled dataset of real PyTorch shape/device bugs** mined from public
GitHub history (issues + pull requests). It exists to measure TensorGuard's
precision/recall against a large, real-world fault distribution rather than a
hand-picked corpus.

Current snapshot: **2704 unique labeled bugs** (target was ≥ 500).

| Domain | Count |
| --- | --- |
| shape | 2394 |
| device | 310 |

| Category | Signature matched | Count |
| --- | --- | --- |
| `matmul_linear_mismatch` | `mat1 and mat2 shapes cannot be multiplied` | 400 |
| `dim_out_of_range` | `Dimension out of range` | 400 |
| `conv_channel_mismatch` | `Given groups=1, weight of size` | 399 |
| `view_reshape_total_size` | `is invalid for input of size` | 399 |
| `broadcast_mismatch` | `The size of tensor a` | 398 |
| `cat_stack_mismatch` | `Sizes of tensors must match except in dimension` | 398 |
| `device_mismatch` | `Expected all tensors to be on the same device` | 291 |
| `dtype_device_input_mismatch` | `Input type` (PyTorch-guarded) | 19 |

## How labels are grounded

Each record is mined by matching a **verbatim PyTorch runtime error signature**
in an issue/PR. Because the matched string *is* the PyTorch error message, the
category is high-confidence by construction: an issue containing
`mat1 and mat2 shapes cannot be multiplied` is, definitionally, a real
matmul/linear shape mismatch. This was spot-validated by re-fetching sampled
issue bodies and confirming 6/6 literally contain their matched signature.

## Layout

| Path | Purpose |
| --- | --- |
| `mine_github_bugs.py` | The miner (queries the GitHub Search API, dedups, labels, freezes). |
| `mined_bugs_dataset.jsonl` | One labeled bug per line: `source_url`, `repository`, `title`, `matched_signature`, `domain`, `category`, `state`, `created_at`. |
| `mined_bugs_manifest.json` | Frozen manifest: total, per-domain/category/signature breakdowns, mining date, and a `dataset_sha256` content hash. |
| `load.py` | Offline integrity + label-consistency checker. |

## Usage

```bash
# Verify the frozen dataset (offline):
python experiments_v5/github_bug_mining/load.py

# Re-mine from live GitHub (network; requires `gh auth`):
python experiments_v5/github_bug_mining/mine_github_bugs.py --target 500
```

## Reproducibility (network-qualified)

GitHub Search is a **live, growing index**, so re-mining later returns *more*
hits — the corpus only grows over time. The committed `mined_bugs_dataset.jsonl`
is therefore a **frozen snapshot**, content-addressed by `dataset_sha256` and
pinned by `tests/test_github_bug_mining.py` (which runs fully offline). Re-mining
is a network operation, analogous to this repo's other CUDA/HuggingFace/Lean
environment-qualified artifacts.
