# PyTorch Shape Bug Corpus Protocol (v5)

## Overview
This document describes the methodology used to extract 60 reproducible static tensor shape bugs from PyTorch's GitHub issue tracker. The corpus is organized as JSONL with individual Python reproducer files.

## Search Strategy

### Queries Used
The following search queries were executed against the pytorch/pytorch repository, limited to closed issues:

1. **Direct shape error keywords**:
   - `"shape mismatch"` (122 results)
   - `"size mismatch"` (125 results)
   - `"RuntimeError" "expected" "got"` (933 results)
   - `"broadcast" RuntimeError` (277 results)
   - `"view" RuntimeError "size"` (683 results)
   - `"Conv2d" "expected"` (597 results)
   - `"einsum" RuntimeError` (58 results)
   - `"reshape" RuntimeError` (349 results)

2. **Dimension-specific keywords**:
   - `"dimension" RuntimeError` (651 results)
   - `"Linear" "expected"` (1162 results)
   - `"BatchNorm" RuntimeError` (100 results)
   - `"matmul" RuntimeError` (318 results)

3. **Shape operation keywords**:
   - `"cannot reshape" RuntimeError` (19 results)
   - `"cannot view" RuntimeError` (20 results)
   - `"Dimensions must match" RuntimeError` (3 results)
   - `"input shape" RuntimeError` (453 results)
   - `"numel" mismatch` (56 results)
   - `"Conv2d" "channels" mismatch` (39 results)
   - `.unsqueeze RuntimeError` (320 results)
   - `.squeeze RuntimeError` (175 results)

4. **Temporal search (2019-2021)**:
   - `"view" "RuntimeError"` created:2019-01-01..2021-12-31 (381 results)
   - `"expand" "RuntimeError"` created:2019-01-01..2021-12-31 (112 results)
   - `"permute" "RuntimeError"` created:2019-01-01..2021-12-31 (74 results)
   - `"transpose" "RuntimeError"` created:2019-01-01..2021-12-31 (116 results)

**Total unique issues identified: 1,087**

## Inclusion Criteria

An issue was included in the corpus if:

1. **CPU-reproducible**: The bug can be reproduced using only CPU tensors with standard PyTorch ops
2. **Self-contained**: The reproducer requires only `torch` and `torch.nn`, no datasets or external data
3. **Minimal**: The reproducer is ≤40 lines of code
4. **Concrete shape error**: The issue describes a `RuntimeError` specifically related to tensor shapes, dimensions, or broadcasting
5. **No special backends**: Excludes GPU/CUDA, MPS, Distributed, TorchScript compilation (`torch.compile`), quantization
6. **Not numerical**: Excludes dtype mismatches that don't involve shape incompatibility
7. **Reproducible**: When executed as `python3 bug_repro.py`, produces the expected RuntimeError

## Exclusion Criteria

Issues were explicitly excluded if:

1. **Compile-specific**: Error only occurs with `torch.compile` or torch.jit
2. **GPU-only**: Requires CUDA, MPS, or other accelerators
3. **Distributed**: Requires DDP, FSDP, or NCCL
4. **Numerical precision**: Numerical drift, dtype casting, quantization issues
5. **Tensor creation**: Issues with device placement, dtype conversion at creation time
6. **Pull requests**: GitHub PRs instead of issues
7. **No concrete reproducer**: Issues describing bugs without minimal reproducible code
8. **Inplace operations**: Errors specific to in-place semantics rather than shape
9. **Sparse tensors**: Special sparse tensor semantics

## Extraction Process

For each candidate issue:

1. **Automated reading**: Used `github-mcp-server-issue_read` to fetch the full issue body
2. **Content analysis**: Searched for:
   - Concrete Python code snippets
   - RuntimeError stack traces with shape-related messages
   - Descriptions of dimension mismatches
   - Input/output shape specifications
3. **Repro construction**: 
   - If exact code existed in the issue, extracted and minimized it
   - Otherwise, constructed a minimal reproducer based on the error description
4. **Testing**: Each reproducer was run with `python3` to verify it produces the expected RuntimeError
5. **Categorization**: Assigned one of 10 bug categories based on the root cause

## Bug Categories

| Category | Count | Description |
|----------|-------|-------------|
| `conv_channel_mismatch` | 8 | Convolution input channels don't match kernel channels |
| `linear_inout_mismatch` | 7 | Linear layer input/output dimension incompatibility |
| `view_reshape_total_size` | 12 | view/reshape called with incompatible total element count |
| `broadcasting` | 10 | Broadcasting rules violated (shape dimensions incompatible) |
| `attention_dim` | 4 | Attention operations (SDPA, scaled_dot_product) with wrong dimensions |
| `einsum_dim` | 3 | Einstein summation with dimension mismatch |
| `transpose_axes` | 5 | transpose/permute with invalid axis specifications |
| `batchnorm_features` | 3 | BatchNorm feature count mismatch |
| `embedding_index` | 3 | Embedding with out-of-range indices |
| `other` | 5 | Other shape-related errors (repeat_interleave, squeeze, etc.) |

## Corpus Format

### JSONL Schema
Each line is a JSON object:
```json
{
  "id": "bug_NNN",
  "github_url": "https://github.com/pytorch/pytorch/issues/XXXXX",
  "title": "Brief title of the issue",
  "category": "view_reshape_total_size",
  "is_buggy": true,
  "description": "1-2 sentence summary of the bug",
  "repro_file": "experiments_v5/bug_repros/bug_NNN_*.py",
  "expected_error_substring": "exact substring expected in RuntimeError"
}
```

### Repro File Structure
Each `bug_repros/bug_NNN_*.py` file contains:

```python
"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/XXXXX
Expected Error: RuntimeError: [exact error message]
"""

import torch
import torch.nn as nn

INPUT_SHAPES = {
    'input': (batch_size, height, width, ...),
    ...
}

class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer definitions
    
    def forward(self, x):
        # Code that triggers the shape mismatch
        return result

if __name__ == '__main__':
    try:
        m = BuggyModule()
        x = torch.randn(INPUT_SHAPES['input'])
        output = m(x)
        print("ERROR: No exception was raised!")
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
```

## Corpus Statistics

- **Total bugs collected**: 60
- **Verified bugs** (reproduction tested): 60
- **Pass rate**: 100% of retained bugs
- **Date range of issues**: 2018-2024
- **Average reproducer size**: ~20 lines
- **Categories represented**: All 10 categories

### Breakdown by Category
- `view_reshape_total_size`: 12 (20%)
- `broadcasting`: 10 (17%)
- `conv_channel_mismatch`: 8 (13%)
- `transpose_axes`: 5 (8%)
- `other`: 5 (8%)
- `linear_inout_mismatch`: 7 (12%)
- `attention_dim`: 4 (7%)
- `einsum_dim`: 3 (5%)
- `batchnorm_features`: 3 (5%)
- `embedding_index`: 3 (5%)

## Limitations

1. **GitHub API rate limiting**: Initial searches were rate-limited after ~1000 results per query. Full enumeration of all shape bugs is not possible without pagination support.

2. **Search precision**: Many queries returned high false-positive rates (e.g., "RuntimeError expected got" matched 933 issues, most not shape-related). Manual filtering was required.

3. **Temporal bias**: Older issues (2018-2019) are underrepresented due to pagination limitations. Most corpus issues are from 2021+.

4. **Compile-heavy corpus**: PyTorch development shifted heavily toward `torch.compile` validation. ~30-40% of recent issues are compile-specific and were excluded.

5. **GPU-centric bugs**: Many modern bugs are GPU/distributed-specific. CPU-only reproducible bugs represent a minority of the issue tracker.

6. **Issue quality**: Some issues lack concrete reproducer code. These required synthetic construction based on error descriptions, which may not perfectly reproduce the original reporter's exact scenario.

## Validation Notes

- All 60 repro files were tested with `python3` on CPU
- Expected error substrings verified to appear in `stderr` or `stdout`
- No GPU/CUDA required for any reproducer
- All reproducers use only standard PyTorch modules: `torch`, `torch.nn`, `torch.nn.functional`
- No external dependencies (no transformers, vision, etc.)

## Files Generated

1. **v5_bug_corpus.jsonl**: 60-line JSONL file with bug metadata
2. **bug_repros/bug_NNN_*.py**: 60 individual reproducer Python scripts
3. **bug_corpus_protocol.md**: This documentation file

## Future Work

- Automated continuous integration to re-verify corpus as PyTorch versions advance
- Coverage expansion to huggingface/transformers and pytorch/vision repos
- Integration with TensorGuard's test suite for bug detection
- Analysis of bug fix patterns to improve static analysis

## References

- PyTorch GitHub: https://github.com/pytorch/pytorch
- Issues identified via: `pytorch/pytorch` repository issue tracker (closed issues)
- Search tool: GitHub Search API via `github-mcp-server-search_issues`
- Issue reader: `github-mcp-server-issue_read`
