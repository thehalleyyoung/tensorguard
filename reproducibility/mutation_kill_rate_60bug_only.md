# Targeted mutation kill rate (60-bug corpus only): conv2d & einsum

## Command

```bash
python3 reproducibility/mutation_kill_rate_60bug_only.py
```

## Setup

Handler ranges:

| Handler | Lines |
|---|---|
| conv_channel_mismatch | 4911--5017 |
| einsum_dim            | 8259--8302 |

Corpus used (baseline only):

  * 60-bug historical corpus alone, with NO targeted extension.
  * This measurement provides the regressor-alone calibration anchor,
    unadjusted by the 18-case targeted extension.

Baseline: 53/60 RP on the 60-bug corpus.

We enumerate **every (line, occurrence, mutation)** triple that is
syntactically applicable on a non-docstring, non-comment line in the
handler range.  A mutant is killed iff at least one verdict in the
corpus differs from the clean baseline.

## Headline (full mutation class enumeration)

All mutation classes across all syntactic occurrences:

| Handler | Killed | Total | Kill rate |
|---|---|---|---|
| conv_channel_mismatch | 1 | 50 | 2% |
| einsum_dim | 0 | 11 | 0% |
| **Union** | **1** | **61** | **2%** |

## Subset: comparison-flip + arithmetic-swap only

Restricting to comparison-flip and arithmetic-swap mutation classes
(`<`, `>`, `<=`, `>=`, `==`, `!=`, `+`, `-`, `*`, `/`) and
excluding boolean-op flips (`and`/`or`):

| Handler | Killed | Total | Kill rate |
|---|---|---|---|
| conv_channel_mismatch | 1 | 38 | 3% |
| einsum_dim | 0 | 7 | 0% |
| **Union** | **1** | **45** | **2%** |

## Interpretation

This measurement represents the **regressor-alone baseline**: the mutation
kill rate on the original 60-bug corpus without the targeted extension 
corpus. These numbers provide the anchor against which to contrast the
load-bearing-extension union results.

**Full enumeration:**
  * conv2d: 1/50 = 2%
  * einsum: 0/11 = 0%

**Comparison-flip + arithmetic-swap subset:**
  * conv2d: 1/38 = 3%
  * einsum: 0/7 = 0%

The 60-bug-corpus-only baseline allows direct measurement of load-bearing
regressor effectiveness on the historical bug suite, independent of any
targeted-extension amplification effects. Comparison with the union corpus
numbers reveals how much additional detection capability the targeted
extension provides for these specific handlers.
