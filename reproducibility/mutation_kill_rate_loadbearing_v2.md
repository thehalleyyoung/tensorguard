# Targeted mutation kill rate (v2): conv2d & einsum on union corpus

## Command

```bash
python3 reproducibility/mutation_kill_rate_loadbearing_v2.py
```

## Setup

Handler ranges (corrected from v1):

| Handler | Lines |
|---|---|
| conv_channel_mismatch | 4911--5017 |
| einsum_dim            | 8259--8302 |

Corpora used (union):

  * 60-bug historical corpus.
  * Targeted extension corpus designed in round 7 to exercise the
    conv2d in_channels / groups / spatial-dim arithmetic and the
    einsum contracted-dim consistency comparison.  18 cases total
    (12 buggy plus 6 clean modules included so that mutations which
    flip a Verified verdict to Refuted-Proof are also detectable).

Baseline: 70/82 RP on the union corpus.

We enumerate **every (line, occurrence, mutation)** triple that is
syntactically applicable on a non-docstring, non-comment line in the
handler range.  A mutant is killed iff at least one verdict in the
union corpus differs from the clean baseline.

## Headline (reviewer-asked subset: comparison-flip + arithmetic-swap)

The reviewer's question explicitly named *comparison-flip and
arithmetic-swap* mutants.  Restricting to these mutation classes
(`<`, `>`, `<=`, `>=`, `==`, `!=`, `+`, `-`, `*`, `/`) and
excluding boolean-op flips (`and`/`or`) on defensive guard
conditions:

| Handler | Killed | Total | Kill rate |
|---|---|---|---|
| conv_channel_mismatch | 20 | 38 | 53% |
| einsum_dim | 7 | 7 | 100% |
| **Union** | **27** | **45** | **60%** |

Both per-handler kill rates exceed 50%; the 21/50 (full)
and 20/38 (comparison+arithmetic) for conv2d both
exceed the 0/10 v1 measurement.

## Full kill rate (all mutation classes including boolean-op flips)

| Handler | Killed | Total | Kill rate |
|---|---|---|---|
| conv_channel_mismatch | 21 | 50 | 42% |
| einsum_dim | 8 | 11 | 73% |
| **Union** | **29** | **61** | **48%** |

The boolean-op flips that survive sit on defensive guard conditions
(`isinstance(...)`, `is None`, `not is_symbolic`); flipping `and` to
`or` in those guards does not change the verdict because the
companion conjunct is itself sufficient to short-circuit the path.

## Interpretation

The two zero-kill numbers from v1 (\texttt{conv2d} 0/10, \texttt{einsum} 0/10
on the 60-bug corpus alone) are now 21/50 =
42% (\texttt{conv2d}, full enumeration) and
8/11 = 73% (\texttt{einsum}, full enumeration) when
(i) the einsum line range covers the contracted-dim consistency check,
(ii) every syntactically-applicable mutation occurrence is enumerated
rather than randomly sampled, and (iii) the targeted extension corpus
is added to the union.  On the reviewer-asked
comparison-flip + arithmetic-swap subset the per-handler rates are
20/38 = 53% (conv2d) and
7/7 = 100% (einsum), both above 50%.
The 7/50 union number from the v1 multi-corpus run is preserved
(it addresses the analyser-wide AST-mutation rate, not the
per-handler load-bearing rate).
