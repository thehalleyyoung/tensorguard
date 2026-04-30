# Multi-corpus mutation kill rates

## Command

```bash
python3.11 reproducibility/mutation_kill_rate_corpora.py
```

## Mutation operators (same as mutation_kill_rate_60bug.py)

- M1: comparison flip (`<`->`>`, `<=`->`>=`, `==`->`!=`)
- M2: boolean operator flip (`and`->`or`)
- M3: arithmetic op swap (`+`->`-`, `*`->`+`)
- M4: small-integer constant `+1` (constants 0..4)
- M5: boolean constant flip (`True`->`False`)

## Corpora

- **60-bug**: experiments_v5/v5_bug_corpus.jsonl (60 items)
- **488-block**: experiments_v5/v5_block_corpus.jsonl (stratified sample: 50 items)
- **25-stress**: experiments_v5/v8/hybrid_falsify/blocks/ (25 items)

## Results

| Metric | 60-bug | 488-block | 25-stress | **Union** |
|--------|-------:|----------:|----------:|----------:|
| Killed | 3 | 7 | 5 | **7** |
| Total | 50 | 50 | 50 | 50 |
| **Kill rate** | **6.0%** | **14.0%** | **10.0%** | **14.0%** |

- Mutants with real mutation: 50
- Strong kills (load errors): 0
- Clean baseline RP: 60-bug=53/60, 488-block=33/50, 25-stress=16/25

## Interpretation

The **best-of (union) kill rate** is **14.0%** (7/50 mutants), meaning that these mutants trigger at least one verdict change across the three corpora. 
This substantially improves over the 60-bug-only kill rate of 6.0%, demonstrating that the expanded corpus provides better mutation coverage.

## Paper claim (W4)

Round-4 W5 requested mutation-testing data beyond the hand-picked four-fault TCB exposure. 
This experiment extends the original 60-bug corpus (6% kill rate) with the 488-block and 25-stress corpora, 
yielding a union kill rate of **14.0%**. This provides a more robust automated lower bound 
on analyzer-level fault detection capability.

## Notes

- The 488-block corpus used a stratified sample of 50 blocks (out of 488) for performance.
- Each mutant is scored in a fresh subprocess to avoid Z3 process-global state leakage.
- A mutant is 'killed' if at least one verdict changes from the clean baseline.
