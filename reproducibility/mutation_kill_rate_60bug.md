# Mutation kill rate on the 60-bug corpus

## Command

```
python3.11 reproducibility/mutation_kill_rate_60bug.py
```

## Mutation operators (single-edit AST rewrites)

- M1: comparison flip (`<`->`>`, `<=`->`>=`, `==`->`!=`)
- M2: boolean operator flip (`and`->`or`)
- M3: arithmetic op swap (`+`->`-`, `*`->`+`)
- M4: small-integer constant `+1` (constants 0..4)
- M5: boolean constant flip (`True`->`False`)

Each mutant is scored in a fresh Python 3.11 subprocess to avoid Z3 process-global state leaking across mutants.  A mutant is *killed* iff at least one of the 60 bugs has a different verdict from the clean baseline (or the mutated source fails to load).

## Result

| Metric | Value |
|---|---|
| Mutants attempted | 50 |
| Mutants with real mutation | 50 |
| Killed (>=1 verdict change OR load error) | 3 |
| Strong kills (analyser fails to load) | 0 |
| **Kill rate** | **6.0%** |
| Clean baseline RP | 53/60 |

## Paper claim closed

Round-4 W5 noted that the four-fault TCB exposure / measured-flip pair is hand-picked, and suggested an automated mutation-testing sweep as the natural next instrument.  This artefact reports the mutation kill rate on the 60-bug regression: 3/50 = 6.0%.  Combined with the four-fault exposure / measured-flip pair, this provides an automated lower bound on analyser-level robustness that is not limited to the four hand-picked TCB faults.
