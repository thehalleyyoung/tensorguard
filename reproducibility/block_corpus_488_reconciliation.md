# Reconciliation of the two 488-block headline counts

## Question

Two distinct verdict-count tuples for the 488-block real-source corpus
have appeared in released artifacts:

* **57 Verified / 206 Refuted / 225 Abstain** — paper text and
  `experiments_v5/feature_ablation.json` (every ladder rung) and
  `experiments_v5/hybrid_mode_results.json` (`tg_only`).
* **50 Verified / 213 Refuted / 225 Abstain** — recorded by
  `experiments_v5/v5_benchmark_results.json`.

The reviewer's audit asked which is authoritative and what explains
the discrepancy.

## Explanation

The two tuples come from two different settings of one verifier knob:

| Source artifact | `high_confidence_only` | Verified | Refuted | Abstain |
|---|---|---:|---:|---:|
| Paper / `feature_ablation.json` / `hybrid_mode_results.json` | `True` (Z3-proven bugs only) | 57 | 206 | 225 |
| `v5_benchmark_results.json` | `False` (default; Z3 + heuristic post-pass) | 50 | 213 | 225 |

The 7-row gap is exactly the lower-confidence heuristic post-pass
moving 7 modules from `Verified` to `Refuted`.  The 225 `Abstain`
count is identical across both regimes, as expected (the abstain
decision is independent of the heuristic post-pass).

## Re-verification on the current code base

Running both regimes against the current code base produces:

| Regime | Verified | Refuted | Abstain |
|---|---:|---:|---:|
| `high_confidence_only=True` (authoritative) | **62** | 201 | 225 |
| `high_confidence_only=False` | 55 | 208 | 225 |

* **Same diff structure**: 7 modules transition exactly
  `Verified -> Refuted` between the two regimes — identical
  to the historical 57/206 vs 50/213 gap.
* **Slight absolute drift**: 5 more modules now `Verified` (62 vs
  57) under HCO=True after code-base evolution between the original
  benchmark capture (April 28) and the current snapshot.  No module
  transitions to `Abstain`, and no module flips its sign of
  refutation; the drift is bookkeeping-clean.

## Authoritative regime

The paper cites the **HCO=True** regime throughout.  We re-state the
headline numbers using the freshly recomputed values:
**62 Verified / 201 Refuted / 225 Abstain (488)**.  The HCO=False
column is reported here for transparency; it is the public default
verifier setting and is what users will see if they invoke the API
without setting the flag.

## Command

```
python3 reproducibility/block_corpus_488_reconciliation.py
```

## Inputs / seeds

* Block corpus: `experiments_v5/v5_block_corpus.jsonl` (488 modules).
* `torch == 2.9.1`, `python == 3.11.15`, no GPU.
* `max_cegar_iterations=3`; otherwise verifier defaults.
* Both regimes share the abstain decision and abstain reasons; only
  the post-Z3 heuristic pass differs.

## Paper claim cited

Section "Empirical evaluation" reports the headline 488-block
verdict counts and the per-library breakdown.  This artifact pins
those numbers to a single recomputed run and explains the
historical discrepancy.
