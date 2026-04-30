# CEGAR and phase: zero-delta on every measured corpus

## Question

Round 1 W6: confirm CEGAR and phase-check are zero-delta on every
measured corpus, and position them as defensive features shipped
for future extension rather than as part of the ablation
contribution.

## Corpora checked

| Corpus | N | Δ (CEGAR off) | Δ (phase off) | Discriminating cases |
|---|---:|---:|---:|---:|
| 60-bug historical             |  60 | 0 | 0 | 0 |
| 488-block real-source         | 488 | 0 | 0 | 0 |
| N=15 unfiltered post-freeze   |  15 | 0 | 0 | 0 |
| 10-bug upstream-faithful real |  10 | 0 | 0 | 0 |
| 25-case stress benchmark      |  25 | 0 | 0 | 0 |

## Search for discriminating cases

`experiments_v5/`, `benchmarks/`, and `tests/` were searched for
fixtures explicitly designed to exercise CEGAR or phase
predicates, and the analyser was run with each feature both
enabled and disabled on every corpus listed above.  Zero cases
discriminate.

## Why

* CEGAR's internal predicates are stored as metadata but never
  surfaced as `Bug` objects (architectural gap between the CEGAR
  loop and the verdict pipeline).  Architectural fix is future
  work.
* The phase-check encodes `Or(TRAIN, EVAL)`, which is always
  satisfiable, so the predicate is trivially discharged on every
  measured input.

## Positioning

CEGAR and phase ship in the analyser as defensive features for
future extension (a future revision can refactor CEGAR to
emit `Bug` objects and tighten the phase predicate).  Both are
zero-delta on every corpus measured to date and contribute zero
discrimination to the headline numbers.  We position them in
Limitations as defensive/future-extension features, not as part
of the ablation contribution.

## Paper claims cited

* Eval section ablation paragraph.
* Per-feature stress benchmark table caption (already labelled
  *shipped, no-op*).
* Limitations paragraph (Round-1 W6: new sentence positioning
  CEGAR and phase as defensive features).
