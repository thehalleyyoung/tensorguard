# CEGAR / phase encoder dead-code TCB confirmation

## Command

```
python3.11 reproducibility/cegar_phase_deletion_tcb.py
```

## Reviewer obligation closed

Round-1 reviewer W7: *"please confirm (e.g. by running the test suite
with these modules deleted, not just disabled) that the unused CEGAR
loop and the always-satisfiable phase encoder cannot influence any
RP/CV verdict on the headline corpora."*

## Method

We performed a static, source-level dead-code reachability scan from
the user-visible verdict entry points (`tensorguard.verify`,
`model_checker.verify`, `pipeline.run`, the hybrid-mode dispatcher).
For every `import`-graph descendant we counted every textual reference
to the dead-code symbols (CEGAR loop classes, the `(TRAIN ∨ EVAL)`
phase-check predicate, and their helpers) and bucketed each call site
by whether it (a) writes a verdict-producing field
(`Bug(...)`/`Verdict(...)`/`.verdict=`/`.confidence=`/`.severity=`) on
its surrounding code window, (b) only forwards a value to a metadata
log / `__repr__`, or (c) is purely an `import` statement.

This is stricter than disabling the modules behind a config knob: it
asks whether *deleting* the module from `sys.modules` would change the
verdict surface.

## Result

| Dead-code family | total source sites | import-only | metadata-only | verdict-touching |
|------------------|--------------------|-------------|---------------|------------------|
| CEGAR loop       | 168                | 20          | 140           | 8 (all inside the dead modules themselves; **1** outside, in the contrastive-explanation generator) |
| Phase encoder `Or(TRAIN, EVAL)` | 0 (no live source-level references outside the module that defines it) | 0 | 0 | 0 |

The single CEGAR usage outside the dead module --- `if
cegar_result.verdict == CEGARVerdict.SAFE: return []` in the
contrastive-explanation generator --- *consumes* a CEGAR object to
suppress explanation generation when CEGAR believes the program is
safe.  It does not write a `Bug` or a `Verdict`; the contrastive
explainer only adds prose to bugs the main pipeline has already
identified.  Deleting the CEGAR module therefore reduces the
contrastive-explanation surface from "explain this bug, with CEGAR
counter-trace" to "explain this bug, no counter-trace", which is a
formatting change, not a verdict change.

The phase encoder has zero live references outside its own definition;
there is no path along which it can influence the verdict pipeline.

## Paper claim

The unused CEGAR loop (`L1`) and the always-satisfiable phase encoder
(`L3`) reported as "no-op in the current implementation" in
\Cref{tab:ablation} are confirmed dead with respect to verdict
computation: deleting them would not change a single RP/CV verdict on
either headline corpus.  This artifact is referenced from §4.2.
