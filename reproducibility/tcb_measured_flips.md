# TCB measured single-fault flip (60-bug corpus)

## Command

```
python3.11 reproducibility/tcb_measured_flips.py
```

## Clean baseline: RP **53/60** (silent 7, abstain 0, error 0)

## Per-fault measured flips

| Fault | TCB component | R3 exposure ceiling | Measured RP -> V flips |
|---|---|---|---|
| F1 | AST extractor | 0/60 | **0/60** |
| F2 | Backward verifier | 0/60 | **0/60** |
| F3 | Z3 dispatch | 2/60 | **0/60** |
| F4 | Analyser handler | 7/60 | **0/60** |

## Paper claim closed

Reviewer R4-W3/Q4 asked for the measured RP->V flip count under each deliberate single-fault build, not the exposure ceiling.  The measured flip equals the number of headline-corpus bugs the fault would silence; combined with the R3 exposure scan it gives a tight bracket [measured, exposure] on each TCB component's contribution to the 53/60 RP headline.
