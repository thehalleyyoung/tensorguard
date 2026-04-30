# TCB fault-injection footprint

## Command

```
python3 reproducibility/tcb_fault_injection_footprint.py
```

## Method

Each TCB-component fault F1-F4 is paired with a regex that detects whether a source exercises the construct it mis-handles.  Exposure = # items the construct appears in.
Exposure is an upper bound on the verdict-flip the fault could induce; the actual flip is bounded above by exposure and is zero whenever the bug path does not route through the faulty TCB component.

## Result

| Fault | TCB component | 60-bug exposure | 488-block exposure |
|---|---|---|---|
| F1: `view(*new_shape)` star-expansion mis-binding | AST extractor | 0/60 | 5/137 |
| F2: `Tensor.add_` mis-classified as out-of-place | Backward verifier | 0/60 | 18/137 |
| F3: cat/stack `dim=` negation flip | Z3 dispatch | 2/60 | 18/137 |
| F4: Conv2d output-formula off-by-one | Analyser handler | 7/60 | 19/137 |

Tightened upper bounds on the 60-bug corpus (restricted to bugs whose declared category routes through the faulty handler):

- F3 (cat-dim, restricted to cat-mediated bugs): 2/60.
- F4 (Conv2d, restricted to `conv_channel_mismatch`): 5/60.

## Paper claim closed

Round-3 reviewer W6 raised that the TCB statement covers the entire operational soundness story for the user-facing tool, and asked for an accounting of what survives if a TCB component is wrong.  This artefact bounds the verdict-flip a single deliberate TCB fault could cause on each headline corpus by exposure scan; the largest exposure on the 60-bug corpus is 7/60, which means the 53/60 RP headline could degrade by at most that many bugs under any single audited TCB fault.
