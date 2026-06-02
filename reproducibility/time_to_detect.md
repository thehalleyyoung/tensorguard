# Per-bug time-to-detect: static vs first failing forward (Step 116)

Seed `20240603` — **400** buggy modules (every one rejected by the live torch dispatcher), drawn from the structured module-AST DSL.

Time-to-detect is measured in *operations* (hardware-independent), not wall-clock: how many forward ops must execute successfully before the bug manifests.

## Static verification (TensorGuard, sound mode)

- detect depth: **0** ops (flagged pre-execution)
- requires a constructed input: **False**
- requires execution: **False**
- caught (UNSAFE): **400** of 400; all at depth zero: **True** (Wilson 0.9905–1.0)

## Dynamic baseline (first failing forward op)

- requires a constructed input: **True**
- detect depth: min **0**, median **1.0**, mean **1.4375**, max **7** ops
- bugs that surface only after at least one successful op: **233** (fraction 0.5825)
- detect-depth histogram (depth: count): `{0: 167, 1: 63, 2: 69, 3: 52, 4: 34, 5: 9, 6: 5, 7: 1}`

## Comparison

- static is never later than dynamic: **True**
- static is strictly earlier on **233** modules
- operations saved before detection: median **1.0**, total **575**
