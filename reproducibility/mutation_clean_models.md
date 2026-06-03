# Mutation testing of clean models

We inject single, local bugs into the **275** validated-clean models of the stress and natural corpora using **5** mutation operators (Linear/Conv width bumps and an integer-dtype cast), keep only the **756** mutants that genuinely raise under eager PyTorch, automatically minimize each admitted mutant while preserving its real PyTorch exception signature, and measure how many the verifier kills (returns UNSAFE).

| mode | killed | survived (SAFE) | survived (UNKNOWN) | kill rate [95% CI] |
| --- | --- | --- | --- | --- |
| sound | 756 | 0 | 0 | 1.0 [0.9949, 1.0] |
| balanced | 756 | 0 | 0 | 1.0 [0.9949, 1.0] |
| heuristic | 756 | 0 | 0 | 1.0 [0.9949, 1.0] |

Automatic minimization (all admitted mutants):

| minimized | shrunk | removed logical lines | preserves failure signature | 1-line minimal |
| --- | --- | --- | --- | --- |
| 756 | 716 | 2990 | True | True |

Per-operator kill rate (sound mode):

| operator | domain | genuine bugs | killed | kill rate [95% CI] |
| --- | --- | --- | --- | --- |
| conv_in_bump | shape | 85 | 85 | 1.0 [0.9568, 1.0] |
| conv_out_bump | shape | 79 | 79 | 1.0 [0.9536, 1.0] |
| dtype_long_cast | dtype | 263 | 263 | 1.0 [0.9856, 1.0] |
| linear_in_bump | shape | 198 | 198 | 1.0 [0.981, 1.0] |
| linear_out_bump | shape | 131 | 131 | 1.0 [0.9715, 1.0] |

- sound mode never calls a genuine bug SAFE: **True**
- sound-mode kill rate (point): **1.0**

No genuine-bug mutant is reported SAFE in sound mode: every injected bug is either killed (UNSAFE) or explicitly abstained (UNKNOWN), so the verifier never silently passes an injected bug.
