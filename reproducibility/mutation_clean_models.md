# Mutation testing of clean models

We inject single, local bugs into the **130** validated-clean models of the stress and natural corpora using **5** mutation operators (Linear/Conv width bumps and an integer-dtype cast), keep only the **376** mutants that genuinely raise under eager PyTorch, and measure how many the verifier kills (returns UNSAFE).

| mode | killed | survived (SAFE) | survived (UNKNOWN) | kill rate [95% CI] |
| --- | --- | --- | --- | --- |
| sound | 376 | 0 | 0 | 1.0 [0.9899, 1.0] |
| balanced | 376 | 0 | 0 | 1.0 [0.9899, 1.0] |
| heuristic | 376 | 0 | 0 | 1.0 [0.9899, 1.0] |

Per-operator kill rate (sound mode):

| operator | domain | genuine bugs | killed | kill rate [95% CI] |
| --- | --- | --- | --- | --- |
| conv_in_bump | shape | 35 | 35 | 1.0 [0.9011, 1.0] |
| conv_out_bump | shape | 34 | 34 | 1.0 [0.8985, 1.0] |
| dtype_long_cast | dtype | 128 | 128 | 1.0 [0.9709, 1.0] |
| linear_in_bump | shape | 103 | 103 | 1.0 [0.964, 1.0] |
| linear_out_bump | shape | 76 | 76 | 1.0 [0.9519, 1.0] |

- sound mode never calls a genuine bug SAFE: **True**
- sound-mode kill rate (point): **1.0**

No genuine-bug mutant is reported SAFE in sound mode: every injected bug is either killed (UNSAFE) or explicitly abstained (UNKNOWN), so the verifier never silently passes an injected bug.
