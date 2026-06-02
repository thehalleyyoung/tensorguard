# Held-out blind split evaluation (pre-registered)

TensorGuard scored **once** on a held-out blind split of **186** cases (138 buggy / 48 clean) generated from parameter grids disjoint from the development corpus. Hypotheses were pre-registered in [`corpus_extended/PRE_REGISTRATION.md`](../corpus_extended/PRE_REGISTRATION.md) before scoring.

- manifest matches registration (`df881add2687...`): **True**

## `balanced` mode

| metric | value |
| --- | --- |
| confusion (tp / fp / tn / fn) | 138 / 0 / 48 / 0 |
| recall on decided (blind) | 1.0000 [0.9729, 1.0000] (n=138) |
| recall on decided (dev) | 1.0 |
| overfitting gap (blind vs dev) | 0.0 |
| H1 zero false positives | True |
| H2 recall >= 0.9 | True |
| H3 gap <= 0.1 | True |
| all hypotheses confirmed | True |

## `sound` mode

| metric | value |
| --- | --- |
| confusion (tp / fp / tn / fn) | 138 / 0 / 48 / 0 |
| recall on decided (blind) | 1.0000 [0.9729, 1.0000] (n=138) |
| recall on decided (dev) | 1.0 |
| overfitting gap (blind vs dev) | 0.0 |
| H1 zero false positives | True |
| H2 recall >= 0.9 | True |
| H3 gap <= 0.1 | True |
| all hypotheses confirmed | True |

**All pre-registered hypotheses confirmed in both modes: True.** The verifier holds up on held-out parameters it was never developed against, with no overfitting collapse and no clean-model false alarms.
