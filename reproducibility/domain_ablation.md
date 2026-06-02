# Per-domain ablation: contribution of each abstract domain (Step 117)

Seed `20240604` — **300** labeled single-bug modules (60 per domain across shape, dtype, device, gradient, phase).

Leave-one-domain-out (LODO): `device`/`gradient` ablated via genuine runtime toggles, `shape`/`dtype` ablated at the report level (no runtime toggle on the always-on base view), `phase` recorded as diagnostic-only.

## Per-domain contribution to recall

| domain | full recall | LODO recall | marginal contribution | necessary |
| --- | --- | --- | --- | --- |
| shape | 60/60 (1.0) | 0/60 (0.0) | 1.0 | True |
| dtype | 60/60 (1.0) | 0/60 (0.0) | 1.0 | True |
| device | 60/60 (1.0) | 0/60 (0.0) | 1.0 | True |
| gradient | 60/60 (1.0) | 0/60 (0.0) | 1.0 | True |
| phase (diagnostic-only) | 0/60 (0.0) | — | — | — |

## Summary

- all verification domains reach full recall: **True**
- every verification domain is necessary (LODO recall drops to zero on its own bugs): **True**
- domains are orthogonal (ablating one does not reduce recall on the others): **True**
- phase is diagnostic-only (refutes nothing): **True**
- toggle/report cross-check agrees on all 600 device+gradient pairs: **True**
