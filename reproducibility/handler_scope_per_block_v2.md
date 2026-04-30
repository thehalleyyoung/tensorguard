# Refreshed soundness footprint after expanded Lean operator audit

Source: reproducibility/handler_scope_per_block.json
Command: python3.11 reproducibility/handler_scope_per_block.py
Inputs: experiments_v5/handler_soundness_scope.json (36 lean / 13 pp / 34 tested-only)

## Per-verdict partition of the 185 in-soundness verdicts (V+CV)

| Verdict | only Lean | Lean+pp | pp only | tested-only touch | out-of-scope | no handlers | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Verified | 24 | 9 | 0 | 10 | 9 | 5 | 57 |
| CV | 89 | 6 | 0 | 2 | 21 | 10 | 128 |
| **Total** | 113 | 15 | 0 | 12 | 30 | 15 | 185 |

**Audited footprint = 128/185 = 69.2%** (was 62/185 = 33.5% under previous 28-Lean / 16-pp / 35-tested partition).

Paper claim cited: §4 Calibrated framing and Table tab:soundness-footprint-185 (eval_v6.tex).
