# Proportional stratified resample (n=83) — tied-weight Verified subset

**Paper claim:** §4.4 stub-mocked tied-weight audit, Wilson-CI
headline; §6 Limitations / deployment-side false-Verified bound.

---

## (i) Exact command
```
PYTHONPATH=. python3.11 experiments_v5/stratified_resample_371.py
```
Outputs:
- `experiments_v5/stratified_resample_371.csv`
- `experiments_v5/stratified_resample_371_wilson.json`

## (ii) Seed / model / inputs
- Seed: `20260430` (deterministic stratified draw)
- Target sample size: `n_target = 80`, minimum `2` per stratum
- Population: 371 Verified tied-weight rows from
  `reproducibility/tied_weight_full_verdict_rows.json`
- Stratification: priority-order regex on class source body
  (same `HANDLER_TOKENS` table as
  `reproducibility/handler_scope_per_block.py`)
- Backend: pure CPU, PyTorch (no CUDA), one
  `loss.backward()` step per row on a stub-mocked input

## (iii) Resulting numbers
| Metric | Value |
|---|---|
| Population size | 371 |
| Sample size n | 83 |
| Instantiable + fwd + bwd OK runs | 47 |
| Silent-error rows | **2** |
| Global Wilson 95% CI | **[0.66%, 8.37%]** |
| Comparison vs. shortest-LoC original (n=31) Wilson upper | 13.32% |
| Comparison vs. companion stratified (n=39, 14 OK) Wilson upper | 21.53% |

### Per-stratum
| Family | Pop | Sampled | OK | Silent | Wilson CI |
|---|---|---|---|---|---|
| linear-only | 134 | 29 | 13 | **2** | [1.91%, 21.96%] |
| broadcast-elementwise | 133 | 29 | 24 | 0 | [0.00%, 11.70%] |
| embedding-family | 35 | 8 | 6 | 0 | [0.00%, 32.44%] |
| conv-family | 23 | 5 | 0 | 0 | [0.00%, 43.45%] |
| norm-family | 17 | 4 | 3 | 0 | [0.00%, 48.99%] |
| no_handler_detected | 17 | 4 | 0 | 0 | [0.00%, 48.99%] |
| attention-family | 8 | 2 | 1 | 0 | [0.00%, 65.76%] |
| reshape-only | 4 | 2 | 0 | 0 | [0.00%, 65.76%] |

### Silently-incorrect rows
Both rows are linear-only classification heads where the
analyser's first-order `requires_grad` lattice predicts a
`Verified` topology but the one-step backward delivers
gradient to zero of the required-grad leaf parameters
(loss-side reduction sits behind a detached projection):

1. `PPDocLayoutV3GlobalPointer`
   (`benchmarks/_corpus/transformers/src/transformers/models/pp_doclayout_v3/modeling_pp_doclayout_v3.py`)
2. `RobertaClassificationHead`
   (`benchmarks/_corpus/transformers/src/transformers/models/roberta/modular_roberta.py`)

## (iv) Paper claims this artifact backs

- **§4.4 stub-mocked tied-weight headline.** The proportional
  stratified resample replaces the prior shortest-LoC-first
  estimate (0/25, Wilson upper 13.32%) and the 5-per-stratum
  companion (0/14, Wilson upper 21.53%) as the most-powered
  silent-error envelope on the Verified tied-weight
  subpopulation. The paper now reports the
  $n{=}83$ figure 2/47, Wilson 95% CI [0.66%, 8.37%].
- **§6 deployment-side false-Verified bound.** The Wilson
  upper of 8.37% is the headline direct envelope on the
  Verified tied-weight subpopulation; the regex-screened
  product bound (≤3.0% on training-script population)
  is retained as a separate, narrower estimate keyed to
  the worst-case construct family.
- **Abstract.** The abstract reports both the
  Wilson-upper ≤8.37% on the resample and the regex-screened
  product bound ≤3.0%, in place of the prior
  single-number "≤3.0%" framing.
