● Read brainstorm.prompt.md
  │ .comet_neurips/exploration/round_05/brainstorm.prompt.md
  └ 113 lines read

```
### Candidate 1: Theorem-footprint-restricted real-source rerun with new headline table
goal: Add a new evaluation mode that re-runs the 488-block real-source corpus restricted to verdicts whose entire derivation path lies strictly inside the Lean-or-pen-and-paper audited operator footprint, and emit a CSV+JSON artifact with per-block footprint classification (audited / tested-only / out-of-scope) plus aggregate {V, CV, RP, A} counts. This directly addresses the reviewer's borderline condition for a 6 by producing a strictly theorem-backed real-source headline number, lifting Soundness from 2→3 and Contribution from 3→4.
plan:
  - Locate the existing 488-block real-source eval driver under `experiments_v5/` or `real_benchmarks/` and the audited-operator list (Lean fragment + pen-and-paper handlers) referenced by `tab:soundness-footprint-185`.
  - Add a `--footprint-strict` flag to the eval driver that classifies every operator touched on each derivation path and assigns each block to {audited, tested-only-touched, oos-touched}; emit `experiments_v5/footprint_strict_488.csv` with columns `block_id, verdict, footprint_class, ops_touched`.
  - Add a small aggregator script that reads the CSV and writes `experiments_v5/footprint_strict_488_summary.json` with `{audited: {V, CV, RP, A}, total_audited, total_blocks}`.
  - Add `tests/test_footprint_strict_488.py` that runs the driver, asserts the CSV has 488 rows, the JSON sums to 488, and `audited.V + audited.CV + audited.RP > 0`.
success_criterion: `pytest tests/test_footprint_strict_488.py -x` exits 0 AND `experiments_v5/footprint_strict_488.csv` exists with ≥488 data rows AND `experiments_v5/footprint_strict_488_summary.json` parses and contains an `audited` key whose `V+CV+RP+A` sums to the audited-class row count.
fallback_message: If the audited-operator list cannot be located or the eval driver cannot be invoked headlessly within the budget, the subagent should print `INFEASIBLE: footprint-strict rerun blocked` and make no commits so the harness reverts cleanly.

### Candidate 2: Stratified random resample of the 371 Verified tied-weight population with Wilson CI
goal: Replace the selection-biased shortest-LoC-first 25-row stub-mocked validation with a stratified random sample (stratified by handler family) of size ≥80 over the 371 Verified tied-weight rows, run the existing stub-mocked runtime check, and emit a tightened Wilson 95% CI artifact. This directly addresses an active obligation and lifts Soundness 2→3 by replacing a biased point estimate with a defensible stratified estimate.
plan:
  - Find the existing 371 Verified tied-weight row list (likely under `experiments_v5/` or `reproducibility/`) and the stub-mocked runtime harness that produced the original `0/25`.
  - Implement `experiments_v5/stratified_resample_371.py` that loads the 371 rows, groups by handler-family tag, draws a seeded stratified random sample of ≥80 rows (proportional allocation, min 2 per stratum), runs the stub-mocked harness on each, and writes `experiments_v5/stratified_resample_371.csv` plus `experiments_v5/stratified_resample_371_wilson.json` with `{n, k_silently_incorrect, wilson_lo, wilson_hi, per_stratum: {...}}`.
  - Use a fixed seed (e.g. 20260430) so results are reproducible.
  - Add `tests/test_stratified_resample_371.py` that runs the script and asserts: CSV has ≥80 rows, ≥3 distinct strata represented, and the Wilson upper bound in the JSON is < 13.32 (i.e. tighter than the original interval).
success_criterion: `pytest tests/test_stratified_resample_371.py -x` exits 0 AND `experiments_v5/stratified_resample_371_wilson.json` exists with `wilson_hi < 0.1332` AND `n >= 80`.
fallback_message: If the 371-row population list or the stub-mocked harness cannot be invoked headlessly within budget, the subagent should print `INFEASIBLE: stratified resample blocked` and make no commits so the harness reverts cleanly.


Changes   +0 -0
Requests  7.5 Premium (30s)
Tokens    ↑ 71.7k • ↓ 1.7k • 52.4k (cached)
