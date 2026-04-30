● Read brainstorm.prompt.md
  │ .comet_neurips/exploration/round_01/brainstorm.prompt.md
  └ 108 lines read

### Candidate 1: Close the matmul and broadcast_add Lean composition lemmas
goal: Replace `Axiom ax_operator_agnostic_witness` (as it applies to `matmul` and `broadcast_add`) with closed Lean lemmas `applyOpExt_sound_matmul` and `applyOpExt_sound_broadcast_add`, so that Theorem `thm:ag-sound` becomes an unconditional 17-of-17 theorem rather than 15+2-axiom. Lifts Soundness from 2 → 3 and removes the single most cited blocker in the borderline reasons.
plan:
- Inspect `lean/TensorGuard/*.lean` and `lean/TheoryCombination.lean`; locate the existing `applyOpExt_sound_*` lemmas (e.g. for `conv2d`, `einsum`) and the operator-agnostic witness axiom.
- Mirror the `conv2d`/`einsum` proof structure for `matmul` (last-dim contraction: `(...,m,k) ⊗ (...,k,n) → (...,m,n)` with broadcast on the leading dims) and for `broadcast_add` (NumPy broadcasting: pairwise dim equality or one is 1).
- Rewire `thm_ag_sound` so the matmul/broadcast_add cases call the new lemmas instead of the agnostic axiom; if the axiom is still referenced by other ops, keep it but narrow its statement.
- Run `cd lean && lake build` and `grep -nE "(:= sorry|by sorry|^[[:space:]]*sorry$|axiom ax_operator_agnostic_witness)" lean/**/*.lean` over the new code to confirm no `sorry` and no residual axiom use for those two operators.
success_criterion: `cd lean && lake build 2>&1 | tee /tmp/lake.log` exits 0 AND `grep -c "theorem applyOpExt_sound_matmul\|theorem applyOpExt_sound_broadcast_add\|lemma applyOpExt_sound_matmul\|lemma applyOpExt_sound_broadcast_add" lean/**/*.lean` returns ≥ 2 AND `grep -L "ax_operator_agnostic_witness" lean/TensorGuard/MatmulSound.lean lean/TensorGuard/BroadcastAddSound.lean` lists both files (i.e. neither new file invokes the axiom).
fallback_message: If the Lean `Shape`/`applyOpExt` definitions cannot be lined up with a tractable matmul broadcast proof inside the budget, emit `INFEASIBLE: matmul/broadcast_add Lean lemma blocked by <one-line reason>` to stdout and revert all `lean/` edits via `git checkout -- lean/`.

### Candidate 2: torch.compile head-to-head benchmark on the unfiltered post-freeze N=15 sample
goal: Build a reproducible head-to-head benchmark harness that runs TensorGuard, `torch.compile` (FakeTensorMode + dynamo export), and Pytea on the pre-registered post-freeze N=15 unfiltered corpus, emitting a per-bug verdict-pair table plus the BH-adjusted Fisher p-values. Directly answers the "abandoned Pytea baseline" weakness and the "per-bug verdict pair against torch.compile" reviewer question; lifts Contribution by ~1 because it converts a buried paragraph into a load-bearing experiment.
plan:
- Locate the pre-registered post-freeze N=15 sample (search `benchmarks/`, `real_benchmarks/`, `experiments_v5/` for `post_freeze`, `n15`, `unfiltered`); if absent, materialise it from the corpus mining script using the documented seed/freeze date.
- Write `benchmarks/torch_compile_headtohead.py` that, for each of the 15 bug snippets, runs (a) `tensorguard` in unconditional mode, (b) `torch.compile(mode="reduce-overhead")` + `FakeTensorMode` shape inference wrapped in try/except to capture pre-runtime errors, and (c) the existing Pytea harness; record `{bug_id, tg_verdict, compile_verdict, pytea_verdict, ground_truth}` to `benchmarks/results/headtohead_n15.csv`.
- Compute pairwise McNemar exact p-values and BH-adjusted Fisher p-values across the three tools; write to `benchmarks/results/headtohead_n15_stats.json`.
- Add `tests/test_headtohead_n15.py` asserting CSV has exactly 15 rows, all three verdict columns are populated for every row, and the stats JSON parses with the three required keys.
success_criterion: `python -m benchmarks.torch_compile_headtohead --out benchmarks/results/headtohead_n15.csv && pytest tests/test_headtohead_n15.py -q` exits 0 AND `python -c "import csv; rows=list(csv.DictReader(open('benchmarks/results/headtohead_n15.csv'))); assert len(rows)==15 and all(r['tg_verdict'] and r['compile_verdict'] and r['pytea_verdict'] for r in rows)"` exits 0.
fallback_message: If the post-freeze N=15 sample cannot be reconstructed or `torch.compile` cannot be imported in this environment within the budget, emit `INFEASIBLE: headtohead-n15 blocked by <one-line reason>` to stdout and revert all new files via `git clean -fd benchmarks/ tests/ && git checkout -- benchmarks/ tests/`.


Changes   +0 -0
Requests  7.5 Premium (40s)
Tokens    ↑ 67.3k • ↓ 2.4k • 48.2k (cached)
