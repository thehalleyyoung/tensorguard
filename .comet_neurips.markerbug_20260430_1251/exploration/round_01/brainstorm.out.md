● Read brainstorm.prompt.md
  │ .comet_neurips/exploration/round_01/brainstorm.prompt.md
  └ 108 lines read

```
### Candidate 1: Mechanized completeness witness for Theorem soundness(ii) in Lean
goal: Close the most damaging soundness gap by mechanizing, in Lean, a model-extraction lemma that turns a Z3-style satisfying assignment of the negated obligation into a concrete heap σ ⊨ Γ that triggers the runtime exception, for at least three operators (matmul, cat, reshape) in the existing 17-op DSL. This directly targets reviewer item (ii) and would lift Soundness from 2 → 4 by converting an axiomatic gap into checked Lean proof.
plan:
  - Inspect lean/TensorGuard/V5OperatorRules.lean and Soundness.lean to find the existing per-op preservation lemmas and operational-semantics definitions.
  - Add a new file lean/TensorGuard/RPCompleteness.lean defining `RPWitness`, `heapOfModel`, and a theorem `rp_completeness_matmul`, `rp_completeness_cat`, `rp_completeness_reshape` stating: if the rule rejects, then there exists a heap σ ⊨ Γ on which the small-step semantics raises a ShapeError at the witnessed subterm. Use the existing deterministic small-step lemma for these three ops; if absent, prove a local determinism lemma restricted to these constructors.
  - Wire the new file into lakefile.lean / TensorGuard.lean as an import.
  - Run `cd lean && lake build TensorGuard.RPCompleteness` and ensure no `sorry` in the new file (`grep -n sorry lean/TensorGuard/RPCompleteness.lean | grep -v -- '--' || true` returns empty).
success_criterion: `cd lean && lake build TensorGuard.RPCompleteness 2>&1 | tee /tmp/rpc.log` exits 0 AND `grep -E '^\s*sorry' lean/TensorGuard/RPCompleteness.lean` returns no matches AND `grep -c 'theorem rp_completeness_' lean/TensorGuard/RPCompleteness.lean` outputs `3` or greater.
fallback_message: If the existing Lean operational semantics does not expose enough of the heap/model interface to close even one of the three lemmas in 10 minutes, write a single line `EXTENSION_INFEASIBLE: rp_completeness requires extending operational-semantics heap interface` to stdout and make no commits.

### Candidate 2: Sub-catalogue decomposition of the 53/60 historical-corpus RP count with reproducible script
goal: Answer the reviewer's headline question by adding an executable script that classifies every refutation in the 60-bug historical corpus by which sub-catalogue (Cat_sound / Cat_audit / Cat_tested / mixed) the firing handler chain lies in, emitting a CSV and aggregate counts. This directly addresses two open obligations (sub-catalogue decomposition + 5/488 thin-anchor concern) and would lift Contribution from 3 → 4 and Soundness from 2 → 3 by giving the soundness-grade RP count a defensible empirical floor.
plan:
  - Locate the historical bug corpus driver (likely under benchmarks/ or real_benchmarks/) and the handler-catalogue partition (search src/ for `Cat_sound`, `cat_sound`, `sound_handlers`, or the soundness table loader).
  - Implement scripts/decompose_rp_by_catalogue.py that: (a) re-runs TensorGuard over the 60 historical bugs, (b) records the ordered list of handler IDs that fired on the refutation path for each RP verdict, (c) classifies each row as `sound`, `audit`, `tested`, or `mixed` based on membership of every handler in the firing chain, (d) writes results/rp_catalogue_decomposition.csv with columns `bug_id, verdict, firing_handlers, partition`, (e) prints a summary line `SOUND=<n> AUDIT=<n> TESTED=<n> MIXED=<n> TOTAL_RP=<n>`.
  - If the analyser does not currently log firing handler IDs per refutation, add a minimal trace hook in src/model_checker.py guarded by an env var `TG_TRACE_HANDLERS=1`.
  - Add a pytest tests/test_rp_decomposition.py that imports the script, runs it on the corpus, and asserts the CSV has ≥ 50 rows and the summary counts sum to TOTAL_RP.
success_criterion: `TG_TRACE_HANDLERS=1 python scripts/decompose_rp_by_catalogue.py && pytest -x tests/test_rp_decomposition.py` exits 0 AND `wc -l < results/rp_catalogue_decomposition.csv` is ≥ 51 (header + ≥50 data rows) AND the script's final stdout line matches the regex `^SOUND=[0-9]+ AUDIT=[0-9]+ TESTED=[0-9]+ MIXED=[0-9]+ TOTAL_RP=[0-9]+$`.
fallback_message: If the historical bug corpus is not reachable in-repo or the handler-firing chain cannot be extracted in 10 minutes, write `EXTENSION_INFEASIBLE: historical corpus or handler trace not accessible` to stdout and make no commits.
```


Changes   +0 -0
Requests  7.5 Premium (39s)
Tokens    ↑ 67.4k • ↓ 2.0k • 48.2k (cached)
