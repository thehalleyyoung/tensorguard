● Read brainstorm.prompt.md
  │ .comet_neurips/exploration/round_04/brainstorm.prompt.md
  └ 1 line read

● check file size and dir (shell)
  │ wc -l /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguar
  │ d/.comet_neurips/exploration/round_04/brainstorm.prompt.md && ls
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/.com
  │ et_neurips/exploration/round_04/
  └ 3 lines...

● Read brainstorm.prompt.md
  │ .comet_neurips/exploration/round_04/brainstorm.prompt.md
  └ L1:112 (112 lines read)

### Candidate 1: Deterministic 60→34 fragment-fair filter with per-bug audit CSV
goal: Ship an end-to-end script that materialises the deterministic filter rule mapping the 60-bug historical corpus to the 34-bug fragment-fair Pytea head-to-head, and emits a single appendix-grade CSV with one row per bug containing `(bug_id, included_in_34, exclusion_reason, tg_verdict, pytea_verdict)`. This directly discharges the round-4 "auditable filter" obligation and the round-5 ESCALATED "ship an artifact, not paper text" obligation, and should lift Soundness +1 and Presentation +1 by making the only frequentist-significant headline number reproducible from a script.
plan:
  - Inspect `bugclasses.jsonl`, `benchmarks/`, and any existing pytea comparison code to locate the 60-bug corpus and existing per-bug TG/Pytea verdict records.
  - Implement `reproducibility/build_fragment_fair_filter.py` that (a) iterates all 60 bugs, (b) applies a deterministic rule based on the operator-fragment membership already encoded in the Lean operator registry / handler catalogue, (c) writes `reproducibility/fragment_fair_audit.csv` with the 5 columns above, (d) prints summary counts and the McNemar 2x2 table.
  - Add `tests/test_fragment_fair_filter.py` asserting: exactly 60 rows in CSV, exactly 34 with `included_in_34=True`, every excluded row has a non-empty `exclusion_reason` drawn from a closed enumeration, and the recomputed `(TG_wins, Pytea_wins, both, neither)` McNemar table matches the published 32/34 vs 25/34 counts (or, if it does not, the script writes the actual recomputed counts and the test asserts the script's own claimed numbers — never silently fudge).
  - Wire the script into `verify_neurips_revision.py` as an additional check.
success_criterion: `python reproducibility/build_fragment_fair_filter.py && pytest tests/test_fragment_fair_filter.py -x` exits 0 AND `wc -l reproducibility/fragment_fair_audit.csv` reports exactly 61 lines (60 data + 1 header) AND `awk -F, 'NR>1 && $2=="True"' reproducibility/fragment_fair_audit.csv | wc -l` prints exactly 34.
fallback_message: If the underlying per-bug Pytea verdict log cannot be located or reconstructed within the budget, emit `FRAGMENT_FAIR_FILTER_INFEASIBLE: <one-line root cause>` to stdout and make no other changes so the harness reverts cleanly.

### Candidate 2: Extend Lean operator-soundness mechanisation from 17 to ≥25 operators, covering majority CV traffic
goal: Grow the Lean 4 mechanised operator fragment from the current 17 operators to at least 25 by adding `applyOp_sound_*` lemmas (sorry-free) for the 8 highest-CV-traffic handlers currently outside the fragment, then export an updated operator registry JSON consumed by the Python analyser and recompute the "fraction of CV verdicts entirely under Lean-witnessed handlers" headline number. Directly substantiates C2 and answers the round-4 question about the 79-vs-17 gap; expected to lift Contribution +1 and Soundness +1.
plan:
  - From `src/model_checker.py` handler table and any CV-verdict log under `experiments_v5/` or `real_benchmarks/`, rank the 79 handlers by CV-verdict count on the 488-block corpus and pick the top 8 not already in the Lean fragment.
  - For each of those 8 ops, add a `applyOp_sound_<op>` theorem in `lean/` mirroring the existing 17 lemmas' shape (input refinement → output refinement under the same DSL); reuse existing tactics, no new axioms, no `sorry`.
  - Regenerate the operator-registry JSON (whatever script currently exports it) and add a Python check that every handler tagged `lean_audited=True` has a corresponding theorem name in the JSON.
  - Add `reproducibility/cv_lean_coverage.py` that recomputes and prints `<n_cv_in_fragment>/128` after the registry update, and writes the number to `reproducibility/cv_lean_coverage.txt`.
success_criterion: `cd lean && lake build` exits 0 AND `grep -c "^theorem applyOp_sound_" lean/**/*.lean` reports >=25 AND `grep -c "sorry" $(grep -rl "applyOp_sound_" lean/)` reports 0 AND `python reproducibility/cv_lean_coverage.py` writes a file `reproducibility/cv_lean_coverage.txt` containing an integer strictly greater than the pre-existing baseline (record baseline in the same script before the update).
fallback_message: If `lake build` cannot be reached in the budget or the existing Lean DSL lacks primitives for the chosen ops, emit `LEAN_EXTENSION_INFEASIBLE: <one-line root cause>` to stdout, revert any partial Lean edits via `git checkout -- lean/`, and exit non-zero so the harness reverts cleanly.


Changes   +0 -0
Requests  7.5 Premium (1m 2s)
Tokens    ↑ 142.6k • ↓ 3.2k • 119.5k (cached)
